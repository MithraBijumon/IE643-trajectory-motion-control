"""
caption_clips.py
Per-clip text prompts for LTX: caption each clip's first frame (image.png) with a small vision-language model.

Compare models on a few clips first (writes compare.md + compare.jpg into --out):
    python scripts/caption_clips.py --models blip,florence,qwen --num 6
Then caption every clip with the one you picked (writes <clip>/caption.txt):
    python scripts/caption_clips.py --models qwen --write --data_root data/dataset_got10k
Out-of-domain test images (for benchmark.py --ood_dir):
    python scripts/caption_clips.py --models qwen --write --images_dir path/to/images
"""

import argparse
import gc
import os
import re
import textwrap

import torch
from PIL import Image, ImageDraw

QWEN_ASK = (
    "Describe this image for a video-generation prompt in 2-3 sentences (about 40-60 words). Name the main subject and "
    "its colours or clothing, then the setting, background and lighting. Describe only what you can clearly see: do not "
    "mention camera or subject motion, text overlays, or anything you are unsure about."
)
SUFFIX = " The subject moves smoothly along its path, keeping its exact shape and colours, while the background stays stable and sharp."
FILLER = re.compile(r"^(in this (frame|image|scene|picture),?|the (video|image|picture|frame) (features|shows|depicts|captures)|this (image|frame|picture) (shows|features|depicts))\s*,?\s*", re.I)


STILL = re.compile(r"\b(stationary|static|motionless|standing still|not moving|remains? still)\b", re.I)


def clean(text):
    text = " ".join(text.split())
    text = FILLER.sub("", text)
    kept = [s for s in re.split(r"(?<=[.!?])\s+", text) if not STILL.search(s)]  # a still-life claim would fight the trajectory
    text = " ".join(kept) if kept else text
    if text and text[-1] not in ".!?" and "." in text:
        text = text[: text.rindex(".") + 1]  # a generation cut off mid-sentence: keep the complete sentences
    return (text[:1].upper() + text[1:]).rstrip(". ") + "."
IDS = {
    "blip": "Salesforce/blip-image-captioning-large",
    "florence": "microsoft/Florence-2-large",
    "qwen": "Qwen/Qwen2-VL-2B-Instruct",
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float16 if DEV == "cuda" else torch.float32


def load_blip():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    proc = BlipProcessor.from_pretrained(IDS["blip"])
    model = BlipForConditionalGeneration.from_pretrained(IDS["blip"], torch_dtype=DT).to(DEV).eval()

    def caption(img):
        x = proc(images=img, return_tensors="pt").to(DEV, DT)
        out = model.generate(**x, max_new_tokens=60, num_beams=3)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    return caption


def load_florence():
    from transformers import AutoModelForCausalLM, AutoProcessor

    kw = dict(trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(IDS["florence"], **kw)
    model = AutoModelForCausalLM.from_pretrained(IDS["florence"], torch_dtype=DT, attn_implementation="eager", **kw).to(DEV).eval()
    task = "<MORE_DETAILED_CAPTION>"

    def caption(img):
        x = proc(text=task, images=img, return_tensors="pt").to(DEV, DT)
        out = model.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=200, num_beams=3)
        text = proc.batch_decode(out, skip_special_tokens=False)[0]
        return proc.post_process_generation(text, task=task, image_size=img.size)[task].strip()

    return caption


def load_qwen():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(IDS["qwen"], min_pixels=128 * 28 * 28, max_pixels=448 * 28 * 28)
    model = Qwen2VLForConditionalGeneration.from_pretrained(IDS["qwen"], torch_dtype=DT).to(DEV).eval()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QWEN_ASK}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    proc.tokenizer.padding_side = "left"  # batched generation: pad on the left so every row ends where the prompt ends

    def caption_batch(imgs):
        x = proc(text=[prompt] * len(imgs), images=list(imgs), return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**x, max_new_tokens=110, do_sample=False)
        return [t.strip() for t in proc.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)]

    def caption(img):
        return caption_batch([img])[0]

    caption.batch = caption_batch
    return caption


LOADERS = {"blip": load_blip, "florence": load_florence, "qwen": load_qwen}


def clip_dirs(root, splits=("val", "train")):
    return [os.path.join(root, s, n) for s in splits if os.path.isdir(os.path.join(root, s))
            for n in sorted(os.listdir(os.path.join(root, s))) if os.path.exists(os.path.join(root, s, n, "image.png"))]


def compare(args):
    clips = clip_dirs(args.data_root, ("val",))
    pick = [c for c in clips if os.path.basename(c) in ("GOT-10k_Train_000028", "GOT-10k_Train_000074")]
    pick += [c for c in clips if c not in pick][: max(args.num - len(pick), 0)]
    images = [Image.open(os.path.join(c, "image.png")).convert("RGB") for c in pick]
    names = [os.path.basename(c) for c in pick]
    results = {}
    for m in args.models.split(","):
        try:
            cap = LOADERS[m]()
            results[m] = [clean(cap(im)) + SUFFIX for im in images]
        except Exception as e:  # a model that does not load on this machine should not stop the others
            results[m] = [f"[failed: {type(e).__name__}: {str(e)[:160]}]"] * len(images)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"done: {m}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "compare.md"), "w", encoding="utf-8") as f:
        for i, n in enumerate(names):
            f.write(f"## {n}\n" + "".join(f"- **{m}**: {results[m][i]}\n" for m in results) + "\n")
    w, tw = 300, 560
    rows = []
    for i, im in enumerate(images):
        sheet = Image.new("RGB", (w + tw * len(results), 330), "white")
        sheet.paste(im.resize((w, int(w * im.height / im.width))), (0, 0))
        d = ImageDraw.Draw(sheet)
        for j, m in enumerate(results):
            d.text((w + tw * j + 6, 4), f"{m}  ({names[i]})", fill="red")
            d.text((w + tw * j + 6, 22), "\n".join(textwrap.wrap(results[m][i], 82)), fill="black")
        rows.append(sheet)
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save(os.path.join(args.out, "compare.jpg"), quality=90)
    print(f"wrote {args.out}/compare.md and compare.jpg")


def write(args):
    cap = LOADERS[args.models.split(",")[0]]()
    if args.images_dir:   # loose images: <name>.jpg -> <name>.txt (the format benchmark.py --ood_dir reads)
        names = [f for f in sorted(os.listdir(args.images_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) and not os.path.splitext(f)[0].endswith("_mask")]
        pairs = [(os.path.join(args.images_dir, f), os.path.join(args.images_dir, os.path.splitext(f)[0] + ".txt")) for f in names]
    else:                 # dataset clips: <clip>/image.png -> <clip>/caption.txt
        pairs = [(os.path.join(c, "image.png"), os.path.join(c, "caption.txt")) for c in clip_dirs(args.data_root)]
    todo = [(i, o) for i, o in pairs if args.overwrite or not os.path.exists(o)]
    step = args.batch if hasattr(cap, "batch") else 1
    for i in range(0, len(todo), step):
        group = todo[i:i + step]
        imgs = [Image.open(src).convert("RGB") for src, _ in group]
        texts = cap.batch(imgs) if step > 1 else [cap(imgs[0])]
        for (_, dst), t in zip(group, texts):
            text = clean(t) + SUFFIX
            open(dst, "w", encoding="utf-8").write(text)
        print(f"{i + len(group)}/{len(todo)} {os.path.basename(group[0][1])}: {text[:100]}", flush=True)
    print("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="blip,florence,qwen")
    p.add_argument("--data_root", default="data/dataset_got10k")
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--out", default="results/caption_compare")
    p.add_argument("--write", action="store_true", help="caption every clip into <clip>/caption.txt (first model only)")
    p.add_argument("--images_dir", default=None, help="caption loose images in this folder into <name>.txt (with --write)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--batch", type=int, default=8, help="images per generate call (qwen)")
    a = p.parse_args()
    write(a) if a.write else compare(a)
