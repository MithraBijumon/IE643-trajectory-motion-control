import os
import torch
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diffusers import LTXConditionPipeline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import TrajectoryDataset
from src.conditioning import rasterize_trajectories
from src.condition_encoder import ConditionEncoder
from src.adapter import MotionAdapter


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

MODEL_ID = "Lightricks/LTX-Video-0.9.7-dev"

DATA_ROOT = os.environ.get("DATA_ROOT", "data/dataset")

NUM_FRAMES = 33
HEIGHT = 480
WIDTH = 864

BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 10

DEVICE = "cuda"
DTYPE = torch.bfloat16

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------
# LTX
# ---------------------------------------------------------

print("Loading LTX...")

pipe = LTXConditionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
)

pipe.to(DEVICE)

vae = pipe.vae
transformer = pipe.transformer


# ---------------------------------------------------------
# Freeze LTX
# ---------------------------------------------------------

vae.requires_grad_(False)
transformer.requires_grad_(False)

vae.eval()
transformer.eval()


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

dataset = TrajectoryDataset(
    root=DATA_ROOT,
    split="train",
    num_frames=NUM_FRAMES,
    random_crop=True,
    load_video=True,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

print("Training samples:", len(dataset))


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

condition_encoder = ConditionEncoder()

adapter = MotionAdapter(dim=128)

condition_encoder = condition_encoder.to(DEVICE)

adapter = adapter.to(
    DEVICE,
    dtype=DTYPE,
)


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    list(condition_encoder.parameters())
    + list(adapter.parameters()),
    lr=LR,
)

print("Trainable parameters:")

print(
    "Condition encoder:",
    sum(p.numel() for p in condition_encoder.parameters()
        if p.requires_grad)
)

print(
    "Adapter:",
    sum(p.numel() for p in adapter.parameters()
        if p.requires_grad)
)


# ---------------------------------------------------------
# LTX packing
# ---------------------------------------------------------

def pack_latents(latents):
    """
    [B,C,F,H,W] -> [B,F*H*W,C]

    LTX uses patch_size=1 and patch_size_t=1
    for this model.
    """

    B, C, F_, H_, W_ = latents.shape

    return latents.permute(
        0, 2, 3, 4, 1
    ).reshape(
        B,
        F_ * H_ * W_,
        C,
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

for epoch in range(EPOCHS):

    adapter.train()
    condition_encoder.train()

    total_loss = 0.0

    for step, batch in enumerate(loader):

        video = batch["video"].to(
            DEVICE,
            dtype=DTYPE,
        )

        video = video.permute(0, 2, 1, 3, 4)

        trajectory = batch["trajectory"].to(
            DEVICE,
            dtype=torch.float32,
        )

        visible = batch["visible"].to(
            DEVICE
        )

        B = video.shape[0]

        # -------------------------------------------------
        # 1. Encode real video into LTX latent space
        # -------------------------------------------------

        with torch.no_grad():

            video_latent = vae.encode(
                video
            ).latent_dist.sample()

        print(
            "video latent:",
            video_latent.shape
        )

        # -------------------------------------------------
        # 2. Create trajectory condition
        # -------------------------------------------------

        condition = rasterize_trajectories(
            trajectory,
            visible,
            height=video.shape[-2],
            width=video.shape[-1],
        )

        # condition:
        # [B,F,3,H,W]

        condition = condition.permute(
            0, 2, 1, 3, 4
        )

        condition = condition.float()

        # now:
        # [B,3,F,H,W]

        # -------------------------------------------------
        # 3. Encode trajectory condition
        # -------------------------------------------------

        condition_latent = condition_encoder(
            condition
        )

        print(
            "condition latent:",
            condition_latent.shape
        )

        # -------------------------------------------------
        # 4. Make sure condition has same latent grid
        # -------------------------------------------------

        if condition_latent.shape[2:] != video_latent.shape[2:]:

            condition_latent = F.interpolate(
                condition_latent,
                size=video_latent.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        # -------------------------------------------------
        # 5. Pack both into LTX tokens
        # -------------------------------------------------

        video_tokens = pack_latents(
            video_latent
        )

        condition_tokens = pack_latents(
            condition_latent
        )

        print(
            "video tokens:",
            video_tokens.shape
        )

        print(
            "condition tokens:",
            condition_tokens.shape
        )

        # -------------------------------------------------
        # 6. Generate diffusion noise
        # -------------------------------------------------

        noise = torch.randn_like(
            video_latent
        )

        # Random diffusion timestep
        timestep = torch.randint(
            0,
            1000,
            (B,),
            device=DEVICE,
        )

        # -------------------------------------------------
        # 7. Add noise
        # -------------------------------------------------

        # Simple first version.
        # This will be replaced by the exact LTX scheduler
        # training formulation once the forward pass works.

        alpha = 1.0 - timestep.float() / 1000.0

        alpha = alpha.view(
            B, 1, 1, 1, 1
        ).to(DTYPE)

        noisy_latent = (
            alpha.sqrt() * video_latent
            + (1.0 - alpha).sqrt() * noise
        )

        # -------------------------------------------------
        # 8. Pack noisy latent
        # -------------------------------------------------

        noisy_tokens = pack_latents(
            noisy_latent
        )

        # -------------------------------------------------
        # 9. Apply trajectory adapter
        # -------------------------------------------------
        print(condition_tokens.dtype, noisy_tokens.dtype)
        adapted_tokens = adapter(
            noisy_tokens,
            condition_tokens,
        )

        # -------------------------------------------------
        # 10. LTX timestep format
        # -------------------------------------------------

        timestep_input = (
            timestep.float()
            .view(B, 1)
            .expand(
                B,
                adapted_tokens.shape[1],
            )
        )

        # -------------------------------------------------
        # 11. Text conditioning
        # -------------------------------------------------

        # For the first experiment we use an empty prompt.
        #
        # IMPORTANT:
        # We will replace this with the proper LTX prompt
        # embedding path after this forward pass works.

        prompt_embeds = torch.zeros(
            B,
            256,
            4096,
            device=DEVICE,
            dtype=DTYPE,
        )

        prompt_attention_mask = torch.ones(
            B,
            256,
            device=DEVICE,
            dtype=torch.long,
        )

        # -------------------------------------------------
        # 12. LTX forward pass
        # -------------------------------------------------

        noise_pred = transformer(
            hidden_states=adapted_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep_input,
            encoder_attention_mask=prompt_attention_mask,
            return_dict=False,
        )[0]

        # -------------------------------------------------
        # 13. Diffusion loss
        # -------------------------------------------------

        loss = F.mse_loss(
            noise_pred.float(),
            noise_tokens.float()
            if False else
            pack_latents(noise).float(),
        )

        # -------------------------------------------------
        # 14. Backprop
        # -------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(condition_encoder.parameters())
            + list(adapter.parameters()),
            1.0,
        )

        optimizer.step()

        total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Step {step+1}/{len(loader)} "
            f"Loss: {loss.item():.6f} "
            f"Adapter scale: {adapter.scale.item():.6f}"
        )

    avg_loss = total_loss / len(loader)

    print(
        f"\nEpoch {epoch+1} complete "
        f"| average loss = {avg_loss:.6f}\n"
    )

    torch.save(
        {
            "epoch": epoch + 1,
            "condition_encoder": condition_encoder.state_dict(),
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        },
        os.path.join(
            SAVE_DIR,
            f"checkpoint_{epoch+1}.pt",
        ),
    )