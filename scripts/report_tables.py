"""
report_tables.py

Turns the output of scripts/benchmark.py into the LaTeX of the report's result tables.

    python scripts/report_tables.py --results results/benchmark/results.json
    python scripts/report_tables.py --results results/benchmark/results.json --out results/benchmark/tables.tex

Prints (1) the rows of the main table (static reference, plain LTX, adapter), (2) a per-motion table, and
(3) the counts for the caption. The static reference is the ATE of a point that never moves from the start
of a straight path of length L: the distance grows linearly from 0 to L, so its mean is L / 2.
"""

import argparse
import json

COLS = [("ate", "{:.3f}"), ("epe_final", "{:.3f}"), ("direction_cos", "{:.2f}"), ("progress", "{:.2f}"),
        ("clip_image_sim", "{:.3f}"), ("clip_temporal", "{:.3f}")]


def fmt(row, key, spec):
    return spec.format(row[key]) if key in row else "--"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/benchmark/results.json")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    overall = data["summary"]["overall"]
    per_motion = data["summary"]["per_motion"]
    records = data["records"]
    length = data["args"]["length"]
    motions = list(per_motion["adapter"])

    out = []
    out.append("% main table rows (columns: ATE, final err., dir. cos, progress, img sim, temporal)")
    out.append(f"Static reference & {length / 2:.3f} & -- & -- & -- & -- & -- \\\\")
    for label, key in (("Plain LTX I2V", "baseline"), (f"Adapter ({data['adapter_mode']})", "adapter")):
        cells = " & ".join(fmt(overall[key], k, s) for k, s in COLS)
        out.append(f"{label} & {cells} \\\\")

    images = sorted({r["image"] for r in records["adapter"]})
    out.append("")
    out.append(f"% caption: {len(images)} images ({', '.join(images)}), {len(motions)} motions "
               f"({', '.join(motions)}), {len(records['adapter'])} adapter videos and {len(images)} baseline videos, "
               f"path length {length}, motion guidance {data['args'].get('motion_guidance')}")

    out.append("")
    out.append("% per-motion table")
    out.append("\\begin{tabular}{lcccccc}")
    out.append("\\hline")
    out.append("Motion & ATE base & ATE adapter & Dir. cos base & Dir. cos adapter & Progress base & Progress adapter \\\\")
    out.append("\\hline")
    for m in motions:
        b, a = per_motion["baseline"][m], per_motion["adapter"][m]
        cells = [fmt(b, "ate", "{:.3f}"), fmt(a, "ate", "{:.3f}"),
                 fmt(b, "direction_cos", "{:.2f}"), fmt(a, "direction_cos", "{:.2f}"),
                 fmt(b, "progress", "{:.2f}"), fmt(a, "progress", "{:.2f}")]
        out.append(f"{m} & " + " & ".join(cells) + " \\\\")
    out.append("\\hline")
    out.append("\\end{tabular}")

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
