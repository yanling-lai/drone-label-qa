import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_visdrone import load_annotations

TINY_AREA = 16          # boxes smaller than 4x4 px
DUPLICATE_IOU = 0.9     # same-class boxes overlapping more than this are considered duplication

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a and every box in b. Box format: [x1, y1, x2, y2]."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]) # output: NxM(i, j) = max_x1(a[i], b[j])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)

def find_duplicates(df):
    dups = []
    for (image_id, cls), g in df.groupby(["image_id", "class_name"]):
        if len(g) < 2: # impossible to have duplicates
            continue
        boxes = g[["x", "y", "w", "h"]].to_numpy(float).copy()
        boxes[:, 2:] += boxes[:, :2] # [x, y, w, h] -> [x1, y1, x2, y2]
        ious = np.triu(iou_matrix(boxes, boxes), k=1) # k: skip steps above diagonal
        for i, j in zip(*np.where(ious > DUPLICATE_IOU)):
            dups.append((image_id, cls, g.index[i], g.index[j], round(ious[i, j], 3)))
    return dups

def main(split_dir, out_dir="outputs/qa"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_annotations(split_dir)
    df["area"] = df["w"] * df["h"]

    # Keep only real object labels (score == 1, category 1-10) for the object checks
    ignored = df[(df["score"] == 0) | (df["category"].isin([0, 11]))]
    obj = df.drop(ignored.index)

    checks = {
        "zero_or_negative_size": obj[(obj["w"] <= 0) | (obj["h"] <= 0)],
        "out_of_bounds": obj[(obj["x"] < 0) | (obj["y"] < 0) | (obj["x"] + obj["w"] > obj["img_w"]) | (obj["y"] + obj["h"] > obj["img_h"])],
        f"tiny_area_lt_{TINY_AREA}px": obj[obj["area"] < TINY_AREA],
    }
    dups = find_duplicates(obj)
    lines = ["# Label QA report", "",
             f"- Images: {df['image_id'].nunique()}",
             f"- Object boxes checked: {len(obj)}",
             f"- Excluded regions ('ignored' / 'others'): {len(ignored)}",
             "",
             "| Check | Flagged boxes | Share |",
             "| --- | --- | --- |"]
    for name, flagged in checks.items():
        lines.append(f"| {name} | {len(flagged)} | {len(flagged) / len(obj):.2%} |")
        flagged.to_csv(out / f"{name}.csv")
    lines.append(f"| near_duplicate_iou_gt_{DUPLICATE_IOU} | {len(dups)} pairs | - |")
    pd.DataFrame(dups, columns=["image_id", "class", "box_i", "box_j", "iou"]).to_csv(out / "near_duplicates.csv", index=False)
    lines += ["", "## Class distribution", "", "| Class | Boxes | Share |", "| --- | --- | --- |"]
    counts = obj["class_name"].value_counts()
    for cls, n in counts.items():
        lines.append(f"| {cls} | {n} | {n / len(obj):.1%} |")
    (out / "qa_report.md").write_text("\n".join(lines))

    # Plot 1: class distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.sort_values().plot.barh(ax=ax, color="#4C72B0")
    ax.set_xlabel("Number of boxes")
    ax.set_title("VisDrone val: class distribution")
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)

    # Plot 2: box size distribution (sqrt of area = side length of an equal-area square)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.sqrt(obj["area"].clip(lower=1)), bins=60, color="#55A868")
    ax.axvline(32, color="k", ls="--", lw=1)
    ax.text(33, ax.get_ylim()[1] * 0.9, "COCO 'small' < 32 px", fontsize=8)
    ax.set_xlabel("Box size, sqrt(area) in pixels")
    ax.set_ylabel("Number of boxes")
    ax.set_title("VisDrone val: box size distribution")
    fig.tight_layout()
    fig.savefig(out / "box_size_distribution.png", dpi=150)

    small_share = (obj["area"] < 32 * 32).mean()
    print("\n".join(lines))
    print(f"\nShare of 'small' boxes (< 32x32 px): {small_share:.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated label QA checks on VisDrone ground truth.")
    parser.add_argument("split_dir", help="dataset split folder, e.g. data/VisDrone2019-DET-val")
    parser.add_argument("--out", default="outputs/qa", help="output folder")
    args = parser.parse_args()
    main(args.split_dir, args.out)