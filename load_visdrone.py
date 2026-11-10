from pathlib import Path

import pandas as pd
from PIL import Image

# VisDrone category ids -> names
CATEGORIES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car",
    5: "van", 6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus",
    10: "motor", 11: "others",
}
COLUMNS = ["x", "y", "w", "h", "score", "category", "truncation", "occlusion"]

def load_annotations(split_dir: str) -> pd.DataFrame:
    split_dir = Path(split_dir)
    rows = []
    for ann_file in sorted((split_dir / "annotations").glob("*.txt")):
        image_id = ann_file.stem # gets the file name (image id), without file extension
        img_w, img_h = Image.open(split_dir / "images" / f"{image_id}.jpg").size
        for line in ann_file.read_text().strip().splitlines():
            values = [int(v) for v in line.strip().strip(",").split(",")[:8]]
            rows.append([image_id, img_w, img_h, *values])
    df = pd.DataFrame(rows, columns=["image_id", "img_w", "img_h", *COLUMNS])
    df["class_name"] = df["category"].map(CATEGORIES)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load VisDrone annotations and print a summary.")
    parser.add_argument("split_dir", help="dataset split folder, e.g. data/VisDrone2019-DET-val")
    args = parser.parse_args()
    df = load_annotations(args.split_dir)
    print(df.head())
    print(f"\n{df['image_id'].nunique()} images, {len(df)} boxes")
    print(df["class_name"].value_counts())