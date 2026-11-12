import argparse
from pathlib import Path

import pandas as pd
from torch.cuda import is_available
from ultralytics import YOLO

MODEL = "yolo11s.pt"
IMGSZ = 1280
MIN_CONF = 0.1

def pick_device():
    """Use an NVIDIA GPU (CUDA) if available, else the CPU."""
    if is_available():
        return 0
    return "cpu"


def main(split_dir, out_csv="outputs/predictions.csv", batch_size=1):
    model = YOLO(MODEL)
    device = pick_device()
    rows = []
    image_paths = sorted(Path(split_dir, "images").glob("*.jpg"))
    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start:start + batch_size]
        results = model.predict([str(p) for p in batch], imgsz=IMGSZ, conf=MIN_CONF, verbose=False, device=device, batch=batch_size)
        for path, result in zip(batch, results):
            for (x1, y1, x2, y2), conf, cls in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist()
            ):
                rows.append([path.stem, x1, y1, x2, y2, conf, model.names[int(cls)]])
        done = min(start + batch_size, len(image_paths))
        if done % 50 < batch_size or done == len(image_paths):
            print(f"{done}/{len(image_paths)} images done")

    df = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "conf", "coco_class"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} detections on {len(image_paths)} images to {out_csv}")
    print(df["coco_class"].value_counts().head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a COCO-pretrained YOLO on VisDrone and save detections.")
    parser.add_argument("split_dir", help="dataset split folder, e.g. data/VisDrone2019-DET-val")
    parser.add_argument("--out", default="outputs/predictions.csv", help="output CSV")
    parser.add_argument("--batch", type=int, default=1, help="batch size per forward pass (default 1)")
    args = parser.parse_args()
    main(args.split_dir, args.out)
