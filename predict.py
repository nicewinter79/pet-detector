"""
predict.py — Run the trained pet-detector model on an image or video
and count how many of each animal class were detected.

Usage:
    python predict.py --source path/to/image_or_video --model ../runs/detect/train/weights/best.pt
"""

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def run_detection(model_path: str, source: str, conf: float = 0.25, save: bool = True):
    model = YOLO(model_path)
    results = model.predict(source=source, conf=conf, save=save, stream=True)

    counts = Counter()
    total_frames = 0

    for result in results:
        total_frames += 1
        for cls_idx in result.boxes.cls:
            class_name = model.names[int(cls_idx)]
            counts[class_name] += 1

    return counts, total_frames


def main():
    parser = argparse.ArgumentParser(description="Detect and count animals with the pet-detector YOLO model.")
    parser.add_argument("--model", default="../runs/detect/train/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Path to image, folder, or video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save annotated output")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model weights not found at: {args.model}")

    counts, total_frames = run_detection(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        save=not args.no_save,
    )

    print("\n" + "=" * 40)
    print(f"Processed {total_frames} image(s)/frame(s)")
    print("=" * 40)

    if not counts:
        print("No animals detected.")
    else:
        for class_name, count in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"{class_name:12s}: {count}")
        print("-" * 40)
        print(f"{'TOTAL':12s}: {sum(counts.values())}")

    print("=" * 40)


if __name__ == "__main__":
    main()