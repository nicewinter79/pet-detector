"""
zone_detector.py — Detects when tracked animals enter/exit a defined
rectangular zone in a video, using the pet-detector YOLO model.

New concepts vs predict.py:
- Point-in-rectangle geometry (is a detection's center inside our zone?)
- State tracking per animal (was it inside last frame? is it inside now?)
- Event logging (ENTERED / EXITED) instead of just raw counting

Usage:
    python zone_detector.py --source your_video.mp4
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# Zone coordinates are in the video's ORIGINAL resolution, not the
# smaller size shown during inference (e.g. 640x384). Use debug
# prints of box centers to find real coordinates for a new video.


def is_inside_zone(px: float, py: float, zone: tuple) -> bool:
    """
    Check if point (px, py) is inside the rectangle defined by
    zone = (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = zone
    return x1 <= px <= x2 and y1 <= py <= y2


def get_box_center(box) -> tuple:
    """Given a YOLO box (xyxy format), return its center point (px, py)."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    px = (x1 + x2) / 2
    py = (y1 + y2) / 2
    return px, py


def main():
    parser = argparse.ArgumentParser(description="Detect zone entry/exit events using the pet-detector model.")
    parser.add_argument("--model", default="../runs/detect/train/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Path to a video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model weights not found at: {args.model}")

    model = YOLO(args.model)

    # was_inside keeps track, per track ID, of whether that animal was
    # inside the zone on the PREVIOUS frame. This is the "state" we need
    # to detect ENTER/EXIT events rather than just "currently inside".
    was_inside = {}

    results = model.track(source=args.source, conf=args.conf, stream=True, persist=True)

    frame_num = 0
    for result in results:
        frame_num += 1

        if result.boxes.id is None:
            continue

        for box, track_id, cls_idx in zip(result.boxes, result.boxes.id, result.boxes.cls):
            track_id = int(track_id)
            class_name = model.names[int(cls_idx)]

            px, py = get_box_center(box)
            currently_inside = is_inside_zone(px, py, ZONE)
            previously_inside = was_inside.get(track_id, False)

            if currently_inside and not previously_inside:
                print(f"[frame {frame_num}] ENTERED zone: {class_name} (id {track_id})")
            elif not currently_inside and previously_inside:
                print(f"[frame {frame_num}] EXITED zone:  {class_name} (id {track_id})")

            was_inside[track_id] = currently_inside


if __name__ == "__main__":
    main()
