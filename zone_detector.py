"""
zone_detector.py – Detects when tracked animals enter/exit one or more
defined rectangular zones in a video, using the pet-detector YOLO model.
Also draws each zone on the video and saves an annotated output so
you can visually see the zones and the crossings.

Concepts:
- Point-in-rectangle geometry (is a detection's center inside a zone?)
- State tracking per (animal, zone) pair (was it inside last frame? now?)
- Event logging (ENTERED / EXITED) instead of just raw counting
- Drawing overlays on video frames with OpenCV

Usage:
    python zone_detector.py --source your_video.mp4
    python zone_detector.py --source your_video.mp4 --log events.csv
    python zone_detector.py --source your_video.mp4 --debug-centers
"""

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

# Zone coordinates are in the video's ORIGINAL resolution, not the
# smaller size shown during inference (e.g. 640x384). Use --debug-centers
# to print box centers and find real coordinates for a new video.
#
# Each zone is a dict: id (used in logs/UI), coords (x1, y1, x2, y2),
# and color (BGR, for drawing so zones are visually distinguishable).
ZONES = [
    {"id": "zone_1", "coords": (800, 2400, 1200, 2700), "color": (0, 0, 255)},    # red
    {"id": "zone_2", "coords": (600, 2650, 1000, 2850), "color": (255, 0, 0)},    # blue
]


def is_inside_zone(px: float, py: float, zone_coords: tuple) -> bool:
    """Check if point (px, py) is inside the rectangle zone_coords = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = zone_coords
    return x1 <= px <= x2 and y1 <= py <= y2


def get_box_center(box) -> tuple:
    """Given a YOLO box (xyxy format), return its center point (px, py)."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    px = (x1 + x2) / 2
    py = (y1 + y2) / 2
    return px, py


def draw_zone(frame, zone_coords, color=(0, 0, 255), thickness=3):
    """Draw one zone rectangle on a video frame (BGR color, default red)."""
    x1, y1, x2, y2 = zone_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def log_event(csv_writer, frame_num, video_source, class_name, track_id, event, zone_id):
    """Write a single ENTERED/EXITED event as a CSV row and flush immediately.

    Flushing per-row keeps the file readable/tail-able while a long video is
    still processing, at basically no performance cost for this workload.
    """
    csv_writer.writerow([
        datetime.now().isoformat(timespec="seconds"),
        frame_num,
        video_source,
        class_name,
        track_id,
        event,
        zone_id,
    ])


def main():
    parser = argparse.ArgumentParser(description="Detect zone entry/exit events using the pet-detector model.")
    parser.add_argument("--model", default="weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", required=True, help="Path to a video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out", default="zone_output.mp4", help="Path to save annotated output video")
    parser.add_argument("--log", default="zone_events.csv", help="Path to save the event log CSV")
    parser.add_argument("--debug-centers", action="store_true", help="Print each detection's center point per frame")
    args = parser.parse_args()

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model weights not found at: {args.model}")

    model = YOLO(args.model)

    # Keyed by (track_id, zone_id) since an animal can be inside one zone
    # and outside another at the same time -- each zone needs its own
    # independent enter/exit state per animal.
    was_inside = {}
    event_counts = defaultdict(int)  # e.g. ("dog", "ENTERED", "zone_1") -> 3

    results = model.track(source=args.source, conf=args.conf, stream=True, persist=True)

    out_writer = None
    frame_num = 0

    # Append across runs so history accumulates (e.g. multiple videos over
    # time), instead of each run wiping out the previous one's events.
    log_path = Path(args.log)
    write_header = not log_path.exists()
    log_file = open(args.log, "a", newline="")
    csv_writer = csv.writer(log_file)
    if write_header:
        csv_writer.writerow(["timestamp", "frame_num", "video_source", "class_name", "track_id", "event", "zone_id"])

    for result in results:
        frame_num += 1
        frame = result.orig_img
        for zone in ZONES:
            frame = draw_zone(frame, zone["coords"], color=zone["color"])

        if result.boxes.id is not None:
            for box, track_id, cls_idx in zip(result.boxes, result.boxes.id, result.boxes.cls):
                track_id = int(track_id)
                class_name = model.names[int(cls_idx)]

                px, py = get_box_center(box)
                if args.debug_centers:
                    print(f"frame {frame_num}: center=({px:.0f}, {py:.0f})")

                for zone in ZONES:
                    zone_id = zone["id"]
                    currently_inside = is_inside_zone(px, py, zone["coords"])
                    key = (track_id, zone_id)
                    previously_inside = was_inside.get(key, False)

                    if currently_inside and not previously_inside:
                        print(f"[frame {frame_num}] ENTERED {zone_id}: {class_name} (id {track_id})")
                        log_event(csv_writer, frame_num, args.source, class_name, track_id, "ENTERED", zone_id)
                        event_counts[(class_name, "ENTERED", zone_id)] += 1
                    elif not currently_inside and previously_inside:
                        print(f"[frame {frame_num}] EXITED {zone_id}: {class_name} (id {track_id})")
                        log_event(csv_writer, frame_num, args.source, class_name, track_id, "EXITED", zone_id)
                        event_counts[(class_name, "EXITED", zone_id)] += 1

                    was_inside[key] = currently_inside

        if out_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(args.out, fourcc, 30, (w, h))

        out_writer.write(frame)

    log_file.close()

    if out_writer is not None:
        out_writer.release()
        print(f"\nSaved annotated video with zone overlay to: {args.out}")

    print(f"Saved event log to: {args.log}")

    if event_counts:
        print("\nSummary:")
        for (class_name, event, zone_id), count in sorted(event_counts.items()):
            print(f"  {class_name:10s} {event:8s} {zone_id:8s} {count}")
    else:
        print("\nNo zone crossings detected.")


if __name__ == "__main__":
    main()
