"""
zone_analysis.py - Post-hoc analysis of zone_detector.py's event log.

Currently provides: rapid zone-crossing detection, a simple "unusual
behavior" flag for animals entering/exiting the same zone unusually
often in a short time window (e.g. pacing, agitation, or a zone
boundary sitting on the animal's normal path).
"""

import pandas as pd


def detect_rapid_crossing(events: pd.DataFrame, window_seconds: int = 10, min_crossings: int = 5) -> pd.DataFrame:
    empty_result = pd.DataFrame(columns=[
        "video_source", "track_id", "zone_id", "class_name",
        "crossing_count", "window_start", "window_end",
    ])

    if events.empty:
        return empty_result

    entered = events[events["event"] == "ENTERED"].copy()
    if entered.empty:
        return empty_result

    entered["timestamp"] = pd.to_datetime(entered["timestamp"])

    flagged = []
    group_cols = ["video_source", "track_id", "zone_id"]

    for (video_source, track_id, zone_id), group in entered.groupby(group_cols):
        group = group.sort_values("timestamp").reset_index(drop=True)
        timestamps = group["timestamp"].tolist()
        class_name = group["class_name"].iloc[0]

        n = len(timestamps)
        i = 0
        while i < n:
            window_end_limit = timestamps[i] + pd.Timedelta(seconds=window_seconds)
            j = i
            while j < n and timestamps[j] <= window_end_limit:
                j += 1
            crossing_count = j - i

            if crossing_count >= min_crossings:
                flagged.append({
                    "video_source": video_source,
                    "track_id": track_id,
                    "zone_id": zone_id,
                    "class_name": class_name,
                    "crossing_count": crossing_count,
                    "window_start": timestamps[i],
                    "window_end": timestamps[j - 1],
                })
                i = j
            else:
                i += 1

    return pd.DataFrame(flagged) if flagged else empty_result
