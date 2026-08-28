import pandas as pd
from zone_analysis import detect_rapid_crossing


def make_events(timestamps, track_id=1, zone_id="zone_1", video_source="test.mp4", class_name="dog"):
    return pd.DataFrame({
        "timestamp": timestamps,
        "frame_num": range(len(timestamps)),
        "video_source": video_source,
        "class_name": class_name,
        "track_id": track_id,
        "event": "ENTERED",
        "zone_id": zone_id,
    })


def test_no_events():
    events = pd.DataFrame(columns=["timestamp", "frame_num", "video_source", "class_name", "track_id", "event", "zone_id"])
    result = detect_rapid_crossing(events)
    assert result.empty


def test_normal_crossings_not_flagged():
    events = make_events(["2026-01-01T10:00:00", "2026-01-01T10:00:05"])
    result = detect_rapid_crossing(events, window_seconds=10, min_crossings=5)
    assert result.empty


def test_rapid_crossings_flagged():
    events = make_events([
        "2026-01-01T10:00:00", "2026-01-01T10:00:01", "2026-01-01T10:00:02",
        "2026-01-01T10:00:03", "2026-01-01T10:00:04",
    ])
    result = detect_rapid_crossing(events, window_seconds=10, min_crossings=5)
    assert len(result) == 1
    assert result.iloc[0]["crossing_count"] == 5
    assert result.iloc[0]["track_id"] == 1
    assert result.iloc[0]["zone_id"] == "zone_1"


def test_crossings_spread_out_not_flagged():
    events = make_events([
        "2026-01-01T10:00:00", "2026-01-01T10:00:15", "2026-01-01T10:00:30",
        "2026-01-01T10:00:45", "2026-01-01T10:01:00",
    ])
    result = detect_rapid_crossing(events, window_seconds=10, min_crossings=5)
    assert result.empty
