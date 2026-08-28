import pandas as pd
from zone_analysis import calculate_dwell_times


def make_events(rows, video_source="test.mp4", zone_id="zone_1", class_name="dog"):
    return pd.DataFrame([
        {
            "timestamp": ts,
            "frame_num": i,
            "video_source": video_source,
            "class_name": class_name,
            "track_id": track_id,
            "event": event,
            "zone_id": zone_id,
        }
        for i, (ts, track_id, event) in enumerate(rows)
    ])


def test_no_events():
    events = pd.DataFrame(columns=["timestamp", "frame_num", "video_source", "class_name", "track_id", "event", "zone_id"])
    result = calculate_dwell_times(events)
    assert result.empty


def test_completed_visit():
    events = make_events([
        ("2026-01-01T10:00:00", 1, "ENTERED"),
        ("2026-01-01T10:00:30", 1, "EXITED"),
    ])
    result = calculate_dwell_times(events)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["status"] == "completed"
    assert row["dwell_seconds"] == 30


def test_ongoing_visit():
    events = make_events([
        ("2026-01-01T10:00:00", 1, "ENTERED"),
    ])
    result = calculate_dwell_times(events)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["status"] == "ongoing"
    assert pd.isna(row["dwell_seconds"])


def test_multiple_visits_same_track():
    events = make_events([
        ("2026-01-01T10:00:00", 1, "ENTERED"),
        ("2026-01-01T10:00:10", 1, "EXITED"),
        ("2026-01-01T10:05:00", 1, "ENTERED"),
        ("2026-01-01T10:05:20", 1, "EXITED"),
    ])
    result = calculate_dwell_times(events)
    assert len(result) == 2
    assert result.iloc[0]["dwell_seconds"] == 10
    assert result.iloc[1]["dwell_seconds"] == 20
    assert (result["status"] == "completed").all()


def test_completed_then_ongoing():
    events = make_events([
        ("2026-01-01T10:00:00", 1, "ENTERED"),
        ("2026-01-01T10:00:10", 1, "EXITED"),
        ("2026-01-01T10:05:00", 1, "ENTERED"),
    ])
    result = calculate_dwell_times(events)
    assert len(result) == 2
    assert result.iloc[0]["status"] == "completed"
    assert result.iloc[0]["dwell_seconds"] == 10
    assert result.iloc[1]["status"] == "ongoing"
    assert pd.isna(result.iloc[1]["dwell_seconds"])
