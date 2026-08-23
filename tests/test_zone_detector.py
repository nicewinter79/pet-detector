import pytest
from zone_detector import is_inside_zone, get_box_center


def test_is_inside_zone_point_inside():
    zone = (0, 0, 100, 100)
    assert is_inside_zone(50, 50, zone) is True


def test_is_inside_zone_point_outside():
    zone = (0, 0, 100, 100)
    assert is_inside_zone(150, 150, zone) is False


def test_is_inside_zone_point_on_boundary():
    zone = (0, 0, 100, 100)
    assert is_inside_zone(0, 0, zone) is True
    assert is_inside_zone(100, 100, zone) is True


def test_is_inside_zone_point_just_outside_boundary():
    zone = (0, 0, 100, 100)
    assert is_inside_zone(101, 50, zone) is False
    assert is_inside_zone(50, -1, zone) is False


class FakeTensor:
    """Mimics the .tolist() interface of a YOLO box's tensor coordinates."""
    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data


class FakeBox:
    """Mimics the .xyxy[0] interface of a real YOLO box result."""
    def __init__(self, x1, y1, x2, y2):
        self.xyxy = [FakeTensor([x1, y1, x2, y2])]


def test_get_box_center_basic():
    box = FakeBox(0, 0, 100, 100)
    px, py = get_box_center(box)
    assert px == 50
    assert py == 50


def test_get_box_center_offset():
    box = FakeBox(10, 20, 30, 60)
    px, py = get_box_center(box)
    assert px == 20
    assert py == 40
