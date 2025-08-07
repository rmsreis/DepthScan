import pytest
from data_loader import parse_depth_from_filename


def test_parse_depth_integer():
    filename = "sample_5micron_deep.jpg"
    assert parse_depth_from_filename(filename) == 5.0


def test_parse_depth_decimal():
    filename = "sample_7.5um.png"
    assert parse_depth_from_filename(filename) == pytest.approx(7.5)


def test_parse_depth_missing_raises():
    filename = "sample_without_depth.jpg"
    with pytest.raises(ValueError):
        parse_depth_from_filename(filename)
