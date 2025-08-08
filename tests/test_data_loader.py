import os
import sys
import pytest

# Ensure the repository root is on the path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data_loader import parse_depth_from_filename


def test_parse_depth_um():
    assert parse_depth_from_filename('slice_10um.jpg') == 10.0


def test_parse_depth_micron():
    assert parse_depth_from_filename('slice_5micron.png') == 5.0


def test_parse_depth_microns():
    assert parse_depth_from_filename('slice_20microns.tif') == 20.0


def test_parse_depth_missing_value():
    with pytest.raises(ValueError):
        parse_depth_from_filename('slice_no_depth.png')
