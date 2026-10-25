import numpy as np
import pytest

from src.vision import preprocessing as pp
from src.vision import synthetic


def test_decode_valid_png():
    s = synthetic.make_sample(1, defect="spot")
    img = pp.decode_image(synthetic.encode_png(s.image))
    assert img.ndim == 3 and img.shape[2] == 3


def test_decode_empty_raises():
    with pytest.raises(pp.ImageDecodeError):
        pp.decode_image(b"")


def test_decode_corrupt_png_raises():
    # Valid PNG magic but garbage body => decode failure, not a crash.
    corrupt = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    with pytest.raises(pp.ImageDecodeError):
        pp.decode_image(corrupt)


def test_unsupported_format_raises():
    gif = b"GIF89a" + b"\x00" * 16
    with pytest.raises(pp.UnsupportedFormatError):
        pp.decode_image(gif)


def test_unknown_signature_raises():
    with pytest.raises(pp.UnsupportedFormatError):
        pp.decode_image(b"not-an-image-at-all")


def test_normalize_downscales_and_grays():
    big = np.zeros((2000, 1000, 3), dtype=np.uint8)
    prep = pp.normalize(big, max_side=768)
    assert max(prep.bgr.shape[:2]) == 768
    assert prep.gray.ndim == 2
    assert 0 < prep.scale < 1
    assert prep.original_shape == (2000, 1000)


def test_normalize_no_upscale_small_image():
    small = np.zeros((100, 120, 3), dtype=np.uint8)
    prep = pp.normalize(small, max_side=768)
    assert prep.scale == 1.0
    assert prep.bgr.shape[:2] == (100, 120)
