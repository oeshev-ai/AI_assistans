"""Даунскейл входа: длинная сторона, порог, альфа и выключение."""

from __future__ import annotations

import io

from PIL import Image

from app.media.media_utils import downscale_for_edit, fit_image_size


def _jpeg(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (1, 2, 3)).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_twelve_megapixel_long_side_is_1536() -> None:
    scaled, mime = downscale_for_edit(_jpeg(4032, 3024), "image/jpeg", 1536)
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(scaled)) as image:
        assert max(image.size) == 1536
        assert image.size == (1536, 1152)


def test_under_threshold_is_byte_identical() -> None:
    data = _jpeg(800, 600)
    scaled, mime = downscale_for_edit(data, "image/jpeg", 1536)
    assert mime == "image/jpeg"
    assert scaled is data


def test_png_with_alpha_stays_png() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (2000, 800), (1, 2, 3, 128)).save(buffer, format="PNG")
    scaled, mime = downscale_for_edit(buffer.getvalue(), "image/png", 1536)
    assert mime == "image/png"
    with Image.open(io.BytesIO(scaled)) as image:
        assert image.mode == "RGBA"
        assert max(image.size) == 1536


def test_zero_max_edge_disables_downscale() -> None:
    data = _jpeg(4032, 3024)
    scaled, mime = downscale_for_edit(data, "image/jpeg", 0)
    assert mime == "image/jpeg"
    assert scaled is data
    assert fit_image_size(4032, 3024) != "1536x1152"
