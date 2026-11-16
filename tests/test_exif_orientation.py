"""EXIF Orientation=6 учитывается и в замере, и в даунскейле."""

from __future__ import annotations

import io

from PIL import Image

from app.media.media_utils import downscale_for_edit, source_image_size


def _landscape_with_orientation_6() -> bytes:
    image = Image.new("RGB", (2000, 1000), (30, 60, 90))
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_orientation_6_is_portrait_in_measure_and_downscale() -> None:
    data = _landscape_with_orientation_6()
    width, height = source_image_size(data)
    assert (width, height) == (1000, 2000)

    scaled, mime = downscale_for_edit(data, "image/jpeg", 1536)
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(scaled)) as image:
        assert image.size[1] > image.size[0]
        assert max(image.size) == 1536
