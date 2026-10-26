"""The model-file picker's extension filter.

The form sends the list `modelPathExtensions` builds (`ref_image` is
`.png,.jpg,.jpeg,.webp,.bmp`), so the service must split it instead of comparing
the whole string against one suffix.
"""

from __future__ import annotations

from webui.services.file_service import browse_directory


def test_the_extension_filter_accepts_a_comma_separated_list(tmp_path):
    for name in ("a.PNG", "b.jpg", "notes.txt", "c.safetensors"):
        (tmp_path / name).write_bytes(b"x")

    listing = browse_directory(str(tmp_path), ".png,.jpg")
    assert sorted(entry["name"] for entry in listing["files"]) == ["a.PNG", "b.jpg"]

    only = browse_directory(str(tmp_path), ".safetensors")
    assert [entry["name"] for entry in only["files"]] == ["c.safetensors"]
