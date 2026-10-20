"""The demo command's recorded outputs."""

from PIL import Image

import main


def test_demo_writes_looping_gif(tmp_path):
    out = tmp_path / "demo.gif"
    assert main.main(["demo", "--task", "type_key", "--text", "a", "--gif", str(out), "--gif-width", "320"]) == 0
    with Image.open(out) as im:
        assert im.size == (320, 180)
        assert im.n_frames > 3
        assert im.info.get("loop") == 0  # loops forever
        im.seek(im.n_frames - 1)
        assert im.info.get("duration") >= 2500  # result held on screen (identical frames get merged)
