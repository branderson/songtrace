import os
import shutil

import pytest

requires_full = pytest.mark.skipif(
    not os.environ.get("KARAOKE_E2E"),
    reason="set KARAOKE_E2E=1 (needs GPU, ffmpeg, whisper server, network)")

@requires_full
def test_full_pipeline(tmp_path):
    from karaoke_maker.config import load_config
    from karaoke_maker.pipeline import SongInput, run_pipeline
    assert shutil.which("ffmpeg"), "ffmpeg required"
    cfg = load_config()
    cfg.output_dir = tmp_path / "out"
    cfg.work_dir = tmp_path / "work"
    inp = SongInput(
        title=os.environ["KARAOKE_E2E_TITLE"],
        artist=os.environ["KARAOKE_E2E_ARTIST"],
        lyrics_text=open(os.environ["KARAOKE_E2E_LYRICS"]).read(),
        youtube_url=os.environ.get("KARAOKE_E2E_URL"),
    )
    result = run_pipeline(inp, cfg)
    txts = list(result.output_dir.glob("*.txt"))
    assert len(txts) == 1
    from karaoke_maker.ultrastar import get_header, parse_ultrastar
    song = parse_ultrastar(txts[0].read_text(encoding="utf-8"))
    assert get_header(song, "BPM") and get_header(song, "GAP")
    assert len(song.body) > 20
