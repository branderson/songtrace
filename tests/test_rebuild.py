import dataclasses
from pathlib import Path

import pytest

from songtrace.config import Config
from songtrace.pipeline import STAGE_ORDER, PipelineError, SongInput
from songtrace.rebuild import (FIELD_STAGE, changed_fields, earliest, plan,
                                   relocate_work, superseded_output)


def song(**kw):
    base = dict(title="T", artist="A", lyrics_text="la", youtube_url="http://y")
    return SongInput(**{**base, **kw})


def cfg_for(tmp_path):
    return Config(whisper_url="u", work_dir=tmp_path / "work",
                  output_dir=tmp_path / "out")


def test_every_song_input_field_is_mapped():
    """A field added to SongInput without a FIELD_STAGE entry would make edits to
    it invalidate nothing, silently building from stale intermediates."""
    actual = {f.name for f in dataclasses.fields(SongInput)}
    assert actual - set(FIELD_STAGE) == set()
    assert set(FIELD_STAGE) - actual == set()


def test_every_mapped_stage_is_a_real_stage():
    assert set(FIELD_STAGE.values()) <= set(STAGE_ORDER)


def test_no_change_means_no_force_from():
    assert plan(song(), song()) is None


@pytest.mark.parametrize("field,value,stage", [
    ("youtube_url", "http://other", "acquire"),
    ("video_path", Path("/tmp/v.mp4"), "acquire"),
    ("language", "de", "transcribe"),
    ("lyrics_text", "different words", "align"),
    ("title", "Other", "grid"),
    ("artist", "Other", "grid"),
    ("genre", "rock", "grid"),
    ("year", 1999, "grid"),
    ("bpm_override", 123.0, "grid"),
    ("cover_path", Path("/tmp/c.jpg"), "package"),
])
def test_single_field_maps_to_its_stage(field, value, stage):
    assert plan(song(), song(**{field: value})) == stage


def test_earliest_stage_wins_when_several_change():
    new = song(lyrics_text="different", year=1999, language="de")
    assert plan(song(), new) == "transcribe"


def test_changed_fields_lists_what_moved():
    assert sorted(changed_fields(song(), song(year=1999, genre="rock"))) == \
        ["genre", "year"]


def test_slug_fields_invalidate_no_later_than_the_stage_that_writes_the_slug():
    """song.txt embeds the slug in #MP3/#COVER/#VIDEO and is written by `grid`,
    so a title/artist edit must rerun grid or earlier for relocate_work to be safe."""
    for field in ("title", "artist"):
        assert STAGE_ORDER.index(FIELD_STAGE[field]) <= STAGE_ORDER.index("grid")


def test_relocate_moves_the_work_dir_when_the_slug_changes(tmp_path):
    cfg = cfg_for(tmp_path)
    src = cfg.work_dir / "A - T"
    src.mkdir(parents=True)
    (src / "audio.wav").write_bytes(b"x")
    assert relocate_work(song(), song(title="Fixed"), cfg) is True
    assert not src.exists()
    assert (cfg.work_dir / "A - Fixed" / "audio.wav").read_bytes() == b"x"


def test_relocate_is_a_noop_when_the_slug_is_unchanged(tmp_path):
    cfg = cfg_for(tmp_path)
    src = cfg.work_dir / "A - T"
    src.mkdir(parents=True)
    assert relocate_work(song(), song(year=1999), cfg) is False
    assert src.exists()


def test_relocate_is_a_noop_when_there_is_nothing_to_move(tmp_path):
    cfg = cfg_for(tmp_path)
    assert relocate_work(song(), song(title="Fixed"), cfg) is False


def test_relocate_refuses_to_clobber_an_existing_work_dir(tmp_path):
    cfg = cfg_for(tmp_path)
    (cfg.work_dir / "A - T").mkdir(parents=True)
    (cfg.work_dir / "A - Fixed").mkdir(parents=True)
    with pytest.raises(PipelineError, match="already exists"):
        relocate_work(song(), song(title="Fixed"), cfg)
    assert (cfg.work_dir / "A - T").exists()


def test_superseded_output_names_the_replaced_folder(tmp_path):
    root = tmp_path / "out"
    assert superseded_output(str(root / "A - Old"), str(root / "A - New"), root) == \
        (root / "A - Old").resolve()


def test_superseded_output_is_none_when_the_slug_did_not_change(tmp_path):
    root = tmp_path / "out"
    assert superseded_output(str(root / "A - T"), str(root / "A - T"), root) is None


def test_superseded_output_is_none_without_a_previous_build(tmp_path):
    root = tmp_path / "out"
    assert superseded_output(None, str(root / "A - T"), root) is None


def test_superseded_output_refuses_a_path_outside_the_output_root(tmp_path):
    root = tmp_path / "out"
    assert superseded_output(str(tmp_path / "elsewhere"), str(root / "A - T"), root) is None


def test_superseded_output_refuses_the_output_root_itself(tmp_path):
    root = tmp_path / "out"
    assert superseded_output(str(root), str(root / "A - T"), root) is None


def test_earliest_picks_the_earlier_stage():
    assert earliest("package", "separate") == "separate"
    assert earliest("separate", "package") == "separate"


def test_earliest_ignores_unset_stages():
    assert earliest(None, "grid") == "grid"
    assert earliest("grid", None) == "grid"
    assert earliest(None, None) is None
