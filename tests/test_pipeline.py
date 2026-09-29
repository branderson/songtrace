import json
from pathlib import Path
from songtrace.config import Config
from songtrace.pipeline import SongInput, Stage, read_warnings, run_pipeline, song_slug

def make_cfg(tmp_path):
    return Config(whisper_url="http://x/v1", output_dir=tmp_path / "out",
                  work_dir=tmp_path / "work")

def make_inp():
    return SongInput(title="T/est", artist="A", lyrics_text="la la")

def fake_stage(name, calls):
    def outputs(ctx):
        return [ctx.work / f"{name}.done"]
    def run(ctx):
        calls.append(name)
        (ctx.work / f"{name}.done").write_text("x")
    return Stage(name=name, outputs=outputs, run=run)

def test_slug_strips_slash():
    assert song_slug(make_inp()) == "A - Test"

def test_runs_all_stages_then_skips_cached(tmp_path):
    cfg, inp, calls = make_cfg(tmp_path), make_inp(), []
    stages = [fake_stage("s1", calls), fake_stage("s2", calls)]
    run_pipeline(inp, cfg, log=lambda m: None, stages=stages)
    assert calls == ["s1", "s2"]
    run_pipeline(inp, cfg, log=lambda m: None, stages=stages)
    assert calls == ["s1", "s2"]

def test_force_from_reruns_tail(tmp_path):
    cfg, inp, calls = make_cfg(tmp_path), make_inp(), []
    stages = [fake_stage("s1", calls), fake_stage("s2", calls), fake_stage("s3", calls)]
    run_pipeline(inp, cfg, log=lambda m: None, stages=stages)
    calls.clear()
    run_pipeline(inp, cfg, log=lambda m: None, stages=stages, force_from="s2")
    assert calls == ["s2", "s3"]

def test_warnings_surfaced(tmp_path):
    cfg, inp = make_cfg(tmp_path), make_inp()
    def run(ctx):
        (ctx.work / "notes_sec.json").write_text(
            json.dumps({"notes": [], "breaks": [], "warnings": ["w1"]}))
        (ctx.work / "n.done").write_text("x")
    st = Stage(name="n", outputs=lambda ctx: [ctx.work / "n.done"], run=run)
    result = run_pipeline(inp, cfg, log=lambda m: None, stages=[st])
    assert result.warnings == ["w1"]

def test_read_warnings_is_empty_without_a_notes_file(tmp_path):
    assert read_warnings(tmp_path) == []

def test_read_warnings_reads_the_notes_file(tmp_path):
    (tmp_path / "notes_sec.json").write_text(json.dumps({"warnings": ["w1", "w2"]}))
    assert read_warnings(tmp_path) == ["w1", "w2"]

def test_gpu_is_released_after_every_stage(tmp_path, monkeypatch):
    """Not just at the end: a later stage may wait minutes on the Whisper server."""
    from songtrace import models
    cfg, inp, calls = make_cfg(tmp_path), make_inp(), []
    monkeypatch.setattr(models, "release", lambda: calls.append("release"))
    stages = [fake_stage("s1", calls), fake_stage("s2", calls)]
    run_pipeline(inp, cfg, log=lambda m: None, stages=stages)
    assert calls == ["s1", "release", "s2", "release"]

def test_gpu_is_released_when_a_stage_fails(tmp_path, monkeypatch):
    import pytest
    from songtrace import models
    from songtrace.pipeline import PipelineError
    cfg, inp, calls = make_cfg(tmp_path), make_inp(), []
    monkeypatch.setattr(models, "release", lambda: calls.append("release"))
    def boom(ctx):
        raise PipelineError("boom")
    stages = [Stage(name="s1", outputs=lambda ctx: [ctx.work / "s1.done"], run=boom)]
    with pytest.raises(PipelineError):
        run_pipeline(inp, cfg, log=lambda m: None, stages=stages)
    assert calls == ["release"]

def test_a_cached_stage_does_not_pay_for_a_release(tmp_path, monkeypatch):
    from songtrace import models
    cfg, inp, calls = make_cfg(tmp_path), make_inp(), []
    stages = [fake_stage("s1", calls)]
    run_pipeline(inp, cfg, log=lambda m: None, stages=stages)
    calls.clear()
    monkeypatch.setattr(models, "release", lambda: calls.append("release"))
    run_pipeline(inp, cfg, log=lambda m: None, stages=stages)
    assert calls == []

def test_a_release_that_freed_memory_is_reported_in_the_log(tmp_path, monkeypatch):
    from songtrace import models
    cfg, inp, lines = make_cfg(tmp_path), make_inp(), []
    monkeypatch.setattr(models, "release", lambda: 512 * 2 ** 20)
    run_pipeline(inp, cfg, log=lines.append, stages=[fake_stage("s1", [])])
    assert "[s1] released 512 MiB of GPU memory" in lines

def test_a_release_that_freed_nothing_stays_out_of_the_log(tmp_path, monkeypatch):
    from songtrace import models
    cfg, inp, lines = make_cfg(tmp_path), make_inp(), []
    monkeypatch.setattr(models, "release", lambda: 0)
    run_pipeline(inp, cfg, log=lines.append, stages=[fake_stage("s1", [])])
    assert not any("released" in line for line in lines)
