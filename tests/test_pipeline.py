import json
from pathlib import Path
from karaoke_maker.config import Config
from karaoke_maker.pipeline import SongInput, Stage, run_pipeline, song_slug

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
