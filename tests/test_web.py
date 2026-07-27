import io
import threading
import time

from fastapi.testclient import TestClient

from karaoke_maker.config import Config
from karaoke_maker.pipeline import STAGE_ORDER, PipelineError, PipelineResult
from karaoke_maker.web import create_app


def make_client(tmp_path, runner):
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work",
                 output_dir=tmp_path / "out")
    return TestClient(create_app(cfg, runner=runner)), cfg


def ok_runner(inp, cfg, log=print, force_from=None, stages=None):
    log("[acquire] running")
    return PipelineResult(output_dir=cfg.output_dir / "X", warnings=["check 1:32"])


def wait_status(client, job_id, status, timeout=5):
    for _ in range(int(timeout / 0.05)):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == status:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job never reached {status}: {job}")


def test_submit_and_complete(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "la la",
        "youtube_url": "http://y"})
    assert resp.status_code == 201
    job = wait_status(client, resp.json()["id"], "done")
    assert job["warnings"] == ["check 1:32"]
    assert any("[acquire] running" in line for line in job["log"])


def test_failure_reported(tmp_path):
    def bad_runner(*a, **kw):
        raise PipelineError("boom")
    client, _ = make_client(tmp_path, bad_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"})
    job = wait_status(client, resp.json()["id"], "failed")
    assert job["error"] == "boom"


def test_requires_one_source(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"})
    assert resp.status_code == 400


def test_rejects_both_sources(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"},
        files={"video": ("v.mp4", b"data", "video/mp4")})
    assert resp.status_code == 400


def test_job_list_and_404(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.get("/api/jobs").json() == []
    assert client.get("/api/jobs/nope").status_code == 404


def test_index_served(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.get("/")
    assert resp.status_code == 200 and "<form" in resp.text


def test_rejects_nonpositive_bpm(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y", "bpm": "0"})
    assert resp.status_code == 400


def test_rejects_nonfinite_bpm(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y", "bpm": "nan"})
    assert resp.status_code == 400


def test_accepts_positive_bpm(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y", "bpm": "120"})
    assert resp.status_code == 201
    wait_status(client, resp.json()["id"], "done")


def test_worker_survives_arbitrary_exception_and_keeps_processing(tmp_path):
    calls = []

    def flaky_runner(inp, cfg, log=print, force_from=None, stages=None):
        calls.append(inp.title)
        if inp.title == "bad":
            raise ValueError("kaboom")
        log("[acquire] running")
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, flaky_runner)
    bad = client.post("/api/jobs", data={
        "title": "bad", "artist": "A", "lyrics": "x", "youtube_url": "http://y"})
    good = client.post("/api/jobs", data={
        "title": "good", "artist": "A", "lyrics": "x", "youtube_url": "http://y"})

    bad_job = wait_status(client, bad.json()["id"], "failed")
    assert bad_job["error"] == "kaboom"
    good_job = wait_status(client, good.json()["id"], "done")
    assert good_job["status"] == "done"


def test_uploaded_video_filename_is_sanitised(tmp_path):
    seen = {}

    def capture_runner(inp, cfg, log=print, force_from=None, stages=None):
        seen["video_path"] = inp.video_path
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, cfg = make_client(tmp_path, capture_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("../../etc/passwd", io.BytesIO(b"data"), "video/mp4")})
    assert resp.status_code == 201
    job_id = resp.json()["id"]
    wait_status(client, job_id, "done")

    uploads_root = (cfg.work_dir / "uploads").resolve()
    saved_path = seen["video_path"]
    assert saved_path is not None
    assert saved_path.resolve().parent == uploads_root / job_id
    assert saved_path.name == "passwd"
    assert not (tmp_path / "etc" / "passwd").exists()


def test_upload_dir_cleaned_up_after_success(tmp_path):
    client, cfg = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4")})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "done")
    assert not (cfg.work_dir / "uploads" / job_id).exists()


def test_upload_dir_cleaned_up_after_failure(tmp_path):
    def bad_runner(*a, **kw):
        raise PipelineError("boom")
    client, cfg = make_client(tmp_path, bad_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4")})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "failed")
    assert not (cfg.work_dir / "uploads" / job_id).exists()


def test_jobs_run_one_at_a_time(tmp_path):
    counter_lock = threading.Lock()
    state = {"current": 0, "max": 0}

    def slow_runner(inp, cfg, log=print, force_from=None, stages=None):
        with counter_lock:
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
        time.sleep(0.1)
        with counter_lock:
            state["current"] -= 1
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, slow_runner)
    ids = [client.post("/api/jobs", data={
        "title": f"T{i}", "artist": "A", "lyrics": "x", "youtube_url": "http://y"}
        ).json()["id"] for i in range(4)]
    for job_id in ids:
        wait_status(client, job_id, "done")
    assert state["max"] == 1


def test_job_has_timestamps(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "la", "youtube_url": "http://y"})
    job = wait_status(client, resp.json()["id"], "done")
    assert job["created_at"] <= job["started_at"] <= job["finished_at"]


def test_queued_job_has_no_start_or_finish(tmp_path):
    gate = threading.Event()

    def blocking_runner(inp, cfg, log=print, force_from=None, stages=None):
        gate.wait(timeout=5)
        return PipelineResult(output_dir=cfg.output_dir / "X", warnings=[])

    client, _ = make_client(tmp_path, blocking_runner)
    try:
        first = client.post("/api/jobs", data={
            "title": "T1", "artist": "A", "lyrics": "la", "youtube_url": "http://y"})
        second = client.post("/api/jobs", data={
            "title": "T2", "artist": "A", "lyrics": "la", "youtube_url": "http://y"})
        queued = client.get(f"/api/jobs/{second.json()['id']}").json()
        assert queued["created_at"] > 0
        assert queued["started_at"] is None and queued["finished_at"] is None
    finally:
        gate.set()
        wait_status(client, first.json()["id"], "done")


def test_failed_job_still_records_finish(tmp_path):
    def bad_runner(*a, **kw):
        raise PipelineError("boom")
    client, _ = make_client(tmp_path, bad_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"})
    job = wait_status(client, resp.json()["id"], "failed")
    assert job["finished_at"] is not None and job["finished_at"] >= job["started_at"]


def test_job_list_includes_timestamps(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "la", "youtube_url": "http://y"})
    wait_status(client, resp.json()["id"], "done")
    summary = client.get("/api/jobs").json()[0]
    assert {"created_at", "started_at", "finished_at"} <= set(summary)


def test_stages_endpoint_matches_stage_order(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.get("/api/stages").json()["stages"] == STAGE_ORDER


def test_cached_stage_line_advances_stage(tmp_path):
    def cached_runner(inp, cfg, log=print, force_from=None, stages=None):
        log("[acquire] cached, skipping")
        log("[separate] cached, skipping")
        log("[transcribe] running")
        return PipelineResult(output_dir=cfg.output_dir / "X", warnings=[])
    client, _ = make_client(tmp_path, cached_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "la", "youtube_url": "http://y"})
    job = wait_status(client, resp.json()["id"], "done")
    assert job["stage"] == "transcribe"


def test_cached_only_run_reports_last_cached_stage(tmp_path):
    def all_cached(inp, cfg, log=print, force_from=None, stages=None):
        for name in ("acquire", "separate", "transcribe"):
            log(f"[{name}] cached, skipping")
        return PipelineResult(output_dir=cfg.output_dir / "X", warnings=[])
    client, _ = make_client(tmp_path, all_cached)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "la", "youtube_url": "http://y"})
    job = wait_status(client, resp.json()["id"], "done")
    assert job["stage"] == "transcribe"


def test_unrelated_log_lines_do_not_change_stage(tmp_path):
    def noisy(inp, cfg, log=print, force_from=None, stages=None):
        log("[align] running")
        log("some incidental output")
        log("[not a stage line]")
        return PipelineResult(output_dir=cfg.output_dir / "X", warnings=[])
    client, _ = make_client(tmp_path, noisy)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "la", "youtube_url": "http://y"})
    job = wait_status(client, resp.json()["id"], "done")
    assert job["stage"] == "align"


def test_fonts_are_served(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.get("/static/fonts/chakra-petch-600.woff2")
    assert resp.status_code == 200
    assert resp.content[:4] == b"wOF2"


def test_font_licence_is_bundled():
    from karaoke_maker.web import STATIC
    licence = (STATIC / "fonts" / "OFL.txt").read_text(encoding="utf-8", errors="replace")
    assert "SIL OPEN FONT LICENSE" in licence.upper()


def _index_html():
    from karaoke_maker.web import STATIC
    return (STATIC / "index.html").read_text(encoding="utf-8")


def test_index_has_required_hooks():
    html = _index_html()
    for hook in ("song-form", "jobs", "detail", "review", "now-bar", "sidebar"):
        assert f'id="{hook}"' in html, hook
    for name in ("title", "artist", "lyrics", "youtube_url", "video",
                 "genre", "year", "language", "bpm", "cover"):
        assert f'name="{name}"' in html, name


def test_index_makes_no_external_requests():
    import re as _re
    html = _index_html()
    for url in _re.findall(r'(?:src|href)\s*=\s*["\']([^"\']+)', html):
        assert not url.startswith(("http://", "https://", "//")), url
    assert "@import" not in html


def test_index_renders_api_data_without_innerhtml():
    """API-derived values must go through textContent, never innerHTML."""
    import re as _re
    html = _index_html()
    for m in _re.finditer(r"\.(innerHTML|outerHTML|insertAdjacentHTML)\s*(=|\()\s*(.*)", html):
        assert "${" not in m.group(3), f"interpolation into {m.group(1)}: {m.group(3)[:80]}"


def test_index_respects_reduced_motion():
    assert "prefers-reduced-motion" in _index_html()
