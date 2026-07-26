import io
import time

from fastapi.testclient import TestClient

from karaoke_maker.config import Config
from karaoke_maker.pipeline import PipelineError, PipelineResult
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
