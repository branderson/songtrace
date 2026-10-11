import io
import json
import re
import shutil
import threading
import time

import pytest
from fastapi.testclient import TestClient

from songtrace.config import Config
from songtrace.pipeline import STAGE_ORDER, PipelineError, PipelineResult, song_slug
from songtrace.web import STAGE_LINE, create_app


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


def test_submit_normalises_a_blank_genre_to_none(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, capture_runner_factory(seen))
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x",
        "youtube_url": "http://y", "genre": ""})
    wait_status(client, resp.json()["id"], "done")
    assert seen[0]["inp"].genre is None


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


def test_upload_retained_after_failure(tmp_path):
    """A job that died inside acquire must keep its source, or it can never rerun."""
    def bad_runner(*a, **kw):
        raise PipelineError("boom")
    client, cfg = make_client(tmp_path, bad_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4")})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "failed")
    assert (cfg.work_dir / "uploads" / job_id / "v.mp4").exists()


def test_cover_upload_kept_after_success(tmp_path):
    """A rebuild inherits cover_path; deleting it would silently drop #COVER."""
    client, cfg = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4"),
                               "cover": ("c.jpg", io.BytesIO(b"img"), "image/jpeg")})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "done")
    updir = cfg.work_dir / "uploads" / job_id
    assert (updir / "c.jpg").read_bytes() == b"img"
    assert not (updir / "v.mp4").exists()


def test_cleanup_uploads_removes_an_inherited_upload_dir(tmp_path):
    """A rebuild's video may live under its parent's id, not its own."""
    from songtrace.web import _cleanup_uploads
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work",
                 output_dir=tmp_path / "out")
    parent = cfg.work_dir / "uploads" / "parent"
    parent.mkdir(parents=True)
    (parent / "v.mp4").write_bytes(b"data")
    _cleanup_uploads(cfg, "child", parent / "v.mp4")
    assert not parent.exists()


def test_cleanup_uploads_ignores_paths_outside_the_upload_root(tmp_path):
    from songtrace.web import _cleanup_uploads
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work",
                 output_dir=tmp_path / "out")
    outside = tmp_path / "somewhere" / "keep.mp4"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"data")
    _cleanup_uploads(cfg, "child", outside)
    assert outside.exists()


def test_cleanup_uploads_tolerates_a_missing_file(tmp_path):
    from songtrace.web import _cleanup_uploads
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work",
                 output_dir=tmp_path / "out")
    _cleanup_uploads(cfg, "child", cfg.work_dir / "uploads" / "child" / "gone.mp4")
    _cleanup_uploads(cfg, "child", None)


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
    from songtrace.web import STATIC
    licence = (STATIC / "fonts" / "OFL.txt").read_text(encoding="utf-8", errors="replace")
    assert "SIL OPEN FONT LICENSE" in licence.upper()


def _index_html():
    from songtrace.web import STATIC
    return (STATIC / "index.html").read_text(encoding="utf-8")


def test_index_has_required_hooks():
    html = _index_html()
    for hook in ("song-form", "jobs", "detail", "review", "now-bar", "sidebar"):
        assert f'id="{hook}"' in html, hook
    for name in ("title", "artist", "lyrics", "youtube_url", "video",
                 "genre", "year", "language", "bpm", "cover"):
        assert f'name="{name}"' in html, name


def test_video_source_fields_are_outside_advanced_details():
    html = _index_html()
    form_start = html.index('id="song-form"')
    details_start = html.index("<details>", form_start)
    before = html[form_start:details_start]
    assert 'name="youtube_url"' in before
    assert 'name="video"' in before
    after = html[details_start:]
    for name in ("genre", "year", "language", "bpm", "cover"):
        assert f'name="{name}"' in after


def test_song_form_lyrics_uses_a_two_column_grid():
    css = _page_css(_index_html())
    assert re.search(r"\.song-form-grid\s*\{[^}]*grid-template-columns:\s*1fr 1fr",
                     css, re.S)


def test_song_form_grid_stacks_on_narrow_screens():
    css = _page_css(_index_html())
    assert re.search(
        r"@media \(max-width: 720px\)\s*\{[^}]*\.song-form-grid\s*\{[^}]*"
        r"grid-template-columns:\s*1fr\s*;",
        css, re.S)


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
    # the page reaches the same end by never naming a markup-parsing sink at all,
    # which also catches `el.innerHTML = job.title` (no interpolation to spot)
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML"):
        assert sink not in html, f"{sink} must not appear in the page"


def test_index_respects_reduced_motion():
    assert "prefers-reduced-motion" in _index_html()


def test_index_has_lyrics_review_hook():
    html = _index_html()
    assert 'id="lyrics-review"' in html
    assert 'id="warnings"' not in html


def test_index_review_uses_svg_not_canvas():
    html = _index_html()
    assert "createElementNS" in html
    assert "<canvas" not in html


def test_index_review_fetches_review_endpoint():
    html = _index_html()
    assert "/review" in html


def test_index_lyrics_review_uses_endash_for_time_ranges():
    html = _index_html()
    body = _fn_body(html, "missingSpanNode")
    assert "–" in body


def test_index_review_lays_out_words_in_time():
    """Words must be positioned by their sung timing (percentage-based), not
    packed as plain flow text, so notes can line up under their own word."""
    html = _index_html()
    body = _fn_body(html, "lineTimeline")
    assert "review-word" in body
    assert "style.left" in body


def test_lyrics_review_cache_clears_on_job_status_change():
    """Regression: switching between done/failed/running jobs must clear the
    cache so a previously-viewed done job re-fetches when re-selected."""
    html = _index_html()
    body = _fn_body(html, "renderLyricsReview")
    # The "failed" branch must clear the cache before rendering the placeholder
    assert re.search(
        r'if\s*\(\s*job\.status\s*===\s*"failed"\s*\)\s*\{[^}]*'
        r'delete\s+box\.dataset\.jobId',
        body, re.S), "failed branch must delete cache key"
    # The "not done" (queued/running) branch must clear the cache before rendering
    assert re.search(
        r'if\s*\(\s*job\.status\s*!==\s*"done"\s*\)\s*\{[^}]*'
        r'delete\s+box\.dataset\.jobId',
        body, re.S), "not done branch must delete cache key"


def test_index_has_copy_path_control():
    assert "copy-path" in _index_html()


def test_index_fetches_stage_order_from_api():
    html = _index_html()
    assert "/api/stages" in html, "stage list must come from the API, not be hardcoded"
    # Only the distinctive stage names — "grid", "align", "notes" and "pitch" can
    # legitimately appear in CSS properties or UI copy.
    for name in ("acquire", "separate", "transcribe", "package"):
        assert f'"{name}"' not in html, f"stage {name} appears hardcoded in the page"


def test_index_has_dragdrop_and_poll_control():
    html = _index_html()
    assert "dragover" in html and "drop" in html
    assert "clearInterval" in html or "pollTimer" in html


def _fn_body(html, name):
    """The source of a top-level `function name(...) {...}` in the page script."""
    start = html.index(f"function {name}(")
    return html[start:html.index("\n}\n", start)]


def _page_css(html):
    return html[html.index("<style>"):html.index("</style>")]


def _focus_flaws(html):
    """Ways the 3s poll could detach — and so blur — a focused node.

    Both the queue cards and #copy-path are focusable and both are rewritten on
    every poll, so each one has to be patched in place rather than rebuilt.
    """
    flaws = []

    queue = _fn_body(html, "renderJobs")
    if 'querySelectorAll(".job")' not in queue or "dataset.id" not in queue:
        flaws.append("queue cards are not matched to the payload by id")
    if re.search(r"replaceChildren\(\s*\.\.\.", queue):
        flaws.append("the queue is re-rendered wholesale, detaching every card")
    for line in re.findall(r".*insertBefore.*", queue):
        if not re.search(r"if \((\w+)\.children\[(\w+)\] !== (\w+)\) "
                         r"\1\.insertBefore\(\s*\3\s*,\s*\1\.children\[\2\]", line):
            flaws.append(f"a card is moved without checking it needs to be: {line.strip()}")

    out = _fn_body(html, "renderOutputPath")
    guard = re.search(r"if \((\w+) === (\w+)\) return;", out[:out.index("replaceChildren")])
    if guard is None:
        flaws.append("#copy-path is rebuilt without first checking whether it changed")
    elif not re.search(rf"(?:{guard[1]}|{guard[2]}) = outputPathDiv\.querySelector",
                       out[:guard.start()]):
        flaws.append("the #copy-path guard does not compare against what is rendered")
    return flaws


def test_polling_cannot_blur_a_focused_control():
    assert _focus_flaws(_index_html()) == []


def _now_bar_flaws(html):
    body = _fn_body(html, "renderNowBar")
    picked = re.search(r'const (\w+) = jobs\.find\(\w+ => \w+\.status === "running"\)', body)
    if picked is None:
        return ["the bar does not pick out the running job"]
    job, flaws = picked[1], []
    if not re.search(rf'classList\.toggle\("idle", !{job}\)', body):
        flaws.append("the idle class does not track whether a job is running")
    if f"renderStageTrack({job})" not in body:
        flaws.append("the stage track is not driven by the running job")
    if f"{job} ?" not in body:
        flaws.append("the now-bar title is not driven by the running job")
    return flaws


def test_now_bar_state_follows_the_running_job():
    assert _now_bar_flaws(_index_html()) == []


def _stale_detail_flaws(html):
    """A slow response for a since-deselected job must not overwrite the UI."""
    body = _fn_body(html, "refreshDetail")
    m = re.search(r"const (\w+) = selectedId;", body)
    if m is None:
        return ["refreshDetail does not snapshot selectedId before fetching"]
    if f"if ({m[1]} !== selectedId) return;" not in body:
        return ["refreshDetail does not bail when selectedId changed during the fetch"]
    return []


def test_stale_detail_response_cannot_override_a_newer_selection():
    assert _stale_detail_flaws(_index_html()) == []


def _pulse_flaws(html):
    css, flaws = _page_css(html), []
    if "@keyframes stage-pulse" not in css:
        flaws.append("the stage-pulse keyframes are missing")
    applied = re.search(r"([^{}]*)\{[^{}]*animation: stage-pulse[^{}]*\}", css)
    if applied is None:
        flaws.append("stage-pulse is defined but never applied")
    else:
        selector = applied[1].strip().splitlines()[-1].strip()
        if ".stage-active" not in selector or ".stage-dot" not in selector:
            flaws.append(f"the pulse is on {selector!r}, not the running stage's dot")
        if '"stage-active"' not in _fn_body(html, "renderStageTrack"):
            flaws.append("nothing ever gets the stage-active class")
    reduced = re.search(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n\}", css, re.S)
    if reduced is None or not re.search(r"\*[^{}]*\{[^{}]*animation: none !important",
                                        reduced[1]):
        flaws.append("reduced motion does not disable animation everywhere")
    return flaws


def test_running_stage_pulses_unless_motion_is_reduced():
    assert _pulse_flaws(_index_html()) == []


def _submit_body(html):
    start = html.index('form.addEventListener("submit"')
    return html[start:html.index("\n});\n", start)]


def _listener_body(html, prefix):
    start = html.index(prefix)
    return html[start:html.index("\n});\n", start)]


def _rebuild_flaws(html):
    """Ways rebuild mode could quietly stop working."""
    flaws, submit = [], _submit_body(html)
    if "rebuildOf ?" not in submit or "/rebuild" not in submit:
        flaws.append("the form always posts to /api/jobs, so edits create a new song")
    if "form.reset()" in submit or "exitRebuild()" in submit:
        flaws.append("a successful submission empties the form, so correcting "
                     "or varying the song just sent means retyping it")

    actions = _fn_body(html, "renderActions")
    if not re.search(r"if \(.*dataset\.jobId === job\.id.*\) return;", actions):
        flaws.append("the action row is rebuilt on every poll, blurring a focused button")
    if '"failed"' not in actions or '"done"' not in actions:
        flaws.append("the action row does not key off the job status")

    load = _fn_body(html, "loadInputs")
    if "/input" not in load or "form.elements[" not in load:
        flaws.append("rebuild mode does not prefill the form from the job's input")

    select = _fn_body(html, "selectJob")
    if "loadInputs(" not in select:
        flaws.append("selecting a queue entry does not bring its inputs back up")
    if "formDirty" not in select or "confirm(" not in select:
        flaws.append("selecting a queue entry discards unsaved edits without asking")

    if "if (rebuildOf === id) exitRebuild();" not in _fn_body(html, "deleteJob"):
        flaws.append("deleting the loaded job leaves the form pointed at it")

    # selecting a queue entry loads it, so the sidebar button is now the only
    # way back to a blank form
    create = _listener_body(html, 'createSong.addEventListener("click"')
    if "exitRebuild()" not in create:
        flaws.append("the queue's CREATE SONG button does not clear the form")
    if "formDirty" not in create or "confirm(" not in create:
        flaws.append("the queue's CREATE SONG button discards edits without asking")
    if "clearDetail(" not in create:
        flaws.append("CREATE SONG empties the form but leaves the previous "
                     "song's review and player sitting under it")
    if "composing" not in create or "composing" not in _fn_body(html, "refreshJobs"):
        flaws.append("the blank page CREATE SONG made is selected back over by "
                     "the next poll of the queue")

    mode = _fn_body(html, "setFormMode")
    if 'querySelector("details").open' not in mode:
        flaws.append("prefilled genre/year/language/bpm stay hidden behind the "
                     "closed Advanced summary, so they rebuild unseen")
    if "submitButton.disabled" not in _fn_body(html, "setSubmitLock"):
        flaws.append("the form lets you submit a rebuild the API will answer 409")
    return flaws


def _rebuild_full_and_delete_flaws(html):
    flaws = []
    actions = _fn_body(html, "renderActions")
    for action in ('"rebuild-full"', '"delete"'):
        if action not in actions:
            flaws.append(f"{action} is not offered from renderActions")

    rf = _fn_body(html, "rebuildFullJob")
    if "/rebuild-full" not in rf or 'method: "POST"' not in rf:
        flaws.append("rebuildFullJob does not POST to /rebuild-full")

    d = _fn_body(html, "deleteJob")
    confirm_idx = d.find("confirm(")
    fetch_idx = d.find("fetch(")
    if confirm_idx == -1:
        flaws.append("deleteJob does not confirm before deleting")
    elif fetch_idx != -1 and confirm_idx > fetch_idx:
        flaws.append("deleteJob fetches before confirming")
    if 'method: "DELETE"' not in d:
        flaws.append("deleteJob does not send a DELETE request")
    return flaws


def test_rebuild_full_and_delete_controls_are_wired_up():
    assert _rebuild_full_and_delete_flaws(_index_html()) == []


def _empty_state_flaws(html):
    refresh = _fn_body(html, "refreshJobs")
    if "clearDetail(" not in refresh:
        return ["refreshJobs never resets the panel when no job is selected"]
    return []


def test_review_panel_resets_when_no_job_is_selected():
    assert _empty_state_flaws(_index_html()) == []


def _playback_flaws(html):
    """Ways the review could stop tracking the media element."""
    flaws = []
    source = _fn_body(html, "setPlayerSource")
    if "media/audio" not in source:
        flaws.append("the player is not pointed at the job's packaged audio, "
                     "which is the file the game plays and the notes track")
    if "media/video" not in source:
        flaws.append("the packaged video is never shown alongside the audio")

    # the mp3 and the video carry the same recording; playing both is an echo
    if not re.search(r"<video[^>]*\bmuted\b", html):
        flaws.append("the background video is not muted, so its own audio "
                     "plays on top of the mp3")
    pump = _fn_body(html, "pumpPlayback")
    if "slaveVideo(" not in pump:
        flaws.append("the picture is never pulled back onto the audio clock, "
                     "so it drifts out of sync over a long song")

    if "requestAnimationFrame" not in _fn_body(html, "pumpPlayback"):
        flaws.append("sync is not driven by the frame loop, so the playhead "
                     "steps at timeupdate's ~4Hz instead of following the audio")

    if "media.paused" not in _fn_body(html, "focusActive"):
        flaws.append("the view is scrolled even while paused, fighting a reader "
                     "who scrolled away to look at something else")

    sync = _fn_body(html, "syncPlayback")
    if "clearEntry(" not in sync:
        flaws.append("the line the playhead left keeps its highlight")
    clear = _fn_body(html, "clearEntry")
    for name in ("review-word-sung", "review-word-active", "review-line-active"):
        if name not in clear:
            flaws.append(f"{name} is never taken off a line the playhead left")

    # every branch that gives up on rendering a review has to unload the media
    # too, or the previous song stays loaded and playable under the new one
    render = _fn_body(html, "renderLyricsReview")
    if render.count("resetPlayer()") < render.count("emptyNote("):
        flaws.append("a review the page cannot render leaves the previous "
                     "song's media loaded in the player")
    return flaws


def test_review_playback_tracks_the_media_element():
    assert _playback_flaws(_index_html()) == []


# each mutation is a plausible regression; the checker above must reject all of them
MUTATIONS = [
    (_focus_flaws, "queue: card moved unconditionally",
     "if (jobsDiv.children[i] !== card) jobsDiv.insertBefore",
     "jobsDiv.insertBefore"),
    (_focus_flaws, "queue: rebuilt wholesale",
     "jobsDiv.insertBefore(card, jobsDiv.children[i] ?? null)",
     "jobsDiv.replaceChildren(...jobs.map(jobCard))"),
    (_focus_flaws, "copy-path: unchanged panel rebuilt anyway",
     "if (path === shown) return;", ""),
    (_now_bar_flaws, "now bar: idle keyed off the queue instead of the running job",
     'classList.toggle("idle", !running)', 'classList.toggle("idle", !jobs.length)'),
    (_now_bar_flaws, "now bar: shows the newest job instead of the running one",
     'const running = jobs.find(j => j.status === "running")',
     "const running = jobs[0]"),
    (_stale_detail_flaws, "refreshDetail: stale-response guard removed",
     "if (id !== selectedId) return;", ""),
    (_pulse_flaws, "pulse: attached to the wrong stage",
     ".stage-active .stage-dot { color: var(--aqua); animation: stage-pulse",
     ".stage-pending .stage-dot { color: var(--aqua); animation: stage-pulse"),
    (_pulse_flaws, "pulse: reduced motion no longer wins",
     "animation: none !important", "animation: none"),
    (_rebuild_flaws, "rebuild: form posts to /api/jobs even in rebuild mode",
     'const url = rebuildOf ? `/api/jobs/${rebuildOf}/rebuild` : "/api/jobs";',
     'const url = "/api/jobs";'),
    (_rebuild_flaws, "rebuild: action row rebuilt on every poll",
     "if (reviewActions.dataset.jobId === job.id && shown.join() === wanted.join()) return;",
     ""),
    (_rebuild_full_and_delete_flaws, "delete: confirmation dialog removed",
     'if (!confirm("Delete this song? This removes all its files and cannot be undone.")) return;',
     ""),
    (_rebuild_full_and_delete_flaws, "rebuild-full: no longer offered on a done job",
     '["rebuild-full", "delete"]', '["delete"]'),
    (_rebuild_flaws, "prefill: selecting a queue entry no longer loads its inputs",
     "  if (reload) await loadInputs(id, JOBS.find(j => j.id === id)?.status);\n", ""),
    (_rebuild_flaws, "prefill: unsaved edits discarded without asking",
     '  if (reload && formDirty && !confirm("Discard your unsaved edits to this form?")) return;\n',
     ""),
    (_rebuild_flaws, "prefill: deleting the loaded job leaves the form pointed at it",
     "  if (rebuildOf === id) exitRebuild();\n", ""),
    (_rebuild_flaws, "prefill: advanced fields stay hidden behind a closed summary",
     'form.querySelector("details").open =', "const unusedAdvancedOpen ="),
    (_rebuild_flaws, "prefill: an in-progress job can be submitted for rebuild anyway",
     "submitButton.disabled = locked;", ""),
    (_rebuild_flaws, "submit: a successful submission empties the form again",
     "  rebuildOf = null;\n  formDirty = false;\n  setFormMode(null, null);\n  refreshJobs();",
     "  exitRebuild();\n  refreshJobs();"),
    (_rebuild_flaws, "create: the queue button no longer clears the form",
     "  exitRebuild();\n  clearDetail(\"nothing selected\");\n", ""),
    (_rebuild_flaws, "create: the review is left behind under a blank form",
     '  clearDetail("nothing selected");\n', ""),
    (_rebuild_flaws, "create: the next poll selects a song back into the blank page",
     "  composing = true;\n", ""),
    (_rebuild_flaws, "create: the queue button discards edits without asking",
     '  if (formDirty && !confirm("Discard your unsaved edits to this form?")) return;\n  composing = true;',
     "  composing = true;"),
    (_playback_flaws, "playback: sync falls back to the timeupdate event's ~4Hz",
     "syncFrame = media.paused ? null : requestAnimationFrame(pumpPlayback);",
     "syncFrame = null;"),
    (_playback_flaws, "playback: the view is scrolled even while paused",
     "if (!activeEntry || media.paused) return;", "if (!activeEntry) return;"),
    (_playback_flaws, "playback: the line the playhead left keeps its highlight",
     "    if (activeEntry) clearEntry(activeEntry);\n", ""),
    (_playback_flaws, "playback: a word stays lit after the playhead moves on",
     'w.el.classList.remove("review-word-sung", "review-word-active");',
     'w.el.classList.remove("review-word-sung");'),
    (_playback_flaws, "playback: an unrenderable review leaves the old song loaded",
     "    resetPlayer();\n", ""),
    (_playback_flaws, "playback: the player is not pointed at the packaged audio",
     'const src = available?.audio ? `/api/jobs/${job.id}/media/audio` : "";',
     'const src = "";'),
    (_playback_flaws, "playback: the background video plays its own audio too",
     '<video id="player-video" muted playsinline',
     '<video id="player-video" playsinline'),
    (_playback_flaws, "playback: the picture is never re-synced to the audio",
     "  slaveVideo(false);\n", ""),
    (_empty_state_flaws, "empty state: clearDetail no longer wired into refreshJobs",
     "if (selectedId) refreshDetail();\n  else clearDetail(composing ? \"nothing selected\" : undefined);",
     "if (selectedId) refreshDetail();"),
]


@pytest.mark.parametrize("check,name,old,new",
                         MUTATIONS, ids=[m[1] for m in MUTATIONS])
def test_guards_reject_plausible_regressions(check, name, old, new):
    html = _index_html()
    assert old in html, f"mutation no longer applies, rewrite it: {name}"
    assert check(html.replace(old, new)), f"{name} slipped past the check"


def test_stage_state_is_not_conveyed_by_colour_alone():
    html = _index_html()
    body = _fn_body(html, "renderStageTrack")
    assert 'setAttribute("aria-current", "step")' in body
    assert '"sr-only"' in body and ".sr-only {" in _page_css(html)
    for word in ('" done"', '" running"', '" pending"'):
        assert word in body, word


def test_narrow_stage_labels_stay_in_the_accessibility_tree():
    hidden = re.search(r"\.stage:not\(\.stage-active\) \.stage-label \{([^}]*)\}",
                       _page_css(_index_html()))
    assert hidden, "the compact stage track rule moved or was renamed"
    assert "display: none" not in hidden[1] and "visibility: hidden" not in hidden[1]


def test_rebuild_mode_reuses_the_one_form():
    html = _index_html()
    assert html.count('name="lyrics"') == 1    # one song form, not a rebuild copy
    assert _rebuild_flaws(html) == []


def test_index_has_rebuild_hooks():
    html = _index_html()
    for hook in ("review-actions", "review-lineage", "create-song",
                 'data-keep="video"', 'data-keep="cover"'):
        assert hook in html, hook


def test_rebuild_controls_are_not_offered_mid_run():
    """Both endpoints answer 409 while a job is queued or running."""
    actions = _fn_body(_index_html(), "renderActions")
    assert '"queued"' not in actions and '"running"' not in actions


def test_lineage_marker_survives_in_the_accessibility_tree():
    html = _index_html()
    body = _fn_body(html, "paintCard")
    assert "rebuilt_from" in body
    assert '" rebuilt"' in body
    assert '"sr-only"' in _fn_body(html, "jobCard")


def make_done_job(client, **overrides):
    data = {"title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"}
    data.update(overrides)
    job_id = client.post("/api/jobs", data=data).json()["id"]
    wait_status(client, job_id, "done")
    return job_id


def test_rerun_creates_a_linked_job(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    resp = client.post(f"/api/jobs/{parent}/rerun")
    assert resp.status_code == 201
    child = resp.json()["id"]
    assert child != parent
    job = wait_status(client, child, "done")
    assert job["rebuilt_from"] == parent


def test_rerun_copies_the_parents_input(tmp_path):
    seen = []

    def capture(inp, cfg, log=print, force_from=None, stages=None):
        seen.append(inp)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, capture)
    parent = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y",
        "genre": "pop", "year": "1998", "language": "en", "bpm": "120"}).json()["id"]
    wait_status(client, parent, "done")
    child = client.post(f"/api/jobs/{parent}/rerun").json()["id"]
    wait_status(client, child, "done")
    assert seen[0] == seen[1]


def test_rerun_of_a_failed_job_is_allowed(tmp_path):
    calls = []

    def flaky(inp, cfg, log=print, force_from=None, stages=None):
        calls.append(1)
        if len(calls) == 1:
            raise PipelineError("transient")
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, flaky)
    parent = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x",
        "youtube_url": "http://y"}).json()["id"]
    wait_status(client, parent, "failed")
    child = client.post(f"/api/jobs/{parent}/rerun").json()["id"]
    wait_status(client, child, "done")


def test_rerun_of_an_unknown_job_is_404(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.post("/api/jobs/nope/rerun").status_code == 404


def test_rerun_of_a_running_job_is_409(tmp_path):
    release = threading.Event()

    def blocking(inp, cfg, log=print, force_from=None, stages=None):
        release.wait(5)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, blocking)
    parent = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x",
        "youtube_url": "http://y"}).json()["id"]
    try:
        wait_status(client, parent, "running")
        assert client.post(f"/api/jobs/{parent}/rerun").status_code == 409
    finally:
        release.set()
    wait_status(client, parent, "done")


def test_rebuilt_from_is_null_for_an_ordinary_job(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}").json()["rebuilt_from"] is None
    listed = {j["id"]: j for j in client.get("/api/jobs").json()}
    assert listed[job_id]["rebuilt_from"] is None


def test_input_echoes_the_submitted_form(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    job_id = make_done_job(client, genre="pop", year="1998", language="en", bpm="120")
    body = client.get(f"/api/jobs/{job_id}/input").json()
    assert body == {"title": "T", "artist": "A", "lyrics": "x",
                    "youtube_url": "http://y", "genre": "pop", "year": 1998,
                    "language": "en", "bpm": 120.0,
                    "video_filename": None, "cover_filename": None}


def test_input_keys_match_the_form_field_names(tmp_path):
    """The page assigns straight into form.elements[name]; a rename breaks prefill."""
    html = _index_html()
    names = set(re.findall(r'name="([a-z_]+)"', html))
    client, _ = make_client(tmp_path, ok_runner)
    body = client.get(f"/api/jobs/{make_done_job(client)}/input").json()
    assert set(body) - names == {"video_filename", "cover_filename"}


def test_input_reports_bare_filenames_only(tmp_path):
    client, cfg = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4"),
                               "cover": ("c.jpg", io.BytesIO(b"img"), "image/jpeg")})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "done")
    body = client.get(f"/api/jobs/{job_id}/input").json()
    assert body["video_filename"] == "v.mp4"
    assert body["cover_filename"] == "c.jpg"
    assert "/" not in body["video_filename"]
    assert str(tmp_path) not in resp.text


def test_input_of_an_unknown_job_is_404(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.get("/api/jobs/nope/input").status_code == 404


def capture_runner_factory(seen):
    def capture(inp, cfg, log=print, force_from=None, stages=None):
        seen.append({"inp": inp, "force_from": force_from})
        return PipelineResult(output_dir=cfg.output_dir / song_slug(inp))
    return capture


def rebuild_form(**overrides):
    """The form as the page posts it — empty fields are stripped, not sent blank."""
    data = {"title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"}
    data.update(overrides)
    return {k: v for k, v in data.items() if v != ""}


def test_rebuild_forces_the_stage_the_edit_invalidated(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, capture_runner_factory(seen))
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(lyrics="new words")).json()["id"]
    wait_status(client, child, "done")
    assert seen[1]["force_from"] == "align"
    assert seen[1]["inp"].lyrics_text == "new words"


def test_rebuild_with_no_edits_forces_nothing(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, capture_runner_factory(seen))
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form()).json()["id"]
    wait_status(client, child, "done")
    assert seen[1]["force_from"] is None


def test_rebuild_renames_the_work_dir_on_a_retitle(tmp_path):
    seen = []
    client, cfg = make_client(tmp_path, capture_runner_factory(seen))
    parent = make_done_job(client)
    (cfg.work_dir / "A - T").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "A - T" / "audio.wav").write_bytes(b"x")
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(title="T2")).json()["id"]
    wait_status(client, child, "done")
    assert not (cfg.work_dir / "A - T").exists()
    assert (cfg.work_dir / "A - T2" / "audio.wav").exists()
    assert seen[1]["force_from"] == "grid"


def test_rebuild_fails_the_job_on_a_work_dir_collision(tmp_path):
    client, cfg = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    (cfg.work_dir / "A - T").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "A - T2").mkdir(parents=True)
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(title="T2")).json()["id"]
    job = wait_status(client, child, "failed")
    assert "already exists" in job["error"]


def test_rebuild_logs_what_changed_without_faking_a_stage(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(year="1998")).json()["id"]
    job = wait_status(client, child, "done")
    rebuild_lines = [l for l in job["log"] if l.startswith("[rebuild]")]
    assert len(rebuild_lines) == 1
    line = rebuild_lines[0]
    assert "grid" in line and "year" in line
    assert not STAGE_LINE.match(line)


def test_rebuild_inherits_the_parents_source(tmp_path):
    seen = []
    client, cfg = make_client(tmp_path, capture_runner_factory(seen))
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4")})
    parent = resp.json()["id"]
    wait_status(client, parent, "done")
    (cfg.work_dir / "A - T").mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "audio.wav"):
        (cfg.work_dir / "A - T" / name).write_bytes(b"x")
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(youtube_url="", year="1998")).json()["id"]
    wait_status(client, child, "done")
    assert seen[1]["inp"].video_path == seen[0]["inp"].video_path


def test_rebuild_inherits_the_parents_cover(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, capture_runner_factory(seen))
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x",
                                          "youtube_url": "http://y"},
                        files={"cover": ("c.jpg", io.BytesIO(b"img"), "image/jpeg")})
    parent = resp.json()["id"]
    wait_status(client, parent, "done")
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(year="1998")).json()["id"]
    wait_status(client, child, "done")
    assert seen[1]["inp"].cover_path == seen[0]["inp"].cover_path


def test_rebuild_rejects_a_missing_upload_that_would_be_needed(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    resp = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                        files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4")})
    parent = resp.json()["id"]
    wait_status(client, parent, "done")
    resp = client.post(f"/api/jobs/{parent}/rebuild",
                       data=rebuild_form(youtube_url="", language="de"))
    assert resp.status_code == 400
    assert "attach the video file again" in resp.json()["detail"]


def test_rebuild_accepts_a_replacement_source(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, capture_runner_factory(seen))
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild", data=rebuild_form(youtube_url=""),
                        files={"video": ("new.mp4", io.BytesIO(b"d"), "video/mp4")}
                        ).json()["id"]
    wait_status(client, child, "done")
    assert seen[1]["inp"].youtube_url is None
    assert seen[1]["inp"].video_path.name == "new.mp4"
    assert seen[1]["force_from"] == "acquire"


def test_rebuild_rejects_two_sources(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    resp = client.post(f"/api/jobs/{parent}/rebuild", data=rebuild_form(),
                       files={"video": ("v.mp4", io.BytesIO(b"d"), "video/mp4")})
    assert resp.status_code == 400


def test_rebuild_rejects_a_bad_bpm(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    resp = client.post(f"/api/jobs/{parent}/rebuild", data=rebuild_form(bpm="0"))
    assert resp.status_code == 400


def test_rebuild_of_a_running_job_is_409(tmp_path):
    release = threading.Event()

    def blocking(inp, cfg, log=print, force_from=None, stages=None):
        release.wait(5)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, blocking)
    parent = client.post("/api/jobs", data=rebuild_form()).json()["id"]
    try:
        wait_status(client, parent, "running")
        resp = client.post(f"/api/jobs/{parent}/rebuild", data=rebuild_form(year="1998"))
        assert resp.status_code == 409
    finally:
        release.set()
    wait_status(client, parent, "done")


def test_rebuild_of_an_unknown_job_is_404(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.post("/api/jobs/nope/rebuild", data=rebuild_form()).status_code == 404


def test_rebuild_full_forces_every_stage_from_the_first(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, capture_runner_factory(seen))
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild-full").json()["id"]
    wait_status(client, child, "done")
    assert seen[1]["force_from"] == STAGE_ORDER[0]
    assert seen[1]["inp"] == seen[0]["inp"]


def test_rebuild_full_creates_a_linked_job(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    resp = client.post(f"/api/jobs/{parent}/rebuild-full")
    assert resp.status_code == 201
    child = resp.json()["id"]
    assert child != parent
    job = wait_status(client, child, "done")
    assert job["rebuilt_from"] == parent


def test_rebuild_full_logs_that_it_is_a_forced_rebuild(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild-full").json()["id"]
    job = wait_status(client, child, "done")
    rebuild_lines = [l for l in job["log"] if l.startswith("[rebuild]")]
    assert len(rebuild_lines) == 1
    assert STAGE_ORDER[0] in rebuild_lines[0]


def test_rebuild_full_of_an_unknown_job_is_404(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.post("/api/jobs/nope/rebuild-full").status_code == 404


def test_rebuild_full_of_a_running_job_is_409(tmp_path):
    release = threading.Event()

    def blocking(inp, cfg, log=print, force_from=None, stages=None):
        release.wait(5)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, blocking)
    parent = client.post("/api/jobs", data=rebuild_form()).json()["id"]
    try:
        wait_status(client, parent, "running")
        assert client.post(f"/api/jobs/{parent}/rebuild-full").status_code == 409
    finally:
        release.set()
    wait_status(client, parent, "done")


def slug_runner(inp, cfg, log=print, force_from=None, stages=None):
    out = cfg.output_dir / song_slug(inp)
    out.mkdir(parents=True, exist_ok=True)
    (out / "song.txt").write_text(f"#TITLE:{inp.title}\n")
    return PipelineResult(output_dir=out)


def test_rebuild_removes_the_output_it_replaced(tmp_path):
    client, cfg = make_client(tmp_path, slug_runner)
    parent = make_done_job(client)
    assert (cfg.output_dir / "A - T" / "song.txt").exists()
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(title="T2")).json()["id"]
    job = wait_status(client, child, "done")
    assert (cfg.output_dir / "A - T2" / "song.txt").exists()
    assert not (cfg.output_dir / "A - T").exists()
    assert any("superseded" in line for line in job["log"])


def test_rebuild_keeps_the_output_when_the_slug_is_unchanged(tmp_path):
    client, cfg = make_client(tmp_path, slug_runner)
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(year="1998")).json()["id"]
    wait_status(client, child, "done")
    assert (cfg.output_dir / "A - T" / "song.txt").exists()


def test_rebuild_of_a_failed_parent_removes_nothing(tmp_path):
    calls = []

    def flaky(inp, cfg, log=print, force_from=None, stages=None):
        calls.append(1)
        if len(calls) == 1:
            raise PipelineError("boom")
        return slug_runner(inp, cfg, log=log)

    client, cfg = make_client(tmp_path, flaky)
    parent = client.post("/api/jobs", data=rebuild_form()).json()["id"]
    wait_status(client, parent, "failed")
    (cfg.output_dir / "A - T").mkdir(parents=True)
    (cfg.output_dir / "A - T" / "keep.txt").write_text("mine")
    child = client.post(f"/api/jobs/{parent}/rebuild",
                        data=rebuild_form(title="T2")).json()["id"]
    wait_status(client, child, "done")
    assert (cfg.output_dir / "A - T" / "keep.txt").exists()


def test_delete_removes_work_and_output_and_the_job(tmp_path):
    client, cfg = make_client(tmp_path, slug_runner)
    job_id = make_done_job(client)
    (cfg.work_dir / "A - T").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "A - T" / "audio.wav").write_bytes(b"x")
    assert (cfg.output_dir / "A - T").exists()

    resp = client.delete(f"/api/jobs/{job_id}")

    assert resp.status_code == 204
    assert not (cfg.work_dir / "A - T").exists()
    assert not (cfg.output_dir / "A - T").exists()
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert job_id not in [j["id"] for j in client.get("/api/jobs").json()]


def test_delete_tolerates_a_work_dir_that_is_already_gone(tmp_path):
    client, cfg = make_client(tmp_path, slug_runner)
    job_id = make_done_job(client)
    shutil.rmtree(cfg.work_dir / "A - T")
    resp = client.delete(f"/api/jobs/{job_id}")
    assert resp.status_code == 204


def test_delete_of_an_unknown_job_is_404(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.delete("/api/jobs/nope").status_code == 404


def test_delete_of_a_running_job_is_409(tmp_path):
    release = threading.Event()

    def blocking(inp, cfg, log=print, force_from=None, stages=None):
        release.wait(5)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, blocking)
    parent = client.post("/api/jobs", data=rebuild_form()).json()["id"]
    try:
        wait_status(client, parent, "running")
        assert client.delete(f"/api/jobs/{parent}").status_code == 409
    finally:
        release.set()
    wait_status(client, parent, "done")
    assert client.get(f"/api/jobs/{parent}").status_code == 200


def test_delete_of_a_job_with_an_active_rebuild_is_409(tmp_path):
    """A done job's slug must not be deleted out from under a rebuild in flight."""
    release = threading.Event()
    calls = []

    def runner(inp, cfg, log=print, force_from=None, stages=None):
        calls.append(1)
        if len(calls) > 1:
            release.wait(5)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, runner)
    parent = make_done_job(client)
    child = client.post(f"/api/jobs/{parent}/rebuild-full").json()["id"]
    try:
        wait_status(client, child, "running")
        assert client.delete(f"/api/jobs/{parent}").status_code == 409
    finally:
        release.set()
    wait_status(client, child, "done")
    assert client.get(f"/api/jobs/{parent}").status_code == 200


def test_a_plain_job_never_removes_anything(tmp_path):
    client, cfg = make_client(tmp_path, slug_runner)
    make_done_job(client)
    stray = cfg.output_dir / "A - Other"
    stray.mkdir(parents=True)
    make_done_job(client, title="T2")
    assert stray.exists()


def test_discover_jobs_ignores_a_work_dir_without_a_sidecar(tmp_path):
    from songtrace.web import _discover_jobs
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work", output_dir=tmp_path / "out")
    (cfg.work_dir / "A - T").mkdir(parents=True)
    assert _discover_jobs(cfg) == []


def test_discover_jobs_reconstructs_a_completed_song(tmp_path):
    from songtrace.web import _discover_jobs
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work", output_dir=tmp_path / "out")
    client = TestClient(create_app(cfg, runner=slug_runner))
    make_done_job(client)

    discovered = _discover_jobs(cfg)

    assert len(discovered) == 1
    job = discovered[0]
    assert job.title == "T" and job.artist == "A" and job.status == "done"
    assert job.output_dir == str(cfg.output_dir / "A - T")
    assert job.error is None


def test_discover_jobs_reconstructs_a_song_whose_output_is_missing_as_failed(tmp_path):
    from songtrace.web import _discover_jobs
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work", output_dir=tmp_path / "out")
    client = TestClient(create_app(cfg, runner=slug_runner))
    make_done_job(client)
    shutil.rmtree(cfg.output_dir / "A - T")

    discovered = _discover_jobs(cfg)

    assert len(discovered) == 1
    assert discovered[0].status == "failed"
    assert discovered[0].error == "interrupted before completion"
    assert discovered[0].output_dir is None


def test_discovered_jobs_are_visible_from_a_fresh_app(tmp_path):
    cfg = Config(whisper_url="u", work_dir=tmp_path / "work", output_dir=tmp_path / "out")
    client1 = TestClient(create_app(cfg, runner=slug_runner))
    make_done_job(client1)

    client2 = TestClient(create_app(cfg, runner=slug_runner))
    jobs = client2.get("/api/jobs").json()
    assert len(jobs) == 1
    assert jobs[0]["title"] == "T" and jobs[0]["status"] == "done"


def review_runner_factory(notes, dropped_lines=None, missing_spans=None, warnings=None,
                          words=None):
    def runner(inp, cfg, log=print, force_from=None, stages=None):
        work = cfg.work_dir / song_slug(inp)
        work.mkdir(parents=True, exist_ok=True)
        (work / "notes_sec.json").write_text(json.dumps({
            "notes": notes, "breaks": [], "warnings": warnings or []}))
        if dropped_lines is not None or missing_spans is not None:
            (work / "align_meta.json").write_text(json.dumps({
                "dropped_lines": dropped_lines or [],
                "missing_spans": missing_spans or []}))
        if words is not None:
            (work / "alignment.json").write_text(json.dumps(words))
        out = cfg.output_dir / song_slug(inp)
        out.mkdir(parents=True, exist_ok=True)
        return PipelineResult(output_dir=out, warnings=warnings or [])
    return runner


def test_review_endpoint_groups_notes_by_line(tmp_path):
    notes = [
        {"start": 1.0, "duration": 0.4, "pitch": 0, "text": "hi",
         "low_confidence": False, "line_index": 0},
        {"start": 5.0, "duration": 0.3, "pitch": 2, "text": "there",
         "low_confidence": True, "line_index": 1},
    ]
    client, _ = make_client(tmp_path, review_runner_factory(
        notes, dropped_lines=[2], missing_spans=[[8.0, 9.5]]))
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "hi\nthere\ngone",
        "youtube_url": "http://y"})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "done")
    body = client.get(f"/api/jobs/{job_id}/review").json()
    assert body["lines"] == [
        {"index": 0, "text": "hi", "dropped": False,
         "notes": [{"start": 1.0, "end": 1.4, "pitch": 0, "low_confidence": False}],
         "words": []},
        {"index": 1, "text": "there", "dropped": False,
         "notes": [{"start": 5.0, "end": 5.3, "pitch": 2, "low_confidence": True}],
         "words": []},
        {"index": 2, "text": "gone", "dropped": True, "notes": [], "words": []},
    ]
    assert body["missing_spans"] == [{"start": 8.0, "end": 9.5}]


def test_review_endpoint_includes_per_word_timing(tmp_path):
    notes = [
        {"start": 1.0, "duration": 0.4, "pitch": 0, "text": "hi",
         "low_confidence": False, "line_index": 0},
        {"start": 5.0, "duration": 0.3, "pitch": 2, "text": "there",
         "low_confidence": True, "line_index": 1},
    ]
    words = [
        {"word": "hi", "start": 1.0, "end": 1.4, "score": 0.9, "line_index": 0},
        {"word": "there", "start": 5.0, "end": 5.3, "score": 0.9, "line_index": 1},
    ]
    client, _ = make_client(tmp_path, review_runner_factory(
        notes, dropped_lines=[2], missing_spans=[[8.0, 9.5]], words=words))
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "hi\nthere\ngone",
        "youtube_url": "http://y"})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "done")
    body = client.get(f"/api/jobs/{job_id}/review").json()
    assert body["lines"][0]["words"] == [{"text": "hi", "start": 1.0, "end": 1.4}]
    assert body["lines"][1]["words"] == [{"text": "there", "start": 5.0, "end": 5.3}]
    assert body["lines"][2]["dropped"] is True
    assert body["lines"][2]["words"] == []


def test_review_of_a_failed_job_is_404(tmp_path):
    def bad_runner(*a, **kw):
        raise PipelineError("boom")
    client, _ = make_client(tmp_path, bad_runner)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"})
    job_id = resp.json()["id"]
    wait_status(client, job_id, "failed")
    assert client.get(f"/api/jobs/{job_id}/review").status_code == 404


def test_review_of_a_running_job_is_409(tmp_path):
    release = threading.Event()

    def blocking(inp, cfg, log=print, force_from=None, stages=None):
        release.wait(5)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, blocking)
    resp = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"})
    job_id = resp.json()["id"]
    try:
        wait_status(client, job_id, "running")
        assert client.get(f"/api/jobs/{job_id}/review").status_code == 409
    finally:
        release.set()
    wait_status(client, job_id, "done")


def test_review_of_an_unknown_job_is_404(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    assert client.get("/api/jobs/nope/review").status_code == 404


def test_review_missing_when_pipeline_never_wrote_notes(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/review").status_code == 404


# --- review playback -------------------------------------------------------

MEDIA_FILES = {"A - T.mp3": b"ID3audio", "A - T.mp4": b"\0\0\0\x18ftypvideo",
               "A - T.jpg": b"\xff\xd8cover"}
MEDIA_HEADERS = [("TITLE", "T"), ("ARTIST", "A"), ("MP3", "A - T.mp3"),
                 ("COVER", "A - T.jpg"), ("VIDEO", "A - T.mp4")]


def media_runner_factory(files=None, headers=None, write_txt=True):
    """A run that leaves behind a song folder, the way package does."""
    def runner(inp, cfg, log=print, force_from=None, stages=None):
        slug = song_slug(inp)
        out = cfg.output_dir / slug
        out.mkdir(parents=True, exist_ok=True)
        for name, body in (MEDIA_FILES if files is None else files).items():
            (out / name).write_bytes(body)
        if write_txt:
            rows = [f"#{k}:{v}" for k, v in
                    (MEDIA_HEADERS if headers is None else headers)]
            (out / f"{slug}.txt").write_text("\n".join(rows) + "\n: 0 4 0 la\nE\n",
                                             encoding="utf-8")
        return PipelineResult(output_dir=out)
    return runner


def test_media_serves_the_files_the_song_txt_names(tmp_path):
    client, _ = make_client(tmp_path, media_runner_factory())
    job_id = make_done_job(client)
    for kind, name in (("audio", "A - T.mp3"), ("video", "A - T.mp4"),
                       ("cover", "A - T.jpg")):
        resp = client.get(f"/api/jobs/{job_id}/media/{kind}")
        assert resp.status_code == 200, kind
        assert resp.content == MEDIA_FILES[name], kind


def test_media_serves_whatever_the_header_was_rewritten_to(tmp_path):
    """package transcodes to .mp4 and rewrites #VIDEO; the extension is not fixed."""
    files = {**MEDIA_FILES, "A - T.webm": b"transcoded"}
    headers = [(k, "A - T.webm" if k == "VIDEO" else v) for k, v in MEDIA_HEADERS]
    client, _ = make_client(tmp_path, media_runner_factory(files, headers))
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/media/video").content == b"transcoded"


def test_media_supports_range_requests(tmp_path):
    """Without 206 the browser cannot seek, which is the point of the player."""
    client, _ = make_client(tmp_path, media_runner_factory())
    job_id = make_done_job(client)
    resp = client.get(f"/api/jobs/{job_id}/media/video", headers={"Range": "bytes=0-3"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == f"bytes 0-3/{len(MEDIA_FILES['A - T.mp4'])}"
    assert resp.content == MEDIA_FILES["A - T.mp4"][:4]


def test_media_of_an_unknown_kind_is_404(tmp_path):
    client, _ = make_client(tmp_path, media_runner_factory())
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/media/instrumental").status_code == 404


def test_media_refuses_a_header_pointing_outside_the_song_folder(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")
    headers = [(k, "../../secret.txt" if k == "MP3" else v) for k, v in MEDIA_HEADERS]
    client, _ = make_client(tmp_path, media_runner_factory(headers=headers))
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/media/audio").status_code == 404


def test_media_refuses_an_absolute_header_path(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")
    headers = [(k, str(secret) if k == "MP3" else v) for k, v in MEDIA_HEADERS]
    client, _ = make_client(tmp_path, media_runner_factory(headers=headers))
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/media/audio").status_code == 404


def test_media_404s_when_the_named_file_is_missing(tmp_path):
    client, _ = make_client(tmp_path, media_runner_factory(files={}))
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/media/audio").status_code == 404


def test_media_404s_when_the_header_was_dropped(tmp_path):
    """package strips #COVER when there is no usable cover image."""
    headers = [(k, v) for k, v in MEDIA_HEADERS if k != "COVER"]
    client, _ = make_client(tmp_path, media_runner_factory(headers=headers))
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/media/cover").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/media/audio").status_code == 200


def test_media_404s_when_there_is_no_song_txt(tmp_path):
    client, _ = make_client(tmp_path, media_runner_factory(write_txt=False))
    job_id = make_done_job(client)
    assert client.get(f"/api/jobs/{job_id}/media/audio").status_code == 404


def test_media_of_a_failed_job_is_404(tmp_path):
    def bad_runner(*a, **kw):
        raise PipelineError("boom")
    client, _ = make_client(tmp_path, bad_runner)
    job_id = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"}).json()["id"]
    wait_status(client, job_id, "failed")
    assert client.get(f"/api/jobs/{job_id}/media/audio").status_code == 404


def test_media_of_a_running_job_is_409(tmp_path):
    release = threading.Event()

    def blocking(inp, cfg, log=print, force_from=None, stages=None):
        release.wait(5)
        return PipelineResult(output_dir=cfg.output_dir / "X")

    client, _ = make_client(tmp_path, blocking)
    job_id = client.post("/api/jobs", data={
        "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"}).json()["id"]
    try:
        wait_status(client, job_id, "running")
        assert client.get(f"/api/jobs/{job_id}/media/audio").status_code == 409
    finally:
        release.set()
    wait_status(client, job_id, "done")


def test_media_of_an_unknown_job_is_404(tmp_path):
    client, _ = make_client(tmp_path, media_runner_factory())
    assert client.get("/api/jobs/nope/media/audio").status_code == 404


def test_review_reports_which_media_the_player_can_load(tmp_path):
    notes = [{"start": 1.0, "duration": 0.4, "pitch": 0, "text": "hi",
              "low_confidence": False, "line_index": 0}]
    review = review_runner_factory(notes)
    media = media_runner_factory(headers=[(k, v) for k, v in MEDIA_HEADERS
                                          if k != "COVER"])

    def runner(inp, cfg, **kw):
        media(inp, cfg, **kw)
        return review(inp, cfg, **kw)

    client, _ = make_client(tmp_path, runner)
    job_id = make_done_job(client, lyrics="hi")
    body = client.get(f"/api/jobs/{job_id}/review").json()
    assert body["media"] == {"audio": True, "video": True, "cover": False}


# --- restarting a rebuild from a chosen stage ------------------------------

def force_from_runner_factory(seen):
    def runner(inp, cfg, log=print, force_from=None, stages=None):
        seen.append(force_from)
        out = cfg.output_dir / song_slug(inp)
        out.mkdir(parents=True, exist_ok=True)
        return PipelineResult(output_dir=out)
    return runner


def rebuild_with(client, job_id, **extra):
    data = {"title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"}
    data.update(extra)
    return client.post(f"/api/jobs/{job_id}/rebuild", data=data)


def test_rebuild_without_edits_restarts_from_the_stage_asked_for(tmp_path):
    """The point of the control: a change in songtrace invalidates no field."""
    seen = []
    client, _ = make_client(tmp_path, force_from_runner_factory(seen))
    parent = make_done_job(client)
    resp = rebuild_with(client, parent, force_from="package")
    assert resp.status_code == 201
    wait_status(client, resp.json()["id"], "done")
    assert seen[-1] == "package"


def test_rebuild_without_a_stage_still_plans_from_the_edit(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, force_from_runner_factory(seen))
    parent = make_done_job(client)
    resp = rebuild_with(client, parent, lyrics="different words")
    wait_status(client, resp.json()["id"], "done")
    assert seen[-1] == "align"


def test_an_edit_wins_when_it_invalidates_more_than_the_stage_asked_for(tmp_path):
    """A choice may widen what is redone, never narrow it below correctness."""
    seen = []
    client, _ = make_client(tmp_path, force_from_runner_factory(seen))
    parent = make_done_job(client)
    resp = rebuild_with(client, parent, lyrics="different words",
                        force_from="package")
    wait_status(client, resp.json()["id"], "done")
    assert seen[-1] == "align"


def test_the_stage_asked_for_wins_when_it_is_earlier_than_the_edit(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, force_from_runner_factory(seen))
    parent = make_done_job(client)
    resp = rebuild_with(client, parent, title="Renamed", force_from="separate")
    wait_status(client, resp.json()["id"], "done")
    assert seen[-1] == "separate"


def test_rebuild_rejects_a_stage_that_is_not_in_the_pipeline(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    parent = make_done_job(client)
    resp = rebuild_with(client, parent, force_from="deechoing")
    assert resp.status_code == 400
    assert "deechoing" in resp.json()["detail"]


def test_a_blank_stage_is_the_same_as_not_asking(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, force_from_runner_factory(seen))
    parent = make_done_job(client)
    resp = rebuild_with(client, parent, force_from="")
    wait_status(client, resp.json()["id"], "done")
    assert seen[-1] is None


def test_rebuild_from_scratch_still_restarts_at_the_first_stage(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, force_from_runner_factory(seen))
    parent = make_done_job(client)
    resp = client.post(f"/api/jobs/{parent}/rebuild-full")
    wait_status(client, resp.json()["id"], "done")
    assert seen[-1] == STAGE_ORDER[0]


def test_the_restart_control_offers_every_stage():
    html = _index_html()
    assert 'name="force_from"' in html
    stages = _fn_body(html, "loadStages")
    assert "force_from" in stages and "STAGES.map" in stages
    mode = _fn_body(html, "setFormMode")
    assert "restart-field" in mode, "the control is shown outside rebuild mode"


def test_a_textareas_crlf_does_not_read_as_a_lyrics_edit(tmp_path):
    """Browsers submit textareas with CRLF; the stored lyrics use LF."""
    seen = []
    client, _ = make_client(tmp_path, force_from_runner_factory(seen))
    parent = make_done_job(client, lyrics="one\ntwo")
    resp = rebuild_with(client, parent, lyrics="one\r\ntwo")
    wait_status(client, resp.json()["id"], "done")
    assert seen[-1] is None, "the rebuild redid the alignment for nothing"
    body = client.get(f"/api/jobs/{resp.json()['id']}/input").json()
    assert body["lyrics"] == "one\ntwo"


def test_submitting_crlf_lyrics_stores_them_with_lf(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    job_id = make_done_job(client, lyrics="one\r\ntwo")
    assert client.get(f"/api/jobs/{job_id}/input").json()["lyrics"] == "one\ntwo"


def test_rebuild_all_rebuilds_every_song_from_the_stage(tmp_path):
    seen = []
    client, _ = make_client(tmp_path, capture_runner_factory(seen))
    first = make_done_job(client, title="One")
    second = make_done_job(client, title="Two")
    resp = client.post("/api/rebuild-all", data={"force_from": "pitch"})
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["ids"]) == 2 and body["skipped"] == []
    jobs = [wait_status(client, i, "done") for i in body["ids"]]
    assert {j["rebuilt_from"] for j in jobs} == {first, second}
    assert [s["force_from"] for s in seen[2:]] == ["pitch", "pitch"]
    assert [s["inp"] for s in seen[2:]] == [s["inp"] for s in seen[:2]]


def test_rebuild_all_skips_jobs_that_were_already_rebuilt(tmp_path):
    client, _ = make_client(tmp_path, capture_runner_factory([]))
    parent = make_done_job(client)
    renamed = client.post(f"/api/jobs/{parent}/rebuild",
                          data=rebuild_form(title="Renamed")).json()["id"]
    wait_status(client, renamed, "done")
    body = client.post("/api/rebuild-all", data={"force_from": "notes"}).json()
    assert len(body["ids"]) == 1
    job = wait_status(client, body["ids"][0], "done")
    assert job["rebuilt_from"] == renamed


def test_rebuild_all_takes_the_latest_job_for_a_song(tmp_path):
    client, _ = make_client(tmp_path, capture_runner_factory([]))
    make_done_job(client)
    latest = make_done_job(client)
    body = client.post("/api/rebuild-all", data={"force_from": "grid"}).json()
    assert len(body["ids"]) == 1
    assert wait_status(client, body["ids"][0], "done")["rebuilt_from"] == latest


def test_rebuild_all_skips_songs_in_progress(tmp_path):
    gate = threading.Event()

    def blocking_runner(inp, cfg, log=print, force_from=None, stages=None):
        gate.wait(timeout=5)
        return PipelineResult(output_dir=cfg.output_dir / song_slug(inp))

    client, _ = make_client(tmp_path, blocking_runner)
    try:
        busy = client.post("/api/jobs", data={
            "title": "T", "artist": "A", "lyrics": "x", "youtube_url": "http://y"}).json()["id"]
        body = client.post("/api/rebuild-all", data={"force_from": "pitch"}).json()
        assert body == {"ids": [], "skipped": [{"id": busy, "reason": "still in progress"}]}
    finally:
        gate.set()


def test_rebuild_all_from_acquire_skips_a_song_whose_upload_is_gone(tmp_path):
    client, _ = make_client(tmp_path, capture_runner_factory([]))
    job_id = client.post("/api/jobs", data={"title": "T", "artist": "A", "lyrics": "x"},
                         files={"video": ("v.mp4", io.BytesIO(b"data"), "video/mp4")}
                         ).json()["id"]
    wait_status(client, job_id, "done")
    shutil.rmtree(tmp_path / "work" / "uploads", ignore_errors=True)
    body = client.post("/api/rebuild-all", data={"force_from": "acquire"}).json()
    assert body["ids"] == []
    assert body["skipped"][0]["id"] == job_id
    assert "upload" in body["skipped"][0]["reason"]


def test_rebuild_all_rejects_an_unknown_stage(tmp_path):
    client, _ = make_client(tmp_path, ok_runner)
    make_done_job(client)
    assert client.post("/api/rebuild-all", data={"force_from": "nope"}).status_code == 400
    assert client.post("/api/rebuild-all", data={}).status_code == 422
