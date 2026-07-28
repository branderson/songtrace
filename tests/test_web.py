import io
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


def test_index_parses_warnings_with_endash():
    html = _index_html()
    assert "–" in html, "warning parser must handle the en-dash separator"
    assert "data-start" in html and "data-end" in html


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


def _rebuild_flaws(html):
    """Ways rebuild mode could quietly stop working."""
    flaws, submit = [], _submit_body(html)
    if "rebuildOf ?" not in submit or "/rebuild" not in submit:
        flaws.append("the form always posts to /api/jobs, so edits create a new song")

    actions = _fn_body(html, "renderActions")
    if not re.search(r"if \(.*dataset\.jobId === job\.id.*\) return;", actions):
        flaws.append("the action row is rebuilt on every poll, blurring a focused button")
    if '"failed"' not in actions or '"done"' not in actions:
        flaws.append("the action row does not key off the job status")

    start = _fn_body(html, "startRebuild")
    if "/input" not in start or "form.elements[" not in start:
        flaws.append("rebuild mode does not prefill the form from the job's input")
    return flaws


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
    assert html.count("<form") == 1
    assert _rebuild_flaws(html) == []


def test_index_has_rebuild_hooks():
    html = _index_html()
    for hook in ("review-actions", "review-lineage", "cancel-rebuild",
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
