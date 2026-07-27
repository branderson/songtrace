import httpx
from pathlib import Path
from karaoke_maker.config import Config
from karaoke_maker.whisper_client import TranscriptSegment, transcribe

def test_transcribe_parses_segments(tmp_path):
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"RIFF")
    seen = {}
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={
            "text": "hello world",
            "segments": [{"start": 0.5, "end": 2.0, "text": " hello world"}],
        })
    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Config(whisper_url="http://w:9000/v1", whisper_api_key="sekrit")
    segs = transcribe(cfg, audio, client=client)
    assert segs == [TranscriptSegment(start=0.5, end=2.0, text="hello world")]
    assert seen["url"] == "http://w:9000/v1/audio/transcriptions"
    assert seen["auth"] == "Bearer sekrit"


def test_transcribe_maps_language_to_iso_code(tmp_path):
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"RIFF")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode("utf-8", errors="replace")
        return httpx.Response(200, json={"segments": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Config(whisper_url="http://w:9000/v1")
    transcribe(cfg, audio, language="Japanese", client=client)
    assert 'name="language"' in seen["body"]
    assert "\r\n\r\nja\r\n" in seen["body"]


def test_transcribe_asr_webservice_hits_fixed_asr_endpoint(tmp_path):
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"RIFF")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.read().decode("utf-8", errors="replace")
        return httpx.Response(200, json={
            "text": "hello world",
            "segments": [{"start": 0.5, "end": 2.0, "text": " hello world"}],
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    cfg = Config(whisper_url="http://w:9000", whisper_api="asr-webservice")
    segs = transcribe(cfg, audio, language="Japanese", client=client)
    assert segs == [TranscriptSegment(start=0.5, end=2.0, text="hello world")]
    assert seen["url"] == "http://w:9000/asr?output=json&language=ja"
    assert 'name="audio_file"' in seen["body"]
    assert 'name="model"' not in seen["body"]
