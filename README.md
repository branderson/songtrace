# karaoke-maker

A CLI and minimal web UI that turns a song — title, artist, lyrics you supply,
and a YouTube URL or local video file — into a playable
[UltraStar](https://usdx.eu/format/)-format song folder for
[Melody Mania](https://melodymania.net/), a commercial UltraStar Deluxe fork.

The pipeline:

1. **acquire** — download audio/video via `yt-dlp` (or use a local video file)
2. **separate** — split vocals/instrumental with Demucs
3. **transcribe** — send the vocal stem to your own intranet Whisper server
   (OpenAI-compatible, or [onerahmet/openai-whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)),
   as a sanity check that the supplied lyrics match the audio and to locate
   vocal sections
4. **align** — forced-align the *supplied* reference lyrics against the vocal
   stem with wav2vec2 (torchaudio's MMS_FA) to get word timings
5. **pitch** — track pitch on the vocal stem with torchcrepe
6. **notes** — turn word timings + pitch into UltraStar notes
7. **grid** — quantise notes to a beat grid
8. **package** — write the final song folder (`.txt` + `.mp3` + video + cover)

The shipped `.mp3` is the full original mix (vocals included) — the
separated stems are used for analysis only and are not included in the
output.

**The output is a best-effort draft**, not a finished song. Every run ends by
printing the low-confidence regions it found (e.g.
`3 low-confidence notes near 0:42–0:47`) — open the result in an UltraStar
editor such as [Yass](http://www.yass-hq.de/) or
[USDX](https://usdx.eu/) and touch those spots up before playing.

## Configuration

Config is TOML. Resolution order:

1. `--config <path>` passed explicitly on the command line
2. the `KARAOKE_MAKER_CONFIG` environment variable
3. `~/.config/karaoke-maker/config.toml`

Copy the example and fill in your Whisper server:

```sh
cp config.example.toml config.toml
$EDITOR config.toml   # set whisper_url at minimum
```

See `config.example.toml` for every available field (`whisper_url`,
`whisper_api_key`, `whisper_model`, `whisper_api`, `output_dir`, `work_dir`,
`creator`, `demucs_model`, `crepe_model`, `device`, `bind_host`, `port`).

`whisper_api` selects the wire format: `"openai"` (default) speaks the
OpenAI-compatible `POST {whisper_url}/audio/transcriptions` API; `"asr-webservice"`
speaks [onerahmet/openai-whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice)'s
`POST {whisper_url}/asr` API instead (point `whisper_url` at that server's base
URL with no `/v1` suffix — `whisper_model`/`whisper_api_key` are unused for
this backend, since the model is fixed by how that server was started).

## Docker

```sh
cp config.example.toml config.toml
$EDITOR config.toml
docker compose up -d
```

The web UI is then at `http://localhost:8000`. Downloaded model weights
(Demucs, wav2vec2, ...) are cached under `./data` on the host so they survive
container recreation.

![Web UI](docs/images/webui.png)

Run the CLI inside the container instead of the web UI:

```sh
docker compose run --rm karaoke-maker karaoke-maker make \
  --title "Song Title" --artist "Artist Name" \
  --lyrics /data/lyrics.txt --youtube https://youtu.be/...
```

GPU passthrough requires the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
installed on the host; `docker-compose.yml` requests one GPU via the
`nvidia` device driver.

## Bare-metal

```sh
pip install -e .
karaoke-maker make --help
```

```
karaoke-maker make --title TITLE --artist ARTIST --lyrics LYRICS_FILE \
  (--youtube URL | --video PATH) \
  [--genre GENRE] [--year YEAR] [--language LANGUAGE] [--cover COVER_IMAGE] \
  [--bpm BPM] [--force-from STAGE] [--config CONFIG]
```

Run the web UI instead with `karaoke-maker serve [--config CONFIG]`.

## Re-running a stage

Each stage caches its output under `work_dir`. If a later stage needs
another pass (e.g. you want to retry alignment with different lyrics, or
pitch tracking looks off), re-run with `--force-from <stage>` to redo that
stage and everything after it, reusing the cached output of everything
before it:

```sh
karaoke-maker make --title "..." --artist "..." --lyrics lyrics.txt \
  --youtube https://youtu.be/... --force-from align
```

Valid stages, in pipeline order: `acquire`, `separate`, `transcribe`,
`align`, `pitch`, `notes`, `grid`, `package`.

## Regenerating the web UI screenshot

`scripts/screenshot_webui.py` seeds the real app with a fixed set of jobs
covering every status and screenshots it to `docs/images/webui.png`:

```sh
pip install -e '.[dev]'
playwright install chromium
python scripts/screenshot_webui.py
```

## License

[AGPLv3](LICENSE)
