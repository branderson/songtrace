# songtrace

Turn a song and its lyrics into a playable UltraStar chart.

A CLI and minimal web UI that turns a song — title, artist, lyrics you supply,
and a YouTube URL or local video file — into a playable
[UltraStar](https://usdx.eu/format/)-format song folder. Also compatible with
[Melody Mania](https://melodymania.org/).

![Web UI](docs/images/webui.png)

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
editor such as [Yass](https://www.yass-along.com/) or
[USDX](https://usdx.eu/) and touch those spots up before playing.

## Configuration

Config is TOML. Resolution order:

1. `--config <path>` passed explicitly on the command line
2. the `SONGTRACE_CONFIG` environment variable
3. `~/.config/songtrace/config.toml`

Copy the example and fill in your Whisper server:

```sh
cp config.example.toml config.toml
$EDITOR config.toml   # set whisper_url at minimum
```

See `config.example.toml` for every available field (`whisper_url`,
`whisper_api_key`, `whisper_model`, `whisper_api`, `output_dir`, `work_dir`,
`creator`, `demucs_model`, `crepe_model`, `device`, `bind_host`, `port`,
`ytdlp_clients`, `ytdlp_cookies`, `ytdlp_impersonate`,
`ytdlp_pot_provider_url`).

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

Run the CLI inside the container instead of the web UI:

```sh
docker compose run --rm songtrace songtrace make \
  --title "Song Title" --artist "Artist Name" \
  --lyrics /data/lyrics.txt --youtube https://youtu.be/...
```

GPU passthrough requires the [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
installed on the host; `docker-compose.yml` requests one GPU via the
`nvidia` device driver.

## Sharing the GPU

`songtrace serve` is long-lived but only wants the GPU in bursts, so it hands
it back after every stage rather than holding it for the life of the process.
Three stages load a model — `separate` (Demucs), `align` (wav2vec2) and
`pitch` (torchcrepe) — and each one's weights are dropped as soon as that
stage finishes, followed by `torch.cuda.empty_cache()` to return the
allocator's freed blocks to the driver instead of keeping them for reuse. This
matters most between `separate` and `align`: `transcribe` sits between them
waiting on your Whisper server, which can be minutes of a GPU nothing is
using.

Stages that gave memory back say so in the job log:

```
[separate] released 3704 MiB of GPU memory
```

The cost is reloading each model once per job instead of once per process,
which is seconds against stages that run for minutes. What this does not
release is the CUDA context itself, a few hundred MB that belongs to the
process until it exits; freeing that would mean running each stage in a
subprocess. If you need every last MB back, stop the container between runs.

## Bare-metal

```sh
pip install -e .
songtrace make --help
```

```
songtrace make --title TITLE --artist ARTIST --lyrics LYRICS_FILE \
  (--youtube URL | --video PATH) \
  [--genre GENRE] [--year YEAR] [--language LANGUAGE] [--cover COVER_IMAGE] \
  [--bpm BPM] [--force-from STAGE] [--config CONFIG]
```

Run the web UI instead with `songtrace serve [--config CONFIG]`.

## Re-running a stage

Each stage caches its output under `work_dir`. If a later stage needs
another pass (e.g. you want to retry alignment with different lyrics, or
pitch tracking looks off), re-run with `--force-from <stage>` to redo that
stage and everything after it, reusing the cached output of everything
before it:

```sh
songtrace make --title "..." --artist "..." --lyrics lyrics.txt \
  --youtube https://youtu.be/... --force-from align
```

Valid stages, in pipeline order: `acquire`, `separate`, `transcribe`,
`align`, `pitch`, `notes`, `grid`, `package`.

In the web UI this is what selecting a queue entry does: its title, artist,
lyrics and the rest come back up in the form, so you can adjust them and hit
REBUILD SONG. Only the stages your edit actually invalidates are rerun —
fixing a line of lyrics realigns without re-downloading or re-separating. NEW
SONG INSTEAD clears the form to start a fresh song.

## YouTube download failures

`HTTP Error 403: Forbidden` from the acquire stage means YouTube served the
page but refused the media URL. In rough order of likelihood:

1. **yt-dlp is stale.** YouTube changes its defences every few weeks and the
   version baked into the image does not. Hotfix a running container, then
   rebuild so it sticks:

   ```sh
   docker compose exec songtrace pip install -U 'yt-dlp[default,curl-cffi,deno]'
   docker compose restart songtrace
   docker compose build --pull songtrace   # permanent
   ```

2. **Missing JS runtime.** yt-dlp needs `yt-dlp-ejs` plus a JavaScript engine
   to solve YouTube's player challenges; without them it silently falls back to
   formats that 403. The `[default,curl-cffi,deno]` extras in `pyproject.toml`
   cover this. Check with
   `docker compose exec songtrace yt-dlp -v --simulate <url>` — the debug
   header must show a `JS runtimes:` line that is not `none`.

3. **The IP is rate-limited or flagged.** Datacenter and VPN ranges get
   throttled hardest. Try from another network before changing config.

4. **A PO token is required.** If the log says so explicitly, run
   [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)
   as a sidecar, install its plugin into the image, and point
   `ytdlp_pot_provider_url` at it.

`ytdlp_clients` retries the download against each listed player client in
turn, so a client that YouTube has broken is skipped automatically.
`ytdlp_cookies` (a Netscape cookie jar, for age-gated videos) and
`ytdlp_impersonate` (a browser TLS fingerprint, e.g. `"chrome"`) are the other
two levers. Do not use a cookie jar from an account you care about — YouTube
bans for automated access.

Failed downloads leave the work dir intact, so re-running the job resumes
rather than starting over.

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
