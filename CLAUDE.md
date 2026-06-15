# video-posts

Converts martial arts blog posts into short vertical videos for TikTok and Instagram Reels. Each post becomes 2–3 AI-generated video clips in a question-and-answer format: Clip 1 asks a sharp hook question, Clips 2–3 answer it. Total runtime under 60 seconds.

## How it differs from vertical-posts

`vertical-posts` (at `~/vertical-posts/`) generates static AI images with a Ken Burns zoom-pan effect. This project replaces that with AI-generated video clips (via Kling on OpenRouter). No Ken Burns, no image generation. The FFmpeg assembly step is simpler: just concat video clips with audio + subtitle burn-in.

## Where things live

- Code: `~/video-posts/`
- Databases, media, logs: `/mnt/storage/video-posts/`
- Prompt: `~/video-posts/prompts/script-writer.txt`
- Venv: `~/video-posts/venv/`

Storage layout:
```
/mnt/storage/video-posts/
  data/          — SQLite databases
  audio/         — TTS MP3s, one subdir per post_id
  video-clips/   — AI video clips, one subdir per post_id
  scripts/       — generated JSON scripts
  subtitles/     — ASS subtitle files
  videos/        — final assembled MP4s + social copy .txt
  logs/          — pipeline.log
```

## Running the service

```bash
sudo systemctl status video-posts    # check status
sudo systemctl restart video-posts   # restart
sudo systemctl stop video-posts      # stop
```

Dashboard at **http://localhost:5001**. The service auto-starts on boot.

Manual CLI run (useful for debugging a specific post):
```bash
cd ~/video-posts
source venv/bin/activate
python pipeline.py --post-id 42 --resume
python pipeline.py --post-id 42 --force   # ignore current status, re-run from scratch
```

## Databases

Two separate SQLite databases (copies of vertical-posts DBs, reset to all-pending at fork time):

| Key in config | Path | Prefix |
|---|---|---|
| `paths.db` / `paths.db_karate` | `.../blog_posts.db` | `km` |
| `paths.db_kombativ` | `.../blog_posts_kombativ.db` | `ko` |

Switch active DB via the dashboard's DB selector, or edit `paths.db` in `config.yaml`. The dashboard always reads `paths.db`.

**Kombativ `post_id` is TEXT** in the database. SQL ordering uses `CAST(post_id AS INTEGER)`.

## Pipeline steps

1. **Fetch** — get next pending post from DB, set status=processing
2. **Generate script** — Claude via OpenRouter, returns JSON with `title`, `episode_summary`, `segments[]` (each has `index`, `text`, `video_prompt`)
3. **Generate audio (TTS)** — Gemini TTS via OpenRouter, returns raw s16le PCM, converted to MP3 via FFmpeg (`_pcm_to_mp3`)
4. **Measure audio durations** — `ffprobe` on each MP3, clamped to minimum 5s for video API compatibility
5. **Generate video clips** — Kling via OpenRouter, async job submission + polling; saves `segment_N.mp4` per segment
6. **Transcribe** — Whisper (local, `tiny` model, CPU/int8), word-level timestamps
7. **Build ASS subtitles** — one word per cue, timeline built from actual audio durations (not Whisper timestamps, which can undershoot)
8. **Assemble MP4** — FFmpeg concat filter: scale each clip to 1080×1920, overlay audio, burn subtitles
9. **Generate social copy** — Claude writes YouTube/Instagram/Facebook copy, saved as `.txt` next to the video
10. **Mark done/failed** — update DB status

## API and models

All API calls go through OpenRouter (`openrouter_base_url` in config). Key is read at startup from `~/OPENROUTER_API_KEY.txt` — never stored in `config.yaml`.

| Model key | Default | Used for |
|---|---|---|
| `models.script_writer` | `anthropic/claude-sonnet-4-6` | Script + social copy |
| `models.tts` | `google/gemini-3.1-flash-tts-preview` | Text-to-speech |
| `models.video` | `kling/kling-video-1.0-standard-text2video` | Video clip generation |
| `models.whisper` | `tiny` | Local transcription |

Whisper runs locally — no API call.

## Video generation quirks

Kling via OpenRouter uses the `chat/completions` endpoint (not a dedicated video endpoint). The response may:
- Return a direct video URL in `choices[0].message.content` — used immediately
- Return a generation job ID — polled at `GET /generation/{id}` every 10 seconds, up to 15 minutes

The `duration` field is included in the request payload (integer seconds). Most video APIs enforce a minimum clip duration of 5 seconds, so audio durations shorter than 5s are clamped to 5s before being passed to the video API.

If Kling model ID doesn't resolve, alternatives to try: `kling/kling-video-1.0-pro-text2video`, `minimax/video-01`. The model is config-switchable — just change `models.video` in `config.yaml`.

Cost is logged from the `x-openrouter-cost` response header when present.

## FFmpeg assembly

Simpler than vertical-posts (no Ken Burns zoompan). For each segment, the video clip is scaled to 1080×1920 with letterbox/pillarbox padding (`scale=1080:1920:force_original_aspect_ratio=decrease,pad=...`). Audio is trimmed to the measured duration. All segments are concatenated, then subtitles are burned in at the end.

The filter complex is always written to a temp file (`{post_id}_fc.txt` / `{post_id}_fc_remote.txt`) — even for 2–3 segments — to avoid ARG_MAX issues and for consistency with vertical-posts.

Remote rendering (default): files are rsynced to `192.168.86.250:22022` (`/tmp/video-posts-render/{post_id}/`), FFmpeg runs detached, result rsynced back. SSH key auth is pre-configured.

## Config fields worth knowing

```yaml
models.video          # swap to a different video generation model
video_gen.motion_style       # prepended to every video prompt
video_gen.duration_fallback_seconds  # used if audio duration can't be measured
rendering.mode        # "remote" (default) or "local"
paths.db              # controls which DB the dashboard reads
```

## Known gotchas

- **PCM→MP3**: Gemini TTS returns raw s16le PCM at 24kHz mono, not MP3. The `_pcm_to_mp3()` function handles this via `ffmpeg -f s16le`. Do not change the `response_format: pcm` in the TTS call.
- **FFmpeg filter to file**: Always write the filter complex to a file. Do not pass it via `-filter_complex` on the command line.
- **ASS timeline = audio durations**: The subtitle timeline advances by `ffprobe`-measured audio duration per segment, not Whisper's reported duration. Whisper often undershoots by 0.1–0.5s, which causes subtitle drift on long posts.
- **Kombativ post_id is TEXT**: Never `ORDER BY post_id ASC` — always `ORDER BY CAST(post_id AS INTEGER) ASC`.
- **Dashboard always passes --resume**: The run button in the dashboard always passes `--resume` to the pipeline, so interrupted posts pick up where they left off. Use `--force` from the CLI if you need a true clean restart.
- **Eventlet deprecation warning**: Harmless. Eventlet prints a deprecation warning on import; it doesn't affect functionality.
