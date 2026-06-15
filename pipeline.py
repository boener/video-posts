"""
Video Posts Pipeline — converts blog posts to short vertical videos using AI-generated video clips.
"""

import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import html as html_lib
import re
import subprocess
import tempfile

import aiohttp
import yaml
from faster_whisper import WhisperModel

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.expanduser("~/video-posts/config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in cfg.get("paths", {}):
        cfg["paths"][key] = os.path.expanduser(cfg["paths"][key])
    key_file = Path("~/OPENROUTER_API_KEY.txt").expanduser()
    if key_file.exists():
        cfg.setdefault("api", {})["openrouter_api_key"] = key_file.read_text().strip()
    return cfg


cfg = load_config()

# ── Logging ───────────────────────────────────────────────────────────────────

log_path = os.path.join(cfg["paths"]["logs"], "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path),
    ],
)
log = logging.getLogger(__name__)

# ── Whisper model (loaded once at startup) ────────────────────────────────────

log.info("Loading Whisper model '%s' on CPU...", cfg["models"]["whisper"])
whisper_model = WhisperModel(cfg["models"]["whisper"], device="cpu", compute_type="int8")
log.info("Whisper model loaded.")

# ── Database ──────────────────────────────────────────────────────────────────


def get_next_post(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM blog_posts WHERE status = 'pending' ORDER BY CAST(post_id AS INTEGER) ASC LIMIT 1"
    )
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def set_status(conn, post_id, status, error_message=None):
    now = datetime.now(timezone.utc).isoformat() if status in ("done", "failed") else None
    conn.execute(
        "UPDATE blog_posts SET status = ?, error_message = ?, processed_at = ? WHERE post_id = ?",
        (status, error_message, now, post_id),
    )
    conn.commit()


# ── Step 2: Script generation ─────────────────────────────────────────────────


def strip_html(text):
    """Remove HTML tags and decode entities, collapsing whitespace."""
    text = re.sub(r'<[^>]+>', ' ', text or '')
    text = html_lib.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


async def generate_script(post, cfg):
    post_id = post["post_id"]
    log.info("[%s] Generating script...", post_id)

    prompt_path = os.path.join(cfg["paths"]["prompts"], "script-writer.txt")
    with open(prompt_path) as f:
        system_prompt = f.read()

    user_message = (
        f"TITLE: {post['post_title']}\n"
        f"OPENING PARAGRAPH: {strip_html(post.get('opening_paragraph', ''))}\n"
        f"BODY CONTENT: {strip_html(post.get('post_content', ''))}"
    )

    headers = {
        "Authorization": f"Bearer {cfg['api']['openrouter_api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["models"]["script_writer"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    base_url = cfg["api"]["openrouter_base_url"]
    last_exc = None
    for attempt in range(1, 4):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions", headers=headers, json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                cost = resp.headers.get("x-openrouter-cost") or data.get("usage", {}).get("cost")
                if cost:
                    log.info("[%s] Script cost: $%s", post_id, cost)

        content = data["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r'^```[a-z]*\n?', '', content)
            content = re.sub(r'\n?```\s*$', '', content).strip()
        try:
            script = json.loads(content)
            break
        except json.JSONDecodeError as e:
            last_exc = e
            log.warning("[%s] Script JSON parse failed (attempt %d/3): %s", post_id, attempt, e)
    else:
        raise last_exc

    out_path = os.path.join(cfg["paths"]["scripts"], f"{post_id}-script.json")
    with open(out_path, "w") as f:
        json.dump(script, f, indent=2)
    log.info("[%s] Script saved (%d segments).", post_id, len(script["segments"]))
    return script


# ── Step 3: Audio generation ──────────────────────────────────────────────────


def _pcm_to_mp3(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Convert raw s16le PCM bytes (Gemini TTS output) to MP3 bytes via FFmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as fin:
        fin.write(pcm_bytes)
        in_path = fin.name
    out_path = in_path.replace(".pcm", ".mp3")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels),
             "-i", in_path, out_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg PCM->MP3 failed: {result.stderr[-400:]}")
        return Path(out_path).read_bytes()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


async def generate_audio(session, segment, post_id, cfg, sem, resume=False):
    index = segment["index"]
    out_dir = os.path.join(cfg["paths"]["audio"], str(post_id))
    out_path = os.path.join(out_dir, f"segment_{index}.mp3")

    if resume and os.path.exists(out_path):
        log.info("[%s] Audio seg %d exists, skipping.", post_id, index)
        return

    headers = {
        "Authorization": f"Bearer {cfg['api']['openrouter_api_key']}",
        "Content-Type": "application/json",
    }
    tts_text = re.sub(r"\[.*?\]", "", segment["text"]).strip()
    payload = {
        "model": cfg["models"]["tts"],
        "input": tts_text,
        "voice": cfg["voice"]["narrator_voice"],
        "response_format": "pcm",
    }
    base_url = cfg["api"]["openrouter_base_url"]

    attempts = cfg["pipeline"]["retry_attempts"]
    backoff = cfg["pipeline"]["retry_backoff_seconds"]
    for attempt in range(attempts):
        try:
            async with sem:
                async with session.post(
                    f"{base_url}/audio/speech", headers=headers, json=payload
                ) as resp:
                    resp.raise_for_status()
                    pcm_bytes = await resp.read()
                    mp3_bytes = _pcm_to_mp3(pcm_bytes)
                    cost = resp.headers.get("x-openrouter-cost")
                    if cost:
                        log.info("[%s] Audio seg %d cost: $%s", post_id, index, cost)
            with open(out_path, "wb") as f:
                f.write(mp3_bytes)
            return
        except Exception as e:
            if attempt < attempts - 1:
                log.warning("[%s] Audio seg %d attempt %d failed: %s", post_id, index, attempt + 1, e)
                await asyncio.sleep(backoff * (attempt + 1))
            else:
                raise


def _get_audio_duration(audio_path):
    """Return duration of an audio file in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


# ── Step 4: Video clip generation ────────────────────────────────────────────


async def submit_video_job(session, segment, duration_seconds, cfg):
    """Submit a text-to-video job to OpenRouter and return the job ID or direct video URL."""
    index = segment["index"]
    post_id = segment.get("_post_id", "?")

    motion_style = cfg["video_gen"]["motion_style"]
    prompt = f"{motion_style}, {segment['video_prompt']}"

    headers = {
        "Authorization": f"Bearer {cfg['api']['openrouter_api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["models"]["video"],
        "messages": [{"role": "user", "content": prompt}],
        "duration": int(duration_seconds),
    }

    base_url = cfg["api"]["openrouter_base_url"]
    attempts = cfg["pipeline"]["retry_attempts"]
    backoff = cfg["pipeline"]["retry_backoff_seconds"]

    for attempt in range(attempts):
        try:
            async with session.post(
                f"{base_url}/chat/completions", headers=headers, json=payload
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")
                data = await resp.json()
                cost = resp.headers.get("x-openrouter-cost")
                if cost:
                    log.info("[%s] Video seg %d job cost: $%s", post_id, index, cost)

            # Some models return a generation ID, others return the video URL directly.
            # Check for a direct video URL in the response first.
            message = data.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")

            # If content is a URL pointing to a video file, return it directly
            if isinstance(content, str) and (
                content.startswith("http") and (
                    ".mp4" in content or ".webm" in content or "video" in content.lower()
                )
            ):
                log.info("[%s] Video seg %d: got direct URL from response.", post_id, index)
                return content

            # Check for a generation/job ID in the response
            gen_id = (
                data.get("id")
                or data.get("generation_id")
                or (data.get("choices", [{}])[0].get("message", {}).get("generation_id"))
            )
            if gen_id:
                log.info("[%s] Video seg %d: job ID %s", post_id, index, gen_id)
                return gen_id

            # If neither, raise so we can debug
            raise RuntimeError(f"Unexpected video gen response: {json.dumps(data)[:500]}")

        except Exception as e:
            if attempt < attempts - 1:
                log.warning("[%s] Video seg %d job submit attempt %d failed: %s", post_id, index, attempt + 1, e)
                await asyncio.sleep(backoff * (attempt + 1))
            else:
                raise


async def poll_video_job(session, job_id_or_url, post_id, segment_index, cfg):
    """Poll until video clip is ready, then download and save it. Returns local file path."""
    out_dir = os.path.join(cfg["paths"]["video_clips"], str(post_id))
    out_path = os.path.join(out_dir, f"segment_{segment_index}.mp4")

    base_url = cfg["api"]["openrouter_base_url"]
    headers = {
        "Authorization": f"Bearer {cfg['api']['openrouter_api_key']}",
    }

    async def _download(video_url):
        log.info("[%s] Video seg %d: downloading from %s", post_id, segment_index, video_url[:80])
        async with session.get(video_url) as resp:
            resp.raise_for_status()
            data = await resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        log.info("[%s] Video seg %d saved (%d bytes): %s", post_id, segment_index, len(data), out_path)
        return out_path

    # If we were given a direct URL, skip polling
    if job_id_or_url.startswith("http"):
        return await _download(job_id_or_url)

    # Poll the generation endpoint
    poll_url = f"{base_url}/generation/{job_id_or_url}"
    deadline = time.time() + 15 * 60  # 15-minute timeout
    poll_interval = 10

    log.info("[%s] Video seg %d: polling job %s...", post_id, segment_index, job_id_or_url)
    while time.time() < deadline:
        await asyncio.sleep(poll_interval)
        async with session.get(poll_url, headers=headers) as resp:
            if resp.status == 404:
                # Some APIs use a different status endpoint
                log.warning("[%s] Video seg %d: poll 404, retrying in %ds...", post_id, segment_index, poll_interval)
                continue
            resp.raise_for_status()
            data = await resp.json()

        status = data.get("status", "")
        video_url = (
            data.get("video_url")
            or data.get("url")
            or (data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if data.get("choices") else "")
        )

        if status in ("succeeded", "completed", "done") or (
            isinstance(video_url, str) and video_url.startswith("http")
        ):
            if video_url and video_url.startswith("http"):
                return await _download(video_url)
            raise RuntimeError(f"Job succeeded but no video URL found: {json.dumps(data)[:300]}")

        if status in ("failed", "error", "cancelled"):
            raise RuntimeError(f"Video generation job failed: {json.dumps(data)[:300]}")

        log.info("[%s] Video seg %d: status=%s, waiting...", post_id, segment_index, status or "pending")

    raise RuntimeError(f"Video generation timed out after 15 minutes for seg {segment_index}")


# ── Step 3+4 combined: generate all assets ────────────────────────────────────


async def generate_all_assets(script, post_id, cfg, resume=False):
    log.info("[%s] Generating assets for %d segments (resume=%s)...", post_id, len(script["segments"]), resume)

    audio_dir = os.path.join(cfg["paths"]["audio"], str(post_id))
    video_clips_dir = os.path.join(cfg["paths"]["video_clips"], str(post_id))
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(video_clips_dir, exist_ok=True)

    sem = asyncio.Semaphore(cfg["pipeline"]["max_concurrent_requests"])
    segments = script["segments"]
    total = len(segments)

    # Phase A: Audio
    log.info("[%s] Phase A: generating TTS audio for %d segments...", post_id, total)
    connector = aiohttp.TCPConnector(limit=cfg["pipeline"]["max_concurrent_requests"])
    async with aiohttp.ClientSession(connector=connector) as session:
        completed = {"audio": 0}

        async def audio_task(seg):
            await generate_audio(session, seg, post_id, cfg, sem, resume=resume)
            completed["audio"] += 1
            log.info("[%s] Audio %d/%d done.", post_id, completed["audio"], total)

        await asyncio.gather(*[audio_task(seg) for seg in segments])

    log.info("[%s] Phase A complete.", post_id)

    # Measure actual audio durations
    durations = {}
    for seg in segments:
        idx = seg["index"]
        aud_path = os.path.join(audio_dir, f"segment_{idx}.mp3")
        d = _get_audio_duration(aud_path)
        # Enforce minimum duration for video generation APIs
        durations[idx] = max(d, 5.0)
        log.info("[%s] Seg %d audio duration: %.2fs (clamped to %.2fs)", post_id, idx, d, durations[idx])

    # Phase B: Video clips
    log.info("[%s] Phase B: generating video clips for %d segments...", post_id, total)
    connector = aiohttp.TCPConnector(limit=cfg["pipeline"]["max_concurrent_requests"])
    async with aiohttp.ClientSession(connector=connector) as session:
        completed = {"video": 0}

        async def video_task(seg):
            idx = seg["index"]
            clip_path = os.path.join(video_clips_dir, f"segment_{idx}.mp4")
            if resume and os.path.exists(clip_path):
                log.info("[%s] Video clip seg %d exists, skipping.", post_id, idx)
                completed["video"] += 1
                return

            # Tag the segment with post_id for logging in submit_video_job
            seg_tagged = dict(seg, _post_id=post_id)
            async with sem:
                job_id_or_url = await submit_video_job(session, seg_tagged, durations[idx], cfg)
            await poll_video_job(session, job_id_or_url, post_id, idx, cfg)
            completed["video"] += 1
            log.info("[%s] Video %d/%d done.", post_id, completed["video"], total)

        await asyncio.gather(*[video_task(seg) for seg in segments])

    log.info("[%s] All assets generated.", post_id)
    return durations


# ── Step 5: Whisper transcription ─────────────────────────────────────────────


def transcribe_segments(script, post_id, cfg):
    log.info("[%s] Transcribing audio segments with Whisper...", post_id)
    audio_dir = os.path.join(cfg["paths"]["audio"], str(post_id))
    all_words = []

    for seg in script["segments"]:
        index = seg["index"]
        audio_path = os.path.join(audio_dir, f"segment_{index}.mp3")
        tts_text = re.sub(r"\[.*?\]", "", seg["text"]).strip()
        segments_gen, _ = whisper_model.transcribe(audio_path, word_timestamps=True, initial_prompt=tts_text)
        words = []
        for whisper_seg in segments_gen:
            for word in (whisper_seg.words or []):
                words.append({"word": word.word.strip(), "start": word.start, "end": word.end})
        all_words.append(words)

    log.info("[%s] Transcription complete.", post_id)
    return all_words


# ── Step 6: Build ASS subtitle file ──────────────────────────────────────────


def _ass_time(seconds):
    """Convert seconds to ASS timestamp H:MM:SS.cc"""
    cs = int(round(seconds * 100))
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    c = cs % 100
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def build_ass(script, word_timestamps, post_id, cfg):
    log.info("[%s] Building ASS subtitle file...", post_id)
    sub_cfg = cfg["subtitles"]
    bold = 1 if sub_cfg["bold"] else 0
    margin_v = int(1920 * (1 - sub_cfg["vertical_position_percent"] / 100))

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Default,{sub_cfg['font']},{sub_cfg['font_size']},"
        f"&H00FFFFFF,&H00000000,&H00000000,"
        f"{bold},0,1,{sub_cfg['outline_width']},1,2,10,10,{margin_v}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = [header]
    timeline_offset = 0.0

    for seg_idx, (seg, words) in enumerate(zip(script["segments"], word_timestamps)):
        if not words:
            # Advance timeline by actual audio duration if no words transcribed
            audio_path = os.path.join(cfg["paths"]["audio"], str(post_id), f"segment_{seg['index']}.mp3")
            timeline_offset += _get_audio_duration(audio_path)
            continue

        seg_start = timeline_offset
        seg_end = timeline_offset + words[-1]["end"]

        for wi, word_info in enumerate(words):
            w_start = timeline_offset + word_info["start"]
            if wi + 1 < len(words):
                w_end = timeline_offset + words[wi + 1]["start"]
            else:
                w_end = seg_end

            text = word_info["word"].upper()
            lines.append(
                f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},Default,,0,0,0,,{text}\n"
            )

        # Advance by actual audio duration so offsets don't drift from Whisper undershooting
        audio_path = os.path.join(cfg["paths"]["audio"], str(post_id), f"segment_{seg['index']}.mp3")
        timeline_offset += _get_audio_duration(audio_path)

    out_path = os.path.join(cfg["paths"]["subtitles"], f"{post_id}.ass")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    log.info("[%s] ASS file saved: %s", post_id, out_path)
    return out_path


# ── Step 7: FFmpeg assembly ───────────────────────────────────────────────────


def _build_video_filter(segments, audio_dir, video_clips_dir, post_id, subtitles_path, ff_cfg):
    """Build an FFmpeg concat filter for video clips + audio, with subtitle burn-in."""
    escaped_subs = subtitles_path.replace("\\", "\\\\").replace(":", "\\:")
    n = len(segments)

    filter_lines = []
    video_labels = []
    audio_labels = []

    for i, seg in enumerate(segments):
        # Inputs are interleaved: [0]=clip0, [1]=audio0, [2]=clip1, [3]=audio1, ...
        vid_in = i * 2
        aud_in = i * 2 + 1

        # Scale to 1080x1920, preserving aspect ratio with padding
        filter_lines.append(
            f"[{vid_in}:v]"
            f"scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,setpts=PTS-STARTPTS"
            f"[vs{i}]"
        )
        filter_lines.append(
            f"[{aud_in}:a]atrim=start=0,asetpts=PTS-STARTPTS[as{i}]"
        )
        video_labels.append(f"[vs{i}]")
        audio_labels.append(f"[as{i}]")

    interleaved = []
    for v, a in zip(video_labels, audio_labels):
        interleaved.append(v)
        interleaved.append(a)

    filter_lines.append(
        "".join(interleaved) + f"concat=n={n}:v=1:a=1[vout][aout]"
    )
    filter_lines.append(f"[vout]subtitles={escaped_subs}[vfinal]")
    return ";\n".join(filter_lines)


def _run_cmd(cmd, label):
    """Run a command, raise on non-zero exit."""
    log.info("%s: %s", label, " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("%s stderr:\n%s", label, result.stderr[-3000:])
        raise RuntimeError(f"{label} failed with code {result.returncode}")
    return result


def _slug(title):
    s = title.lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')


def _db_prefix(cfg):
    db = cfg['paths']['db']
    return 'ko' if 'kombativ' in db else 'km'


def _assemble_video_remote(script, post_id, cfg, segments, durations,
                            video_clips_dir, audio_dir, subtitles_path, output_path, ff):
    r = cfg["rendering"]
    host = r["remote_host"]
    user = r["remote_user"]
    port = str(r["remote_port"])
    work_dir = r["remote_work_dir"].rstrip("/") + f"/{post_id}"
    remote = f"{user}@{host}"
    ssh_opts = ["-p", port, "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=60"]
    ssh_base = ["ssh"] + ssh_opts + [remote]

    log.info("[%s] Remote rendering: %s:%s", post_id, remote, work_dir)

    _run_cmd(ssh_base + ["mkdir", "-p", work_dir], "ssh mkdir")

    try:
        ssh_e = f"ssh {' '.join(ssh_opts)}"

        # rsync video clips
        clip_files = [
            os.path.join(video_clips_dir, f"segment_{seg['index']}.mp4")
            for seg in segments
        ]
        log.info("[%s] rsync video clips (%d files)...", post_id, len(clip_files))
        _run_cmd(
            ["rsync", "-a", "-e", ssh_e] + clip_files + [f"{remote}:{work_dir}/"],
            "rsync video-clips",
        )

        # rsync audio
        aud_files = [
            os.path.join(audio_dir, f"segment_{seg['index']}.mp3")
            for seg in segments
        ]
        log.info("[%s] rsync audio (%d files)...", post_id, len(aud_files))
        _run_cmd(
            ["rsync", "-a", "-e", ssh_e] + aud_files + [f"{remote}:{work_dir}/"],
            "rsync audio",
        )

        # rsync subtitles
        log.info("[%s] rsync subtitles: %s", post_id, subtitles_path)
        _run_cmd(
            ["rsync", "-a", "-e", ssh_e, subtitles_path, f"{remote}:{work_dir}/"],
            "rsync subtitles",
        )

        # Build filter with remote paths
        remote_subs = f"{work_dir}/{os.path.basename(subtitles_path)}"

        # Rewrite segment paths to remote equivalents for filter building
        remote_video_clips_dir = work_dir
        remote_audio_dir = work_dir
        filter_script = _build_video_filter(
            segments, remote_audio_dir, remote_video_clips_dir, post_id, remote_subs, ff
        )
        remote_fc = f"{work_dir}/fc.txt"

        local_fc = os.path.join(cfg["paths"]["videos"], f"{post_id}_fc_remote.txt")
        with open(local_fc, "w") as f:
            f.write(filter_script)
        try:
            _run_cmd(
                ["rsync", "-a", "-e", ssh_e, local_fc, f"{remote}:{remote_fc}"],
                "rsync fc.txt",
            )
        finally:
            try:
                os.unlink(local_fc)
            except OSError:
                pass

        # Build remote FFmpeg command — video clip + audio input pairs
        remote_output = f"{work_dir}/output.mp4"
        ffmpeg_cmd = ["ffmpeg", "-y"]
        for seg in segments:
            idx = seg["index"]
            clip = f"{work_dir}/segment_{idx}.mp4"
            aud = f"{work_dir}/segment_{idx}.mp3"
            # Use -t to trim each clip to its audio duration
            aud_dur = durations.get(idx, cfg["video_gen"]["duration_fallback_seconds"])
            ffmpeg_cmd += ["-t", f"{aud_dur:.6f}", "-i", clip, "-t", f"{aud_dur:.6f}", "-i", aud]

        ffmpeg_cmd += [
            "-filter_complex_script", remote_fc,
            "-map", "[vfinal]",
            "-map", "[aout]",
            "-c:v", ff["video_codec"],
            "-preset", ff.get("preset", "faster"),
            "-crf", str(ff.get("crf", 26)),
            "-pix_fmt", "yuv420p",
            "-c:a", ff["audio_codec"],
            remote_output,
        ]

        remote_done = f"{work_dir}/encode.done"
        remote_failed = f"{work_dir}/encode.failed"
        remote_log = f"{work_dir}/ffmpeg.log"
        ffmpeg_str = " ".join(f'"{a}"' if " " in a else a for a in ffmpeg_cmd)
        launch_script = (
            f"nohup sh -c '{ffmpeg_str} && touch {remote_done} || touch {remote_failed}'"
            f" > {remote_log} 2>&1 &"
        )
        log.info("[%s] Launching detached FFmpeg on remote host %s...", post_id, host)
        t0 = time.time()
        _run_cmd(ssh_base + [launch_script], "ssh launch ffmpeg")

        poll_interval = 15
        while True:
            time.sleep(poll_interval)
            result = subprocess.run(
                ssh_base + [
                    f"if [ -f {remote_done} ]; then echo done;"
                    f" elif [ -f {remote_failed} ]; then echo failed;"
                    f" else echo running; fi"
                ],
                capture_output=True, text=True,
            )
            status = result.stdout.strip()
            elapsed = time.time() - t0
            if status == "done":
                log.info("[%s] Remote FFmpeg finished in %.1fs", post_id, elapsed)
                break
            elif status == "failed":
                ffmpeg_log_result = subprocess.run(
                    ssh_base + ["tail", "-50", remote_log],
                    capture_output=True, text=True,
                )
                log.error("[%s] Remote FFmpeg failed. Log tail:\n%s",
                          post_id, ffmpeg_log_result.stdout)
                raise RuntimeError("Remote FFmpeg failed — see log above")
            else:
                log.info("[%s] Remote FFmpeg still running... %.0fs elapsed", post_id, elapsed)

        log.info("[%s] rsync output.mp4 back to %s...", post_id, output_path)
        _run_cmd(
            ["rsync", "-a", "-e", ssh_e, f"{remote}:{remote_output}", output_path],
            "rsync output",
        )

    finally:
        log.info("[%s] Cleaning up remote work dir %s...", post_id, work_dir)
        cleanup = subprocess.run(
            ssh_base + ["rm", "-rf", work_dir],
            capture_output=True,
        )
        if cleanup.returncode != 0:
            log.warning("[%s] Remote cleanup failed (non-fatal)", post_id)


def assemble_video(script, post_id, cfg, post_title, durations):
    log.info("[%s] Assembling video with FFmpeg...", post_id)

    ff = cfg["ffmpeg"]
    video_clips_dir = os.path.join(cfg["paths"]["video_clips"], str(post_id))
    audio_dir = os.path.join(cfg["paths"]["audio"], str(post_id))
    subtitles_path = os.path.join(cfg["paths"]["subtitles"], f"{post_id}.ass")
    output_path = os.path.join(cfg["paths"]["videos"], f"{_db_prefix(cfg)}{post_id}-{_slug(post_title)}.mp4")
    segments = script["segments"]
    n = len(segments)

    render_mode = cfg.get("rendering", {}).get("mode", "local")

    if render_mode == "remote":
        _assemble_video_remote(
            script, post_id, cfg, segments, durations,
            video_clips_dir, audio_dir, subtitles_path, output_path, ff,
        )
        log.info("[%s] Video assembled (remote): %s", post_id, output_path)
        return output_path

    # ── Local rendering ───────────────────────────────────────────────────────

    filter_script = _build_video_filter(
        segments, audio_dir, video_clips_dir, post_id, subtitles_path, ff
    )

    fc_file = os.path.join(cfg["paths"]["videos"], f"{post_id}_fc.txt")
    with open(fc_file, "w") as f:
        f.write(filter_script)

    cmd = ["ffmpeg", "-y"]
    for seg in segments:
        idx = seg["index"]
        clip = os.path.join(video_clips_dir, f"segment_{idx}.mp4")
        aud = os.path.join(audio_dir, f"segment_{idx}.mp3")
        aud_dur = durations.get(idx, cfg["video_gen"]["duration_fallback_seconds"])
        cmd += ["-t", f"{aud_dur:.6f}", "-i", clip, "-t", f"{aud_dur:.6f}", "-i", aud]

    cmd += [
        "-filter_complex_script", fc_file,
        "-map", "[vfinal]",
        "-map", "[aout]",
        "-c:v", ff["video_codec"],
        "-preset", ff.get("preset", "faster"),
        "-crf", str(ff.get("crf", 26)),
        "-pix_fmt", "yuv420p",
        "-c:a", ff["audio_codec"],
        "-threads", str(ff["threads"]),
        "-movflags", "+faststart",
        output_path,
    ]

    log.info("[%s] FFmpeg: %d segments, filter script: %s", post_id, n, fc_file)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    try:
        os.unlink(fc_file)
    except OSError:
        pass

    if result.returncode != 0:
        log.error("[%s] FFmpeg stderr:\n%s", post_id, result.stderr[-3000:])
        raise RuntimeError(f"FFmpeg failed with code {result.returncode}")

    log.info("[%s] Video assembled in %.1fs: %s", post_id, elapsed, output_path)
    return output_path


# ── Step 8: Social media copy ─────────────────────────────────────────────────


SOCIAL_COPY_PROMPT = """\
You write short social media copy for martial arts videos. Given a blog post title, \
opening paragraph, and URL, write posting copy for three platforms. Follow this exact format:

=== YouTube ===
[2-3 teaser sentences. Can be a bit longer, conversational, hook-driven.]
Read More: {url}
[6-8 relevant hashtags]

=== Instagram ===
[2-3 punchy teaser sentences. Energetic, visual, short.]
Read More: {url}
[10-12 relevant hashtags]

=== Facebook ===
[2-3 teaser sentences. Friendly and conversational, slightly longer than Instagram.]
Read More: {url}
[4-5 relevant hashtags]

Write only the copy — no extra commentary or section labels beyond the === headers shown above.\
"""


async def generate_social_copy(post, cfg, video_path):
    post_id = post["post_id"]
    raw_url = post.get("url") or ""
    domain = "https://kombativ.com/" if _db_prefix(cfg) == "ko" else "https://karatemart.com/"
    url = domain + raw_url if raw_url and not raw_url.startswith("http") else raw_url
    log.info("[%s] Generating social copy...", post_id)

    user_message = (
        f"TITLE: {post['post_title']}\n"
        f"OPENING PARAGRAPH: {strip_html(post.get('opening_paragraph', ''))}\n"
        f"URL: {url}"
    )

    headers = {
        "Authorization": f"Bearer {cfg['api']['openrouter_api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["models"]["script_writer"],
        "messages": [
            {"role": "system", "content": SOCIAL_COPY_PROMPT},
            {"role": "user", "content": user_message},
        ],
    }

    base_url = cfg["api"]["openrouter_base_url"]
    for attempt in range(1, 4):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/chat/completions", headers=headers, json=payload
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    cost = resp.headers.get("x-openrouter-cost") or data.get("usage", {}).get("cost")
                    if cost:
                        log.info("[%s] Social copy cost: $%s", post_id, cost)
            break
        except Exception as e:
            log.warning("[%s] Social copy attempt %d/3 failed: %s", post_id, attempt, e)
            if attempt == 3:
                raise

    copy_text = data["choices"][0]["message"]["content"].strip()

    txt_path = os.path.splitext(video_path)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(copy_text + "\n")
    log.info("[%s] Social copy saved: %s", post_id, txt_path)


# ── Main pipeline ─────────────────────────────────────────────────────────────


async def process_post(post, cfg, resume=False):
    post_id = post["post_id"]
    t_start = time.time()
    log.info("=" * 60)
    log.info("[%s] Processing: %s (resume=%s)", post_id, post["post_title"], resume)

    # Step 2 — script
    script_path = os.path.join(cfg["paths"]["scripts"], f"{post_id}-script.json")
    if resume and os.path.exists(script_path):
        log.info("[%s] Script exists, skipping generation.", post_id)
        with open(script_path) as f:
            script = json.load(f)
    else:
        t = time.time()
        script = await generate_script(post, cfg)
        log.info("[%s] Script generated in %.1fs", post_id, time.time() - t)

    # Steps 3+4 — audio + video clips
    t = time.time()
    durations = await generate_all_assets(script, post_id, cfg, resume=resume)
    log.info("[%s] Assets generated in %.1fs", post_id, time.time() - t)

    # Step 5 — transcribe
    t = time.time()
    word_timestamps = transcribe_segments(script, post_id, cfg)
    log.info("[%s] Transcription done in %.1fs", post_id, time.time() - t)

    # Step 6 — subtitles
    t = time.time()
    build_ass(script, word_timestamps, post_id, cfg)
    log.info("[%s] Subtitles built in %.1fs", post_id, time.time() - t)

    # Step 7 — assemble
    t = time.time()
    video_path = assemble_video(script, post_id, cfg, post["post_title"], durations)
    log.info("[%s] Video assembled in %.1fs", post_id, time.time() - t)

    # Step 8 — social copy
    t = time.time()
    await generate_social_copy(post, cfg, video_path)
    log.info("[%s] Social copy generated in %.1fs", post_id, time.time() - t)

    log.info("[%s] Total time: %.1fs", post_id, time.time() - t_start)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-id', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='Skip assets that already exist on disk')
    parser.add_argument('--force', action='store_true', help='Re-process a post regardless of its current status')
    args = parser.parse_args()

    conn = sqlite3.connect(cfg["paths"]["db"])
    try:
        if args.post_id is not None:
            cur = conn.cursor()
            if args.force:
                cur.execute("SELECT * FROM blog_posts WHERE post_id = ?", (args.post_id,))
            else:
                cur.execute(
                    "SELECT * FROM blog_posts WHERE post_id = ? AND status = 'pending'",
                    (args.post_id,)
                )
            row = cur.fetchone()
            if row is None:
                log.info("Post %s not found or not pending. Exiting.", args.post_id)
                return
            if args.force:
                conn.execute(
                    "UPDATE blog_posts SET status = 'pending', error_message = NULL, processed_at = NULL WHERE post_id = ?",
                    (args.post_id,)
                )
                conn.commit()
                log.info("[%s] Status reset to pending (--force).", args.post_id)
            cols = [d[0] for d in cur.description]
            post = dict(zip(cols, row))
        else:
            post = get_next_post(conn)
        if post is None:
            log.info("No pending posts found. Exiting.")
            return

        post_id = post["post_id"]
        set_status(conn, post_id, "processing")
        log.info("[%s] Status set to processing.", post_id)

        try:
            asyncio.run(process_post(post, cfg, resume=args.resume))
            set_status(conn, post_id, "done")
            log.info("[%s] Done.", post_id)
        except Exception:
            err = traceback.format_exc()
            log.error("[%s] Pipeline failed:\n%s", post_id, err)
            set_status(conn, post_id, "failed", error_message=err[-2000:])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
