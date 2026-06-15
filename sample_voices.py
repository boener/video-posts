#!/usr/bin/env python3
"""
Generate a short TTS sample for every available Gemini voice and save to ~/voice-samples/.
Uses only stdlib + curl/ffmpeg (no aiohttp needed).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.expanduser("~/vertical-posts/config.yaml")

try:
    import yaml
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    API_KEY = cfg["api"]["openrouter_api_key"]
    MODEL = cfg["models"]["tts"]
    BASE_URL = cfg["api"]["openrouter_base_url"]
except Exception as e:
    print(f"Could not load config: {e}")
    sys.exit(1)

OUT_DIR = Path("~/voice-samples").expanduser()
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_TEXT = (
    "The real secret to power isn't just technique — it's understanding. "
    "When you know why something works, you can adapt it to anything."
)

# All documented voices for Gemini TTS (gemini-2.5-flash and gemini-3.x preview share this set)
VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Scheherazade", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib",
    "Rasalghul", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux",
    "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia",
    "Sadaltager", "Sulafat",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def pcm_to_mp3(pcm_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as fin:
        fin.write(pcm_bytes)
        in_path = fin.name
    out_path = in_path.replace(".pcm", ".mp3")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
             "-i", in_path, out_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-200:]}")
        return Path(out_path).read_bytes()
    finally:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def fetch_voice(voice: str) -> tuple[bool, str, bytes]:
    """Returns (success, message, mp3_bytes)."""
    payload = json.dumps({
        "model": MODEL,
        "input": SAMPLE_TEXT,
        "voice": voice,
        "response_format": "pcm",
    })

    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                "curl", "-s", "-w", "%{http_code}",
                "-X", "POST", f"{BASE_URL}/audio/speech",
                "-H", f"Authorization: Bearer {API_KEY}",
                "-H", "Content-Type: application/json",
                "-d", payload,
                "-o", tmp_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        status = result.stdout.strip()
        pcm_data = Path(tmp_path).read_bytes()

        if status != "200":
            error = pcm_data.decode(errors="replace")[:200]
            return False, f"HTTP {status}: {error}", b""

        mp3 = pcm_to_mp3(pcm_data)
        return True, f"{len(mp3):,} bytes", mp3

    except Exception as e:
        return False, str(e), b""
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Model : {MODEL}")
    print(f"Output: {OUT_DIR}")
    print(f"Voices: {len(VOICES)}")
    print("-" * 50)

    ok, skipped = 0, 0
    for voice in VOICES:
        success, msg, mp3 = fetch_voice(voice)
        if success:
            out_path = OUT_DIR / f"{voice}.mp3"
            out_path.write_bytes(mp3)
            print(f"  ✓  {voice:<20} {msg}")
            ok += 1
        else:
            print(f"  ✗  {voice:<20} {msg}")
            skipped += 1

    print("-" * 50)
    print(f"Done: {ok} saved, {skipped} skipped → {OUT_DIR}")


if __name__ == "__main__":
    main()
