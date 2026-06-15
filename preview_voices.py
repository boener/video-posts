#!/usr/bin/env python3
"""Generate a voice sample MP3 for each OpenAI TTS voice using the pipeline config."""

import asyncio
import os
import sys
import yaml
import aiohttp

VOICES = ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"]

SAMPLE_TEXT = (
    "Welcome back. Today we're breaking down one of the most misunderstood techniques "
    "in all of martial arts — and by the end of this, you'll never look at it the same way again."
)

OUTPUT_DIR = os.path.expanduser("~/voice samples")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


async def generate_sample(session: aiohttp.ClientSession, voice: str, cfg: dict, speed: float) -> None:
    suffix = f"_x{speed}" if speed != 1.0 else ""
    out_path = os.path.join(OUTPUT_DIR, f"{voice}{suffix}.mp3")
    if os.path.exists(out_path):
        print(f"  [{voice}] already exists, skipping.")
        return

    headers = {
        "Authorization": f"Bearer {cfg['api']['openrouter_api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["models"]["tts"],
        "input": SAMPLE_TEXT,
        "voice": voice,
        "instructions": cfg["voice"]["style_instructions"],
        "response_format": "mp3",
        "speed": speed,
    }

    print(f"  [{voice}] Generating...")
    async with session.post(
        f"{cfg['api']['openrouter_base_url']}/audio/speech",
        headers=headers,
        json=payload,
    ) as resp:
        if resp.status != 200:
            body = await resp.text()
            print(f"  [{voice}] ERROR {resp.status}: {body}", file=sys.stderr)
            return
        audio = await resp.read()

    with open(out_path, "wb") as f:
        f.write(audio)
    print(f"  [{voice}] Saved -> {out_path}")


async def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    args = sys.argv[1:]
    speed = 1.0
    if args and args[0].replace(".", "", 1).isdigit():
        speed = float(args.pop(0))
        if not 0.25 <= speed <= 4.0:
            print("Speed must be between 0.25 and 4.0")
            sys.exit(1)

    voices = args if args else VOICES
    unknown = [v for v in voices if v not in VOICES]
    if unknown:
        print(f"Unknown voice(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(VOICES)}")
        sys.exit(1)

    print(f"Generating {len(voices)} voice sample(s) at speed {speed} -> {OUTPUT_DIR}\n")
    async with aiohttp.ClientSession() as session:
        tasks = [generate_sample(session, v, cfg, speed) for v in voices]
        await asyncio.gather(*tasks)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
