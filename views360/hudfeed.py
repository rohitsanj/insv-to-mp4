#!/usr/bin/env python3
"""Draw flight-instrument frames for one render chunk and write them (raw RGBA) to stdout.

Each render chunk runs its own copy of this in a separate process, so instrument drawing for parallel
chunks really runs in parallel instead of competing inside the server process.

Usage: python -m views360.hudfeed JOB.json   with {"track": path, "hud": {...}, "t0": utc_of_first_frame, "fps": f, "frames": n}
"""
import json
import sys

from . import hud


def main():
    job = json.load(open(sys.argv[1]))
    engine = hud.HudEngine(hud.Track(open(job["track"], "rb").read(), job["track"]), log=lambda m: None)
    items, size = engine.layout(job["hud"])
    out = sys.stdout.buffer
    try:
        for n in range(job["frames"]):
            img, _ = engine.atlas(job["hud"], job["t0"] + n / job["fps"], items, size)
            out.write(img.tobytes())
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
