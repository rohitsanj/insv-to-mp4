#!/usr/bin/env python3
"""Calibrated stitcher for Insta360 dual-fisheye .insv video -> equirectangular 360 (YouTube-ready).

Uses the per-lens calibration the camera stores in the .insv trailer (omnidirectional lens model,
lens centres, distortion), refines the lens-to-lens alignment from the video's own frames, and
blends the two lenses across their overlap instead of hard-cutting at 90 degrees.

Usage:
    python -m views360.stitch360 INPUT.insv OUTPUT.mp4 [--width 7680] [--bitrate 80] [--start S --duration D]
                         [--no-fit] [--band 8]
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

from .system import FFMPEG, FFPROBE, encoder_args, hwaccel_args

# Camera-frame axes expressed in world axes (columns), world = ffmpeg equirect convention
# (x right, y down, z = yaw 0). Stream 0 faces yaw 180, stream 1 faces yaw 0 (Insta360 X-series .insv).
R_NOM = {0: np.stack([[0, -1., 0], [-1., 0, 0], [0, 0, -1.]], 1),
         1: np.stack([[0, 1., 0], [-1., 0, 0], [0, 0, 1.]], 1)}
DEFAULT_FIT = {"rx": 1.15, "ry": 0.65, "rz": 0.40, "fscale": 1.078}   # measured on an Insta360 X4 in 8K mode


# ------------------------------------------------------------------ calibration

def read_calibration(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 8_000_000))
        tail = f.read()
    for m in re.finditer(rb"2(?:_-?[0-9.]+){39}", tail):
        tok = m.group(0).decode().split("_")
        if len(tok) != 40 or float(tok[1]) > 10:      # want the omnidirectional (xi, fx, fy, ...) record
            continue
        lenses = []
        for i in range(2):
            v = [float(x) for x in tok[1 + 19 * i: 1 + 19 * (i + 1)]]
            lenses.append({"xi": v[0], "fx": v[1], "fy": v[2], "cx": v[3], "cy": v[4], "k": v[11:14], "p": v[14:16],
                           "W": v[16], "H": v[17]})
        # lens 2's centre is given in the full side-by-side sensor frame
        lenses[1]["cx"] -= lenses[1]["W"] / 2
        model = re.search(rb"Insta360 [A-Za-z0-9 ]+", tail)
        return lenses, (model.group(0).decode().strip() if model else "Insta360")
    raise RuntimeError("No lens calibration found in %s (is this an Insta360 .insv file?)" % path)


def probe(path):
    out = json.loads(subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                                     "stream=index,codec_type,width,height,r_frame_rate:format=duration:format_tags=creation_time",
                                     "-of", "json", path], capture_output=True, text=True, check=True).stdout)
    vids = [s for s in out["streams"] if s["codec_type"] == "video"]
    if len(vids) < 2:
        raise RuntimeError("Expected two fisheye video streams (one per lens) in %s" % path)
    return {"N": vids[0]["width"], "duration": float(out["format"]["duration"]),
            "creation_time": out["format"].get("tags", {}).get("creation_time")}


# ------------------------------------------------------------------ projection

def rot(ax, ay, az):
    ax, ay, az = map(math.radians, (ax, ay, az))
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Lenses:
    def __init__(self, cal, N, fit):
        self.cal, self.N, self.fit = cal, N, fit

    def project(self, k, d, fit=None):
        """World directions (...,3) -> source pixel (u, v) in lens k's frame, and angle from its axis (deg)."""
        fit = fit or self.fit
        corr = (fit["rx"], fit["ry"], fit["rz"]) if k == 0 else (0, 0, 0)
        c = d @ (rot(*corr) @ R_NOM[k])
        x, y, z = c[..., 0], c[..., 1], c[..., 2]
        C = self.cal[k]
        den = z + C["xi"]
        mx, my = x / den, y / den
        r2 = mx * mx + my * my
        k1, k2, k3 = C["k"]
        p1, p2 = C["p"]
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = mx * rad + 2 * p1 * mx * my + p2 * (r2 + 2 * mx * mx)
        dy = my * rad + p1 * (r2 + 2 * my * my) + 2 * p2 * mx * my
        side = C["H"]                                   # the video is a square crop of the sensor region
        crop = (C["W"] / 2 - side) / 2
        scale = self.N / side
        u = (C["fx"] * fit["fscale"] * dx + C["cx"] - crop) * scale
        v = (C["fy"] * fit["fscale"] * dy + C["cy"]) * scale
        return u, v, np.degrees(np.arccos(np.clip(z, -1, 1)))


def bilinear(img, u, v):
    N = img.shape[0]
    u = np.clip(u, 0, N - 1.001)
    v = np.clip(v, 0, N - 1.001)
    u0, v0 = u.astype(int), v.astype(int)
    fu, fv = u - u0, v - v0
    return img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv) + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv


def equirect_dirs(W, H, rows):
    lon = ((np.arange(W) + 0.5) / W * 2 - 1) * np.pi
    lat = ((rows + 0.5) / H * 2 - 1) * np.pi / 2
    LON, LAT = np.meshgrid(lon, lat)
    return np.stack([np.sin(LON) * np.cos(LAT), np.sin(LAT), np.cos(LON) * np.cos(LAT)], -1)


# ------------------------------------------------------------------ alignment fit

def grab(path, t, k, N):
    raw = subprocess.run([FFMPEG, "-v", "error", "-ss", "%.2f" % t, "-i", path, "-map", "0:v:%d" % k, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(N, N).astype(np.float32)


def fit_alignment(path, lenses, duration, log=print, samples=12):
    """Refine lens-1-vs-lens-0 tilt and the projection scale by matching the two lenses in their overlap.
    Only content that changes during the video is used, so things fixed to the camera (mounts, the
    vehicle's interior, a selfie stick) don't bias the fit."""
    N = lenses.N
    T = [duration * (i + 0.5) / samples for i in range(samples)]
    log("Sampling %d frame pairs for alignment…" % samples)
    imgs = {(t, k): grab(path, t, k, N) for t in T for k in (0, 1)}
    step = 0.3
    a = np.radians(np.arange(0, 360, step))
    b = np.radians(np.arange(-15, 15.01, step))
    A, B = np.meshgrid(a, b)
    D = np.stack([np.cos(A) * np.cos(B), np.sin(A) * np.cos(B), np.sin(B)], -1)  # band around the seam

    def bands(k, fit):
        u, v, th = lenses.project(k, D, fit)
        return np.stack([bilinear(imgs[(t, k)], u, v) for t in T]), th

    def blur(x, r):
        if r < 1:
            return x
        for ax in (1, 2):
            pad = [(0, 0)] * 3
            pad[ax] = (r, r)
            c = np.cumsum(np.pad(x, pad, mode="edge"), axis=ax)
            n = c.shape[ax]
            x = (np.take(c, range(2 * r, n), axis=ax) - np.take(c, range(0, n - 2 * r), axis=ax)) / (2 * r)
        return x

    base = dict(DEFAULT_FIT)
    S0, _ = bands(0, base)
    S1, _ = bands(1, base)
    moving = (S0.std(0) > 18) & (S1.std(0) > 18)
    if moving.mean() < 0.1:          # static camera: use everything
        moving = np.ones_like(moving)
    cache = {}

    def score(f, r):
        key = (round(f["fscale"], 4), r)
        if key not in cache:
            s1, t1 = bands(1, f)
            s1 = blur(s1, r)
            gy, gx = np.gradient(s1, axis=(1, 2))
            cache[key] = (gx, gy, t1)
        bx, by, t1 = cache[key]
        s0, t0 = bands(0, f)
        s0 = blur(s0, r)
        ay, ax = np.gradient(s0, axis=(1, 2))
        ok = (moving & (t0 < 104) & (t1 < 104))[None]
        num = ((ax * bx + ay * by) * ok).sum()
        return float(num / np.sqrt(((ax * ax + ay * ay) * ok).sum() * ((bx * bx + by * by) * ok).sum() + 1e-9))

    def refine(start):
        best = dict(start)
        for r, steps in ((5, (0.5, 0.25)), (2, (0.25, 0.1)), (0, (0.1, 0.05))):
            bs = score(best, r)
            for st in steps:
                improved = True
                while improved:
                    improved = False
                    for key, mult in (("rx", 1), ("ry", 1), ("rz", 1), ("fscale", 0.02)):
                        for sgn in (-1, 1):
                            f = dict(best)
                            f[key] = round(f[key] + sgn * st * mult, 4)
                            s = score(f, r)
                            if s > bs + 1e-5:
                                best, bs, improved = f, s, True
        return best, score(best, 0)

    log("Searching lens alignment…")
    grid = []
    for fs in np.arange(1.00, 1.141, 0.02):
        for rx in range(-4, 5):
            for ry in range(-4, 5):
                f = {"rx": float(rx), "ry": float(ry), "rz": 0.0, "fscale": float(fs)}
                grid.append((score(f, 5), f))
    grid.sort(key=lambda g: -g[0])
    # the blurred search only narrows things down: refine several starts (always including the known X4
    # alignment) and judge them at full sharpness, so a broad blurred peak can't win over the true one
    starts = [dict(base)] + [f for _, f in grid[:4]]
    results = [refine(s) for s in starts]
    best, bs = max(results, key=lambda r: r[1])
    log("Alignment: tilt %.2f/%.2f/%.2f deg, scale %.4f (match %.3f)" % (best["rx"], best["ry"], best["rz"], best["fscale"], bs))
    return best


# ------------------------------------------------------------------ maps + render

def write_maps(lenses, W, H, up, band, outdir, log=print):
    """Remap tables into the `up`x-upscaled fisheye frames, plus a blend mask (weight of lens 1)."""
    paths = {n: os.path.join(outdir, n) for n in ("x0.pgm", "y0.pgm", "x1.pgm", "y1.pgm", "mask.pgm")}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    files = {n: open(p + ".part", "wb") for n, p in paths.items()}
    for n in ("x0.pgm", "y0.pgm", "x1.pgm", "y1.pgm"):
        files[n].write(b"P5\n%d %d\n65535\n" % (W, H))
    files["mask.pgm"].write(b"P5\n%d %d\n255\n" % (W, H))
    limit = 104.0
    src_max = lenses.N * up - 1
    for r0 in range(0, H, 256):
        rows = np.arange(r0, min(H, r0 + 256))
        D = equirect_dirs(W, H, rows)
        th = {}
        for k in (0, 1):
            u, v, t = lenses.project(k, D)
            th[k] = t
            X = np.floor(u * up + up / 2.0 - 0.5 + 0.5)    # nearest pixel in the upscaled frame
            Y = np.floor(v * up + up / 2.0 - 0.5 + 0.5)
            bad = (t > limit) | (X < 0) | (Y < 0) | (X > src_max) | (Y > src_max)
            X[bad] = 65535
            Y[bad] = 65535
            files["x%d.pgm" % k].write(X.astype(">u2").tobytes())
            files["y%d.pgm" % k].write(Y.astype(">u2").tobytes())
        w1 = np.clip(((th[0] - th[1]) / 2) / (2 * band) + 0.5, 0, 1)
        w1 = np.where(th[1] > limit, 0.0, np.where(th[0] > limit, 1.0, w1))
        m = np.round(w1 * 255).astype(np.uint8)
        files["mask.pgm"].write(m.tobytes())
        log("Building stitch maps… %d%%" % (100 * min(H, r0 + 256) // H))
    for n, f in files.items():
        f.close()
        os.replace(paths[n] + ".part", paths[n])
    return paths


def render(src, out, maps, W, N, up, bitrate, start, duration, log=print, flags="lanczos"):
    M = N * up
    sc = "scale={M}:{M}:flags={f}," .format(M=M, f=flags) if up > 1 else ""
    g = ("[0:v:0]{sc}format=yuv444p[s0];[s0][1:v][2:v]remap=fill=black[e0];"
         "[0:v:1]{sc}format=yuv444p[s1];[s1][3:v][4:v]remap=fill=black[e1];"
         "[5:v]format=gray,split=3[ma][mb][mc];"
         "[ma][mb][mc]mergeplanes=format=yuv444p:map0s=0:map0p=0:map1s=1:map1p=0:map2s=2:map2p=0[m];"
         "[e0][e1][m]maskedmerge,format=yuv420p[v]").format(sc=sc)
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-nostdin", "-y", "-progress", "pipe:1"] + hwaccel_args()
    if start:
        cmd += ["-ss", "%.3f" % start]
    if duration:
        cmd += ["-t", "%.3f" % duration]
    cmd += ["-i", src]
    for n in ("x0.pgm", "y0.pgm", "x1.pgm", "y1.pgm", "mask.pgm"):
        cmd += ["-i", maps[n]]
    cmd += ["-filter_complex", g, "-map", "[v]", "-map", "0:a?", ] + encoder_args("hevc", bitrate) + [
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t0 = time.time()
    for line in p.stdout:
        k, _, v = line.strip().partition("=")
        if k == "out_time_us" and v.isdigit():
            done = int(v) / 1e6
            el = time.time() - t0
            log("Rendering %s / %s (%.2fx)" % (_hms(done), _hms(duration), done / el if el else 0))
    if p.wait() != 0:
        raise RuntimeError(p.stderr.read()[-2000:])


def _hms(s):
    s = int(s or 0)
    return "%d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--width", type=int, default=7680, help="equirectangular width (height = width/2)")
    ap.add_argument("--bitrate", type=int, default=80, help="Mbps")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--band", type=float, default=8.0, help="half-width of the blend band, degrees")
    ap.add_argument("--upscale", type=int, default=2, help="fisheye upscale before remap (quality vs speed)")
    ap.add_argument("--scale-flags", default="lanczos", help="upscale filter (lanczos, bicubic, bilinear)")
    ap.add_argument("--maps-dir", help="keep/reuse stitch maps in this folder")
    ap.add_argument("--no-fit", action="store_true", help="skip alignment fit and use the default alignment")
    ap.add_argument("--fit-json", help="reuse an alignment saved by an earlier run")
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)

    info = probe(a.input)
    cal, model = read_calibration(a.input)
    log("%s, lens images %dx%d, %s long" % (model, info["N"], info["N"], _hms(info["duration"])))
    if a.fit_json and os.path.exists(a.fit_json):
        fit = json.load(open(a.fit_json))
    elif a.no_fit:
        fit = dict(DEFAULT_FIT)
    else:
        fit = fit_alignment(a.input, Lenses(cal, info["N"], DEFAULT_FIT), info["duration"], log)
    json.dump(fit, open(os.path.splitext(a.output)[0] + ".stitch.json", "w"), indent=2)
    lenses = Lenses(cal, info["N"], fit)
    W, H = a.width, a.width // 2
    duration = a.duration or (info["duration"] - a.start)
    def go(d):
        maps = write_maps(lenses, W, H, a.upscale, a.band, d, log)
        render(a.input, a.output, maps, W, info["N"], a.upscale, a.bitrate, a.start, duration, log, a.scale_flags)
    if a.maps_dir:
        os.makedirs(a.maps_dir, exist_ok=True)
        go(a.maps_dir)
    else:
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.output))) as tmp:
            go(tmp)
    from .spherical import tag_360
    tag_360(a.output)
    log("Done: " + a.output)


if __name__ == "__main__":
    main()
