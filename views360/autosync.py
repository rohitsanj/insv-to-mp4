"""Auto-sync: find the time offset between the video and the KML track by matching the
horizon tilt seen out the front of the aircraft against the Sentry's bank angle."""
import hashlib
import math
import os
import re
import subprocess

import numpy as np

from .paths import cache as _cache
W, H, HFOV, VFOV = 200, 150, 100.0, 75.0


def horizon_angles(cam, yaw, log=print):
    """Sample one frame per keyframe of the camera's preview looking forward; return (times, tilt_deg)
    with NaN where no horizon is visible."""
    key = hashlib.sha1(("%s|%s|%.1f" % (os.path.basename(cam.insv), os.path.basename(cam.lrv), yaw)).encode()).hexdigest()[:16]
    cache = _cache("sync_%s.npz" % key)
    if os.path.exists(cache):
        z = np.load(cache)
        return z["t"], z["a"]
    log("Auto-sync: sampling the forward view…")
    frames, times = cam.lrv_frames({"yaw": yaw, "pitch": 0, "roll": 0, "hfov": HFOV}, W, H, keyframes=True)
    n = min(len(times), len(frames))
    times, frames = times[:n], frames[:n]
    ang = np.array([_tilt(f, k=1.0) for f in frames])         # camera views have square pixels
    np.savez(cache, t=times, a=ang)
    return times, ang


def _tilt(frame, k=None):
    f = frame.astype(np.int16)
    sky = (f[..., 2] > f[..., 0] + 12) & (f[..., 2] > 110)      # bright, bluish
    top = int(H * 0.62)                                          # ignore the cowling at the bottom
    s = sky[:top, 10:W - 10:2]
    cols = np.arange(10, W - 10, 2)
    good = s[:8].mean(0) >= 0.6                                  # column starts in sky
    notsky = ~s
    has = notsky.any(0)
    y = np.where(has, notsky.argmax(0), -1)
    ok = good & has & (y >= 6) & (y <= top - 3)
    if ok.sum() < 25:
        return np.nan
    xs, ys = cols[ok].astype(float), y[ok].astype(float)
    rng = np.random.default_rng(0)
    i = rng.integers(0, len(xs), (400, 2))
    dx = xs[i[:, 1]] - xs[i[:, 0]]
    m = np.abs(dx) > 8
    if m.sum() < 50:
        return np.nan
    slope = np.median((ys[i[m, 1]] - ys[i[m, 0]]) / dx[m])       # Theil–Sen, robust to raindrops/glare
    if np.median(np.abs(ys - (np.median(ys - slope * xs) + slope * xs))) > 4:
        return np.nan
    if k is None:   # non-square pixels of an h_fov=HFOV, v_fov=VFOV render
        k = (math.tan(math.radians(VFOV / 2)) / (H / 2)) / (math.tan(math.radians(HFOV / 2)) / (W / 2))
    return math.degrees(math.atan(-slope * k))                   # + = right side of horizon raised = right bank


def find_offset(times, ang, track_t, bank, video_start, duration):
    """Offset (seconds, added to the camera clock) that best aligns horizon tilt with bank."""
    tt, bk = np.asarray(track_t, float), np.asarray(bank, float)
    ok = np.isfinite(ang)
    tv, a = times[ok], ang[ok]
    if len(a) < 60:
        return None
    lo = max(tt[0] - video_start - duration, -6 * 3600)
    hi = min(tt[-1] - video_start, 6 * 3600)

    def corr(off):
        b = np.interp(video_start + tv + off, tt, bk, left=np.nan, right=np.nan)
        m = np.isfinite(b)
        if m.sum() < max(60, 0.25 * len(a)) or np.std(b[m]) < 1e-6:
            return -1.0, 0
        return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())

    coarse = [(corr(o)[0], o) for o in np.arange(math.floor(lo), math.ceil(hi) + 1, 1.0)]
    if not coarse:
        return None
    coarse.sort(reverse=True)
    c0, o0 = coarse[0]
    fine = sorted(((corr(o)[0], o) for o in np.arange(o0 - 1.5, o0 + 1.501, 0.1)), reverse=True)
    c, off = fine[0]
    # second-best peak well away from the best, to judge ambiguity
    runner = max((cc for cc, oo in coarse if abs(oo - o0) > 20), default=-1.0)
    b = np.interp(video_start + tv + off, tt, bk)
    gain = float(np.polyfit(b, a, 1)[0])
    return {"offset": round(float(off), 1), "corr": round(c, 3), "runnerUp": round(runner, 3),
            "samples": corr(off)[1], "gain": round(gain, 2),
            "confident": bool(c > 0.6 and c - runner > 0.1 and gain > 0)}
