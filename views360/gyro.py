"""Gyro stabilization for Insta360 360 video.

Reads the camera's 1 kHz IMU from the .insv trailer, finds the IMU-to-video clock offset by matching
gyro rotation against image motion, integrates the camera's orientation, and produces per-frame
corrections that keep a view on a smoothed path (removing shake/vibration but following real turns).
"""
import hashlib
import math
import os
import struct
import subprocess

import numpy as np

from .paths import cache as _cache
from .system import FFMPEG
GYRO_LSB = math.radians(2000.0 / 32768.0)          # ±2000 deg/s full scale


# ------------------------------------------------------------------ quaternions (w, x, y, z)

def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def qconj(q):
    return q * np.array([1, -1, -1, -1.0])


def qexp(v):
    a = np.linalg.norm(v, axis=-1, keepdims=True)
    s = np.where(a > 1e-12, np.sin(a / 2) / np.maximum(a, 1e-12), 0.5)
    return np.concatenate([np.cos(a / 2), v * s], -1)


def qlog(q):
    q = q * np.sign(q[..., :1] + 1e-15)               # shortest path
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    a = 2 * np.arctan2(n, q[..., :1])
    return np.where(n > 1e-12, v / np.maximum(n, 1e-12) * a, 2 * v)


def rotvec_to_matrix(v):
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


# ------------------------------------------------------------------ IMU extraction

def has_cached_imu(insv):
    return os.path.exists(_cache("imu_%s.npz" % os.path.basename(insv)))


def load_imu(insv, log=print):
    """-> (t seconds from first sample, w rad/s in the 360 frame: x right, y down, z = yaw-0 direction)."""
    cache = _cache("imu_%s.npz" % os.path.basename(insv))   # works later without the SD card
    if os.path.exists(cache):
        z = np.load(cache)
        return z["t"], z["w"]
    if not os.path.exists(insv):
        raise RuntimeError("Camera file not found (%s) — insert the SD card once so the gyro data can be read" % insv)
    log("Reading the camera's gyro data…")
    size = os.path.getsize(insv)
    with open(insv, "rb") as f:
        f.seek(size - 72)
        h = f.read(72)
        if h[-32:] != b"8db42d694ccc418790edff439fe026bf":
            raise RuntimeError("No Insta360 sensor data in this file")
        extra = struct.unpack("<I", h[32:36])[0]
        f.seek(size - extra)
        T = np.frombuffer(f.read(extra - 72), np.uint8)
    rows = []
    for ph in range(20):                           # samples: u64 timestamp (us) + 6 x u16 (accel xyz, gyro xyz)
        n = (len(T) - ph) // 20
        a = T[ph:ph + n * 20].reshape(n, 20)
        ts = a[:, :8].copy().view("<u8").ravel().astype(np.int64)
        d = np.diff(ts)
        g = (d >= 500) & (d <= 2000)
        if g.sum() < 1000:
            continue
        ok = np.zeros(n, bool)
        ok[:-1] |= g
        ok[1:] |= g
        vals = a[ok][:, 8:].copy().view("<u2").astype(np.int32) - 32768
        rows.append(np.c_[ts[ok], vals])
    if not rows:
        raise RuntimeError("No gyro samples found")
    R = np.concatenate(rows)
    R = R[np.argsort(R[:, 0])]
    R = R[np.unique(R[:, 0], return_index=True)[1]]
    R = R[np.abs(R[:, 0] - np.median(R[:, 0])) < 6 * 3600 * 1e6]
    t = (R[:, 0] - R[0, 0]) / 1e6
    g = R[:, 4:7].astype(float)
    g -= np.median(g, 0)                            # gyro bias
    g *= GYRO_LSB
    w = np.c_[-g[:, 0], -g[:, 2], -g[:, 1]]          # Insta360 X-series IMU axes -> 360 frame
    np.savez(cache, t=t, w=w)
    return t, w


# ------------------------------------------------------------------ clock offset

def _frames(video, yaw, start, dur, W=320, H=180):
    vf = "v360=input=equirect:output=flat:yaw=%g:pitch=0:h_fov=90:v_fov=%.4f:w=%d:h=%d" % (
        yaw, math.degrees(2 * math.atan(math.tan(math.radians(45)) * H / W)), W, H)
    p = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-ss", "%.3f" % start, "-t", "%.3f" % dur, "-i", video,
                        "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W).astype(np.float32)


def _shifts(fr):
    """Frame-to-frame translation (dx, dy) by phase correlation on the upper part (outside world)."""
    H, W = fr.shape[1:]
    win = np.outer(np.hanning(int(H * 0.5)), np.hanning(W))
    F = np.fft.rfft2(fr[:, :int(H * 0.5)] * win)
    out = []
    for i in range(len(fr) - 1):
        c = F[i + 1] * np.conj(F[i])
        c /= np.abs(c) + 1e-9
        r = np.fft.irfft2(c, s=win.shape)
        y, x = np.unravel_index(np.argmax(r), r.shape)
        y = y - r.shape[0] if y > r.shape[0] // 2 else y
        x = x - r.shape[1] if x > r.shape[1] // 2 else x
        out.append((x, y))
    return np.array(out, float)


def estimate_offset(frames_fn, cache_name, t_imu, w, duration, fps, log=print, segments=4, seg=8.0):
    """IMU time of the video's first frame (seconds on the IMU clock).
    frames_fn(yaw, start, dur) -> gray float32 frames (n, H, W) of a 90-degree flat view."""
    key = hashlib.sha1(("%s|%d" % (cache_name, len(t_imu))).encode()).hexdigest()[:16]
    cache = _cache("imuoff_%s.txt" % key)
    if os.path.exists(cache):
        return float(open(cache).read())
    log("Matching gyro to video motion…")
    cs = np.vstack([np.zeros(3), np.cumsum(w[:-1] * np.diff(t_imu)[:, None], 0)])
    obs, tms = [], []
    for s in range(segments):
        start = duration * (s + 0.5) / segments - seg / 2
        for yaw in (180, 0):
            fr = frames_fn(yaw, start, seg)
            sh = _shifts(fr)
            obs.append(sh)
            tms.append(start + (np.arange(len(sh)) + 0.5) / fps)

    def score(off):
        tot = 0.0
        for sh, tm in zip(obs, tms):
            a = np.searchsorted(t_imu, tm + off - 0.5 / fps)
            b = np.searchsorted(t_imu, tm + off + 0.5 / fps)
            d = cs[np.clip(b, 0, len(cs) - 1)] - cs[np.clip(a, 0, len(cs) - 1)]   # rotation over the frame
            X = np.c_[d, np.ones(len(d))]
            for col in (0, 1):
                y = sh[:, col]
                coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
                tot += 1 - (res[0] / ((y - y.mean()) ** 2).sum() if res.size and y.std() > 0 else 1)
        return tot

    span = t_imu[-1] - duration
    coarse = sorted(((score(o), o) for o in np.arange(-5, span + 5.01, 0.1)), reverse=True)
    o0 = coarse[0][1]
    fine = sorted(((score(o), o) for o in np.arange(o0 - 0.15, o0 + 0.1501, 0.002)), reverse=True)
    off = fine[0][1]
    log("Gyro clock offset %.3f s" % off)
    with open(cache, "w") as f:
        f.write("%.4f" % off)
    return off


# ------------------------------------------------------------------ corrections

def corrections(t_imu, w, offset, fps, nframes, smooth, max_dev_deg=2.5):
    """Per-frame rotation vectors (radians, 360 frame) of R_dev = q^-1 * q_smooth."""
    tf = offset + np.arange(nframes + 1) / fps
    cs = np.vstack([np.zeros(3), np.cumsum(w[:-1] * np.diff(t_imu)[:, None], 0)])
    idx = np.clip(np.searchsorted(t_imu, tf), 0, len(cs) - 1)
    steps = np.diff(cs[idx], axis=0)                  # body-frame rotation during each frame
    dq = qexp(steps)
    q = np.empty((nframes, 4))
    q[0] = (1, 0, 0, 0)
    for i in range(1, nframes):
        q[i] = qmul(q[i - 1], dq[i - 1])
        if i % 1000 == 0:
            q[i] /= np.linalg.norm(q[i])
    sig = max(1e-3, smooth) * fps
    K = int(math.ceil(3 * sig))
    acc = np.zeros((nframes, 3))
    wsum = np.zeros((nframes, 1))
    qi_inv = qconj(q)
    for k in range(-K, K + 1):
        wt = math.exp(-0.5 * (k / sig) ** 2)
        lo, hi = max(0, -k), min(nframes, nframes - k)
        r = qlog(qmul(qi_inv[lo:hi], q[lo + k:hi + k]))
        acc[lo:hi] += wt * r
        wsum[lo:hi] += wt
    dev = acc / wsum
    a = np.linalg.norm(dev, axis=1, keepdims=True)
    m = math.radians(max_dev_deg)
    dev = np.where(a > 1e-12, dev / np.maximum(a, 1e-12) * m * np.tanh(a / m), dev)   # soft limit
    return dev.astype(np.float32)


# ------------------------------------------------------------------ warping a rendered flat view

def ypr_matrix(yaw, pitch, roll):
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def margin_for(W, H, hfov, max_dev_deg=2.5):
    f = (W / 2) / math.tan(math.radians(hfov) / 2)
    a = math.radians(max_dev_deg)
    m = f * math.tan(a) + 0.5 * math.hypot(W, H) * math.sin(a)   # covers the largest shift plus corner swing from roll
    return int(math.ceil(m / 2.0)) * 2 + 4


def homography(view, W, H, m, dev):
    """Output-pixel -> margin-image-pixel homography (for cv2 WARP_INVERSE_MAP)."""
    f = (W / 2) / math.tan(math.radians(view["hfov"]) / 2)
    Kout = np.array([[f, 0, W / 2 - 0.5], [0, f, H / 2 - 0.5], [0, 0, 1]])
    Kin = np.array([[f, 0, W / 2 + m - 0.5], [0, f, H / 2 + m - 0.5], [0, 0, 1]])
    R0 = ypr_matrix(view["yaw"], view["pitch"], view["roll"])
    return Kin @ R0.T @ rotvec_to_matrix(dev) @ R0 @ np.linalg.inv(Kout)


def margin_fov(view, W, H, m):
    f = (W / 2) / math.tan(math.radians(view["hfov"]) / 2)
    return (math.degrees(2 * math.atan((W / 2 + m) / f)), math.degrees(2 * math.atan((H / 2 + m) / f)))
