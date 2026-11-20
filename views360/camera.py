"""Direct-from-camera access to an Insta360 recording: lens model, per-view remap tables, and the
camera's own low-res preview (.lrv). Views are stitched on the fly from the two fisheye images, so
there is no full 360 intermediate file.
"""
import glob
import hashlib
import json
import math
import os
import re
import subprocess

import numpy as np

from . import stitch360
from .paths import MAP_DIR
from .system import FFMPEG, FFPROBE
LIMIT_DEG = 104.0      # usable radius of each fisheye
BAND_DEG = 8.0         # half-width of the blend between lenses


def ypr_matrix(yaw, pitch, roll):
    """Same convention as ffmpeg v360 / the browser: direction = R @ view_ray."""
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def find_lrv(insv):
    """VID_<date>_<time>_00_<seq>.insv -> LRV_<date>_<time>_01_<seq>.lrv in the same folder."""
    d, b = os.path.split(insv)
    m = re.match(r"VID_(\d+_\d+)_\d+_(\d+)\.insv$", b, re.I)
    if not m:
        return None
    hits = glob.glob(os.path.join(d, "LRV_%s_*_%s.lrv" % (m.group(1), m.group(2))))
    return hits[0] if hits else None


def _rebase(res, d):
    """Cached map entries point at files in their own folder (works after the cache folder moves)."""
    return {k: os.path.join(d, os.path.basename(v)) if isinstance(v, str) and v.endswith(".pgm") else v
            for k, v in res.items()}


class Camera:
    """Insta360 dual-fisheye recording (.insv): two lens streams + the camera's calibration in the file
    trailer, plus the camera's own low-res .lrv preview for the browser.

    Interface the rest of the app relies on (implement it to support another camera family):
      insv, lrv, model, info{width,height,rate,fps,duration,creation_time}, lrv_size, lenses.project(),
      view_maps(), sphere_maps(), shader_params(), lrv_frames()
    `fit` = lens alignment {rx, ry, rz, fscale}; defaults to the one measured on an X4.
    """
    EXTENSIONS = (".insv",)

    def __init__(self, insv, lrv=None, fit=None):
        self.insv = os.path.abspath(insv)
        self.lrv = os.path.abspath(lrv) if lrv else find_lrv(self.insv)
        out = json.loads(subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries",
             "stream=codec_type,width,height,r_frame_rate:format=duration:format_tags=creation_time",
             "-of", "json", self.insv], capture_output=True, text=True, check=True).stdout)
        vids = [s for s in out["streams"] if s["codec_type"] == "video"]
        if len(vids) < 2:
            raise RuntimeError("%s has only %d lens stream(s). Choose a file with both lenses (cameras that "
                               "split them into _00_ and _10_ files aren't supported yet)." % (os.path.basename(insv), len(vids)))
        num, den = vids[0]["r_frame_rate"].split("/")
        self.N = vids[0]["width"]
        self.info = {"width": self.N, "height": self.N, "rate": vids[0]["r_frame_rate"],
                     "fps": float(num) / float(den), "duration": float(out["format"]["duration"]),
                     "creation_time": out["format"].get("tags", {}).get("creation_time")}
        cal, self.model = stitch360.read_calibration(self.insv)
        self.lenses = stitch360.Lenses(cal, self.N, dict(fit or stitch360.DEFAULT_FIT))
        self.lrv_size = None
        if self.lrv and os.path.exists(self.lrv):
            s = json.loads(subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                           "stream=width,height", "-of", "json", self.lrv],
                                          capture_output=True, text=True, check=True).stdout)["streams"][0]
            self.lrv_size = (s["width"], s["height"])     # side by side: [cockpit lens | forward lens]

    # ---------------------------------------------------------------- geometry

    def view_dirs(self, view, W, H):
        hf = math.radians(view["hfov"])
        tx = math.tan(hf / 2)
        ty = tx * H / W
        i = (2 * np.arange(W) + 1) / W - 1
        j = (2 * np.arange(H) + 1) / H - 1
        I, J = np.meshgrid(i * tx, j * ty)
        r = np.stack([I, J, np.ones_like(I)], -1)
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        return r @ ypr_matrix(view["yaw"], view["pitch"], view["roll"]).T

    def sample_coords(self, d):
        """World directions -> per-lens source pixel coords in the .insv frame and blend weight of each lens."""
        u0, v0, t0 = self.lenses.project(0, d)
        u1, v1, t1 = self.lenses.project(1, d)
        w1 = np.clip(((t0 - t1) / 2) / (2 * BAND_DEG) + 0.5, 0, 1)
        w1 = np.where(t1 > LIMIT_DEG, 0.0, np.where(t0 > LIMIT_DEG, 1.0, w1))
        return (u0, v0), (u1, v1), w1

    def to_lrv(self, k, u, v):
        Wl, Hl = self.lrv_size
        s = Hl / self.N
        x = (u + 0.5) * s - 0.5 + (Wl / 2 if k == 0 else 0)
        y = (v + 0.5) * s - 0.5
        return x, y

    # ---------------------------------------------------------------- ffmpeg remap tables

    def view_maps(self, view, W, H, ss=2):
        """PGM remap tables (nearest pixel, rendered at ss x size then downscaled) for one flat view.
        -> {"lenses": [k...], "x0","y0","x1","y1","mask" paths (only those needed), "size": (W*ss, H*ss)}"""
        key = hashlib.sha1(json.dumps([self.insv, os.path.getsize(self.insv) if os.path.exists(self.insv) else 0,
                                       {k: round(view[k], 4) for k in ("yaw", "pitch", "roll", "hfov")},
                                       W, H, ss, self.lenses.fit]).encode()).hexdigest()[:20]
        d = os.path.join(MAP_DIR, key)
        meta = os.path.join(d, "meta.json")
        if os.path.exists(meta):
            return _rebase(json.load(open(meta)), d)
        os.makedirs(d, exist_ok=True)
        Ws, Hs = W * ss, H * ss
        dirs = self.view_dirs(view, Ws, Hs)
        (u0, v0), (u1, v1), w1 = self.sample_coords(dirs)
        used = [k for k, wk in ((0, 1 - w1), (1, w1)) if wk.max() > 0.002]
        res = {"lenses": used, "size": [Ws, Hs]}
        for k, (u, v) in ((0, (u0, v0)), (1, (u1, v1))):
            if k not in used:
                continue
            X = np.clip(np.floor(u + 0.5), 0, self.N - 1).astype(">u2")
            Y = np.clip(np.floor(v + 0.5), 0, self.N - 1).astype(">u2")
            for name, arr in (("x%d" % k, X), ("y%d" % k, Y)):
                p = os.path.join(d, name + ".pgm")
                with open(p, "wb") as f:
                    f.write(b"P5\n%d %d\n65535\n" % (Ws, Hs))
                    f.write(arr.tobytes())
                res[name] = p
        if len(used) == 2:
            p = os.path.join(d, "mask.pgm")
            with open(p, "wb") as f:
                f.write(b"P5\n%d %d\n255\n" % (Ws, Hs))
                f.write(np.round(w1 * 255).astype(np.uint8).tobytes())
            res["mask"] = p
        json.dump(res, open(meta, "w"))
        return res

    def sphere_maps(self, W, yaw=0.0):
        """Remap tables for a full equirectangular 360 frame (W x W/2), rotated so `yaw` is the centre
        (where 360 players start). Built in row bands to keep memory low. Same return format as view_maps."""
        H = W // 2
        key = hashlib.sha1(json.dumps(["sphere", self.insv, W, round(yaw, 3), self.lenses.fit]).encode()).hexdigest()[:20]
        d = os.path.join(MAP_DIR, key)
        meta = os.path.join(d, "meta.json")
        if os.path.exists(meta):
            return _rebase(json.load(open(meta)), d)
        os.makedirs(d, exist_ok=True)
        R = ypr_matrix(yaw, 0, 0)
        names = ["x0", "y0", "x1", "y1", "mask"]
        files = {n: open(os.path.join(d, n + ".pgm.part"), "wb") for n in names}
        for n in names[:4]:
            files[n].write(b"P5\n%d %d\n65535\n" % (W, H))
        files["mask"].write(b"P5\n%d %d\n255\n" % (W, H))
        for r0 in range(0, H, 256):
            dirs = stitch360.equirect_dirs(W, H, np.arange(r0, min(H, r0 + 256))) @ R.T
            (u0, v0), (u1, v1), w1 = self.sample_coords(dirs)
            for k, (u, v) in ((0, (u0, v0)), (1, (u1, v1))):
                files["x%d" % k].write(np.clip(np.floor(u + 0.5), 0, self.N - 1).astype(">u2").tobytes())
                files["y%d" % k].write(np.clip(np.floor(v + 0.5), 0, self.N - 1).astype(">u2").tobytes())
            files["mask"].write(np.round(w1 * 255).astype(np.uint8).tobytes())
        res = {"lenses": [0, 1], "size": [W, H]}
        for n, f in files.items():
            f.close()
            os.replace(os.path.join(d, n + ".pgm.part"), os.path.join(d, n + ".pgm"))
            res[n] = os.path.join(d, n + ".pgm")
        json.dump(res, open(meta, "w"))
        return res

    # ---------------------------------------------------------------- browser shader parameters

    def shader_params(self):
        L = self.lenses
        out = {"N": self.N, "limit": LIMIT_DEG, "band": BAND_DEG, "lrv": self.lrv_size, "lenses": []}
        for k in (0, 1):
            C = L.cal[k]
            corr = (L.fit["rx"], L.fit["ry"], L.fit["rz"]) if k == 0 else (0, 0, 0)
            M = stitch360.rot(*corr) @ stitch360.R_NOM[k]          # camera axes in world (columns)
            side = C["H"]
            out["lenses"].append({"M": M.tolist(), "xi": C["xi"], "fx": C["fx"] * L.fit["fscale"],
                                  "fy": C["fy"] * L.fit["fscale"], "cx": C["cx"], "cy": C["cy"],
                                  "k": list(C["k"]), "p": list(C["p"]), "crop": (C["W"] / 2 - side) / 2,
                                  "scale": self.N / side})
        return out

    # ---------------------------------------------------------------- small analysis renders from the .lrv

    def lrv_frames(self, view, W, H, start=None, dur=None, keyframes=False, pix="rgb24"):
        """Flat view frames rendered from the low-res preview (fast; for sync/analysis)."""
        if not self.lrv_size:
            raise RuntimeError("No .lrv preview file next to %s" % self.insv)
        dirs = self.view_dirs(view, W, H)
        (u0, v0), (u1, v1), w1 = self.sample_coords(dirs)
        x0, y0 = self.to_lrv(0, u0, v0)
        x1, y1 = self.to_lrv(1, u1, v1)
        X = np.where(w1 > 0.5, x1, x0).astype(np.float32)
        Y = np.where(w1 > 0.5, y1, y0).astype(np.float32)
        cmd = [FFMPEG, "-v", "info" if keyframes else "error", "-nostdin"]
        if keyframes:
            cmd += ["-skip_frame", "nokey"]
        if start:
            cmd += ["-ss", "%.3f" % start]
        if dur:
            cmd += ["-t", "%.3f" % dur]
        cmd += ["-i", self.lrv]
        if keyframes:
            cmd += ["-vf", "showinfo", "-fps_mode", "passthrough"]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        import cv2
        import tempfile
        Wl, Hl = self.lrv_size
        fsize = Wl * Hl * 3
        with tempfile.TemporaryFile() as err:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
            out = []
            while True:                                   # stream: remap each frame as it arrives
                buf = p.stdout.read(fsize)
                if len(buf) < fsize:
                    break
                f = cv2.remap(np.frombuffer(buf, np.uint8).reshape(Hl, Wl, 3), X, Y, cv2.INTER_LINEAR)
                out.append(f.mean(-1).astype(np.uint8) if pix == "gray" else f)
            p.wait()
            err.seek(0)
            log = err.read()
        if p.returncode != 0:
            raise RuntimeError(log.decode(errors="replace")[-500:])
        frames = np.stack(out) if out else np.zeros((0, H, W) + (() if pix == "gray" else (3,)), np.uint8)
        times = None
        if keyframes:
            times = np.array([float(x) for x in re.findall(rb"pts_time:\s*([-0-9.]+)", log)])[:len(frames)]
        return frames, times


# Camera families the app can open. Add a class implementing the Camera interface to support another one.
CAMERA_TYPES = [Camera]


def open_camera(path, preview=None, fit=None):
    ext = os.path.splitext(path)[1].lower()
    for cls in CAMERA_TYPES:
        if ext in cls.EXTENSIONS:
            return cls(path, preview, fit=fit)
    raise ValueError("Unsupported file type %s (supported: %s)" % (
        ext or "?", ", ".join(e for c in CAMERA_TYPES for e in c.EXTENSIONS)))


def find_preview(path):
    return find_lrv(path)


PREVIEW_SIDE = 832      # per lens, same as the camera's own .lrv


def make_preview(cam, out, progress=lambda f: None):
    """Build a low-res preview like the camera's .lrv ([lens 1 | lens 0] side by side) for recordings that
    came without one. Keyframe every second so scrubbing in the browser stays quick."""
    from .system import encoder_args, hwaccel_args
    n = PREVIEW_SIDE
    tmp = out + ".part.mp4"
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-nostdin", "-y", "-progress", "pipe:1"] + hwaccel_args() + [
        "-i", cam.insv, "-filter_complex",
        "[0:v:1]scale=%d:%d[a];[0:v:0]scale=%d:%d[b];[a][b]hstack[v]" % (n, n, n, n),
        "-map", "[v]", "-map", "0:a?"] + encoder_args("h264", 6) + [
        "-g", str(max(1, round(cam.info["fps"]))), "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", "-f", "mp4", tmp]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        k, _, v = line.strip().partition("=")
        if k == "out_time_us" and v.isdigit():
            progress(min(0.999, int(v) / 1e6 / cam.info["duration"]))
    err = p.stderr.read()
    if p.wait() != 0:
        raise RuntimeError(err[-800:])
    os.replace(tmp, out)
    progress(1.0)
    return out
