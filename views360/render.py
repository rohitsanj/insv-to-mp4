"""Turning settings into video: parameter validation, ffmpeg filter graphs, the gyro-stabilized main view,
and the render job (parallel chunks joined losslessly, audio added once, optional 360 tagging)."""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from .system import FFMPEG, FramePipe, encoder_args, hwaccel_args
from .spherical import tag_360  # noqa: F401  (used by the job; re-exported)

try:
    import cv2
    import numpy as np
    from . import gyro
    GYRO_ERR = None
except Exception as e:  # opencv/numpy missing
    gyro = None
    GYRO_ERR = "Gyro stabilization unavailable: %s" % e

try:
    from . import hud
    HUD_ERR = None
except Exception as e:  # Pillow missing
    hud = None
    HUD_ERR = "Flight-data overlay unavailable: %s (pip install -r requirements.txt)" % e

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- parameter validation ----------

def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if x != x:
        return default
    return max(lo, min(hi, x))


def _even(x):
    return max(2, int(round(x / 2.0)) * 2)


def clean_view(v):
    v = v or {}
    return {"yaw": _num(v.get("yaw"), -360, 360, 0), "pitch": _num(v.get("pitch"), -90, 90, 0),
            "roll": _num(v.get("roll"), -180, 180, 0), "hfov": _num(v.get("hfov"), 5, 175, 90)}


def clean_hud(h, W, H):
    if not h or not h.get("enabled") or hud is None:
        return None
    out = {"offset": _num(h.get("offset"), -3.2e8, 3.2e8, 0)}
    any_on = False
    for name in ("map", "adi", "data"):
        e = h.get(name) or {}
        w, hh = _even(_num(e.get("w"), 32, W, 300)), _even(_num(e.get("h"), 32, H, 300))
        item = {"on": bool(e.get("on", True)), "w": w, "h": hh,
                "x": int(round(_num(e.get("x"), -w, W, 0))), "y": int(round(_num(e.get("y"), -hh, H, 0)))}
        if name == "map":
            item["scale"] = _num(e.get("scale"), 0.5, 500, 5)
            item["orient"] = "north" if e.get("orient") == "north" else "track"
            item["basemap"] = e.get("basemap") if e.get("basemap") in ("osm", "osm-dim", "none") else "osm-dim"
            for k, dflt in (("pastColor", "#ff40c8"), ("futureColor", "#7fd8ff")):
                v = str(e.get(k, dflt))
                item[k] = v if re.fullmatch(r"#[0-9a-fA-F]{6}", v) else dflt
        out[name] = item
        any_on = any_on or item["on"]
    return out if any_on else None


def clean_params(p, duration, W_max=7680, H_max=4320):
    out = p.get("output", {})
    W = _even(_num(out.get("width"), 160, W_max, 1920))
    H = _even(_num(out.get("height"), 90, H_max, 1080))
    ins = p.get("inset", {})
    iw = _even(_num(ins.get("w"), 16, W, W * 0.3))
    ih = _even(_num(ins.get("h"), 16, H, H * 0.3))
    color = str(ins.get("borderColor", "#ffffff"))
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        color = "#ffffff"
    t_in = _num(p.get("in"), 0, duration, 0)
    t_out = _num(p.get("out"), 0, duration, duration)
    if t_out <= t_in:
        t_out = duration
    return {
        "W": W, "H": H,
        "main": clean_view(p.get("main")),
        "insetView": clean_view(p.get("insetView")),
        "inset": {"enabled": bool(ins.get("enabled", True)),
                  "x": int(round(_num(ins.get("x"), -iw, W, 0))), "y": int(round(_num(ins.get("y"), -ih, H, 0))),
                  "w": iw, "h": ih, "border": int(round(_num(ins.get("border"), 0, 100, 0))),
                  "color": "0x" + color[1:]},
        "codec": "hevc" if out.get("codec") == "hevc" else "h264",
        "bitrate": int(_num(out.get("bitrate"), 2, 200, 20)),
        "in": t_in, "out": t_out,
        "chunks": int(_num(p.get("chunks"), 1, 8, 0)) or None,
        "s360": {"on": bool((p.get("out360") or {}).get("on")),
                 "W": min((3840, 5760, 7680), key=lambda w: abs(w - _num((p.get("out360") or {}).get("width"), 1920, 7680, 7680))),
                 "bitrate": int(_num((p.get("out360") or {}).get("bitrate"), 20, 200, 100))},
        "hud": clean_hud(p.get("hud"), W, H),
        "stab": {"on": bool((p.get("stab") or {}).get("on")),
                 "smooth": round(_num((p.get("stab") or {}).get("smooth"), 0.3, 6, 2), 1)},
    }


SUPERSAMPLE = 2      # views are remapped at 2x then downscaled (antialiasing for the nearest-pixel remap)


def views_graph(cam, views, first_input):
    """Stitch flat views straight from the two fisheye streams of input 0.
    views: [(label, view, W, H)]. Returns (map file inputs, filter graph text) with outputs [label]."""
    maps = [(label, cam.view_maps(v, W, H, SUPERSAMPLE), W, H) for label, v, W, H in views]
    uses = {0: 0, 1: 0}
    for _, m, _, _ in maps:
        for k in m["lenses"]:
            uses[k] += 1
    files, parts, taken = [], [], {0: 0, 1: 0}

    def add(path):
        files.append(path)
        return first_input + len(files) - 1

    for k in (0, 1):
        n = uses[k]
        if n:
            outs = "".join("[l%d_%d]" % (k, i) for i in range(n))
            parts.append("[0:v:%d]format=yuv444p%s%s" % (k, ",split=%d" % n if n > 1 else "", outs))
    for label, m, W, H in maps:
        rs = []
        for k in m["lenses"]:
            xi, yi = add(m["x%d" % k]), add(m["y%d" % k])
            parts.append("[l%d_%d][%d:v][%d:v]remap=fill=black[%s_r%d]" % (k, taken[k], xi, yi, label, k))
            taken[k] += 1
            rs.append("[%s_r%d]" % (label, k))
        if len(rs) == 2:
            mi = add(m["mask"])
            parts.append("[%d:v]format=gray,split=3[%s_ma][%s_mb][%s_mc];[%s_ma][%s_mb][%s_mc]mergeplanes=format=yuv444p:"
                         "map0s=0:map0p=0:map1s=1:map1p=0:map2s=2:map2p=0[%s_mk];%s%s[%s_mk]maskedmerge[%s_mm]"
                         % ((mi,) + (label,) * 7 + (rs[0], rs[1], label, label)))
            src = "[%s_mm]" % label
        else:
            src = rs[0]
        parts.append("%sscale=%d:%d:flags=lanczos[%s]" % (src, W, H, label))
    return files, ";".join(parts)


def filter_graph(c, cam, hud_items=None, main_in=None, hud_in=1, first_map_input=1):
    """main_in: input label of an already-rendered (stabilized) main view. Returns (map inputs, graph)."""
    W, H, ins = c["W"], c["H"], c["inset"]
    views = [] if main_in else [("m", c["main"], W, H)]
    if ins["enabled"]:
        views.append(("ins", c["insetView"], ins["w"], ins["h"]))
    files, g = views_graph(cam, views, first_map_input)
    if main_in:
        g += ";[%s]null[m]" % main_in
    if ins["enabled"]:
        b = ins["border"]
        pad = ",pad={0}:{1}:{2}:{2}:color={3}".format(ins["w"] + 2 * b, ins["h"] + 2 * b, b, ins["color"]) if b > 0 else ""
        g += ";[ins]format=yuv420p{pad}[i];[m][i]overlay=x={x}:y={y}[base]".format(pad=pad, x=ins["x"] - b, y=ins["y"] - b)
    else:
        g += ";[m]null[base]"
    last = "base"
    if hud_items:
        n = len(hud_items)
        g += ";[%d:v]format=rgba," % hud_in + ("split=%d" % n if n > 1 else "null") + "".join("[h%d]" % k for k in range(n))
        for k, (_name, ay, w, h, x, y) in enumerate(hud_items):
            g += ";[h{k}]crop={w}:{h}:0:{ay}[c{k}];[{last}][c{k}]overlay=x={x}:y={y}[o{k}]".format(
                k=k, w=w, h=h, ay=ay, last=last, x=x, y=y)
            last = "o%d" % k
    return files, g + ";[%s]format=yuv420p[v]" % last


def render_cmd(c, cam, output, hud_size=None, hud_items=None, stab_src=None, audio=True, frames=None):
    rate = cam.info["rate"]
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-nostats", "-progress", "pipe:1", "-y"] + hwaccel_args() + [
           "-ss", "%.3f" % c["in"], "-t", "%.3f" % (c["out"] - c["in"]),
           "-i", cam.insv]
    n_in = 1
    if stab_src:
        cmd += ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", "%dx%d" % (c["W"], c["H"]), "-framerate", rate,
                "-thread_queue_size", "64", "-i", stab_src]
        n_in += 1
    if hud_items:
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % hud_size, "-framerate", rate,
                "-thread_queue_size", "64", "-i", "pipe:0"]
        n_in += 1
    files, graph = filter_graph(c, cam, hud_items, main_in="1:v" if stab_src else None,
                                hud_in=2 if stab_src else 1, first_map_input=n_in)
    for f in files:
        cmd += ["-i", f]
    cmd += ["-filter_complex", graph, "-map", "[v]"] + (["-map", "0:a?"] if audio else ["-an"]) + \
        encoder_args(c["codec"], c["bitrate"])
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    if frames:
        cmd += ["-frames:v", str(frames)]     # exact length, so chunks join without drifting from the audio
    cmd += ["-movflags", "+faststart", output]
    return cmd


def sphere_graph(cam, c, first_input):
    """Full 360 (equirectangular) frame stitched from both lenses, centred on the main view's heading."""
    m = cam.sphere_maps(c["s360"]["W"], yaw=c["main"]["yaw"])
    i = first_input
    files = [m["x0"], m["y0"], m["x1"], m["y1"], m["mask"]]
    g = ("[0:v:0]format=yuv444p[l0];[0:v:1]format=yuv444p[l1];"
         "[l0][{x0}:v][{y0}:v]remap=fill=black[r0];[l1][{x1}:v][{y1}:v]remap=fill=black[r1];"
         "[{mk}:v]format=gray,split=3[ma][mb][mc];"
         "[ma][mb][mc]mergeplanes=format=yuv444p:map0s=0:map0p=0:map1s=1:map1p=0:map2s=2:map2p=0[mk];"
         "[r0][r1][mk]maskedmerge,format=yuv420p[v]").format(x0=i, y0=i + 1, x1=i + 2, y1=i + 3, mk=i + 4)
    return files, g


def render360_cmd(c, cam, output, audio=True, frames=None):
    files, g = sphere_graph(cam, c, 1)
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-nostats", "-progress", "pipe:1", "-y"] + hwaccel_args() + [
           "-ss", "%.3f" % c["in"], "-t", "%.3f" % (c["out"] - c["in"]), "-i", cam.insv]
    for f in files:
        cmd += ["-i", f]
    cmd += ["-filter_complex", g, "-map", "[v]"] + (["-map", "0:a?"] if audio else ["-an"]) + \
        encoder_args("hevc", c["s360"]["bitrate"])
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    if frames:
        cmd += ["-frames:v", str(frames)]
    return cmd + ["-movflags", "+faststart", output]


# ---------- render job ----------

class Job:
    """A render, split into parallel chunks (video only) that are joined losslessly, with the audio added once."""

    def __init__(self, chunks=1):
        self.lock = threading.Lock()
        self.chunks = max(1, int(chunks))
        self.procs, self.auxes = [], []
        self.tmpdir = None
        self.finisher = None
        self.state = {"status": "idle"}

    def running(self):
        return self.state.get("status") in ("running", "finishing")

    def start(self, c, cfg):
        with self.lock:
            if self.running():
                return False, "A render is already running."
            cam = cfg["camera"]
            outdir = cfg["prefs"]["outdir"]
            try:
                os.makedirs(outdir, exist_ok=True)
            except OSError as e:
                return False, "Can't use the output folder %s: %s" % (outdir, e)
            if not os.path.exists(cam.insv):
                return False, "Camera file not found (%s). Insert the SD card to render." % cam.insv
            base = os.path.splitext(os.path.basename(cam.insv))[0]
            output = os.path.join(outdir, "%s_views_%s.mp4" % (base, time.strftime("%Y%m%d-%H%M%S")))
            s360 = c["s360"]["on"]
            engine, items, size = (None, None, None) if s360 else hud_setup(c, cfg)
            dev = None
            if c["stab"]["on"] and not s360:
                try:
                    dev = cfg["stab"].dev(c["stab"]["smooth"])
                except Exception as e:
                    return False, "Stabilization unavailable: %s" % e
            fps = cfg["info"]["fps"]
            total = c["out"] - c["in"]
            want = c.get("chunks") or self.chunks
            n = max(1, min(want, int(total // 10) or 1))          # no point splitting very short ranges
            frames = int(round(total * fps))
            fidx = [round(frames * k / n) for k in range(n + 1)]
            edges = [c["in"] + fidx[k] / fps for k in range(n)] + [c["out"]]
            self.tmpdir = tempfile.mkdtemp(prefix=".360views_render_", dir=outdir)
            self.procs, self.auxes, self.parts = [], [], []
            self.done = [0.0] * n
            self.rcs = [None] * n
            self.errs = [[] for _ in range(n)]
            self.state = {"status": "running", "output": output, "duration": total, "done": 0.0, "speed": None,
                          "started": time.time(), "error": None, "chunks": n}
            for k in range(n):
                ck = dict(c, **{"in": edges[k], "out": edges[k + 1]})
                part = os.path.join(self.tmpdir, "part%02d.mp4" % k)
                fifo = FramePipe(self.tmpdir, "main%02d.yuv" % k) if dev is not None else None
                nf = fidx[k + 1] - fidx[k]
                cmd = (render360_cmd(ck, cam, part, audio=False, frames=nf) if s360 else
                       render_cmd(ck, cam, part, size, items, stab_src=fifo and fifo.url, audio=False, frames=nf))
                if k == 0:
                    self.state["command"] = " ".join(_shq(a) for a in cmd)
                feeder = None
                if items:   # instruments for this chunk are drawn by their own process, piped straight into ffmpeg
                    jf = os.path.join(self.tmpdir, "hud%02d.json" % k)
                    with open(jf, "w") as f:
                        json.dump({"track": cfg["track_path"], "hud": ck["hud"], "fps": fps,
                                   "t0": video_start(cfg) + ck["hud"]["offset"] + ck["in"],
                                   "frames": int(math.ceil((ck["out"] - ck["in"]) * fps)) + 2}, f)
                    feeder = subprocess.Popen([sys.executable, "-m", "views360.hudfeed", jf], cwd=PKG_PARENT,
                                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    self.auxes.append(feeder)
                p = subprocess.Popen(cmd, stdin=feeder.stdout if feeder else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False)
                if feeder:
                    feeder.stdout.close()     # ffmpeg owns the pipe now
                self.procs.append(p)
                self.parts.append(part)
                if fifo:
                    m, hf, vf = stab_margin(ck)
                    a = subprocess.Popen(stab_source_cmd(ck, cam, ck["in"], ck["out"] - ck["in"], m, hf, vf),
                                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    self.auxes.append(a)
                    threading.Thread(target=self._warp, args=(a, fifo, ck, m, dev, int(round(ck["in"] * fps))), daemon=True).start()
                threading.Thread(target=self._watch, args=(k, p, c, cfg), daemon=True).start()
            return True, output

    def _feed(self, proc, engine, c, cfg, items, size):
        fps = cfg["info"]["fps"]
        n_frames = int(math.ceil((c["out"] - c["in"]) * fps)) + 2
        t0 = video_start(cfg) + c["hud"]["offset"]
        try:
            for n in range(n_frames):
                img, _ = engine.atlas(c["hud"], t0 + c["in"] + n / fps, items, size)
                proc.stdin.write(img.tobytes())
        except (BrokenPipeError, ValueError, OSError):
            pass
        except Exception as e:  # rendering bug: surface it rather than hang
            self.state["hudError"] = repr(e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def _warp(self, src, fifo, c, m, dev, n0):
        """Stabilized main view: margin frames from `src` -> per-frame gyro warp -> FIFO into the chunk's ffmpeg."""
        W, H = c["W"], c["H"]
        fsize = (W + 2 * m) * (H + 2 * m) * 3 // 2
        try:
            with fifo.open_writer() as out:
                n = 0
                while True:
                    buf = src.stdout.read(fsize)
                    if len(buf) < fsize:
                        break
                    k = min(n0 + n, len(dev) - 1)
                    out.write(warp_yuv(buf, W, H, m, gyro.homography(c["main"], W, H, m, dev[k])))
                    n += 1
        except (BrokenPipeError, OSError, ValueError):
            pass
        except Exception as e:
            self.state["hudError"] = "Stabilization failed: %r\n" % e
        finally:
            try:
                src.terminate()
            except Exception:
                pass

    def _watch(self, k, proc, c, cfg):
        errs = self.errs[k]
        threading.Thread(target=lambda: errs.extend(l.decode(errors="replace") for l in proc.stderr), daemon=True).start()
        for raw in proc.stdout:
            key, _, v = raw.decode(errors="replace").strip().partition("=")
            if key == "out_time_us" and v.isdigit():
                self.done[k] = int(v) / 1e6
                el = time.time() - self.state["started"]
                self.state["done"] = sum(self.done)
                self.state["speed"] = "%.2fx" % (self.state["done"] / el) if el > 0 else None
        rc = proc.wait()
        with self.lock:
            self.rcs[k] = rc
            if self.state["status"] == "cancelled":
                if all(r is not None for r in self.rcs):
                    self._cleanup_tmp()
                return
            if rc != 0:
                self.state["status"] = "error"
                self.state["error"] = (self.state.get("hudError") or "") + "".join(errs[-15:]) or "ffmpeg exited with code %d" % rc
                self.state["finished"] = time.time()
                self._kill_all()
                return
            if any(r is None for r in self.rcs) or self.state["status"] != "running":
                return
            self.state["status"] = "finishing"
        self._finish(c, cfg)

    def _finish(self, c, cfg):
        """Join the video chunks without re-encoding and add the audio once for the whole range."""
        lst = os.path.join(self.tmpdir, "parts.txt")
        with open(lst, "w") as f:
            for p in self.parts:
                f.write("file '%s'\n" % p.replace("'", "'\\''"))
        out = self.state["output"]
        cmd = [FFMPEG, "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", lst,
               "-ss", "%.3f" % c["in"], "-t", "%.3f" % (c["out"] - c["in"]), "-i", cfg["camera"].insv,
               "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart", out]
        p = subprocess.run(cmd, capture_output=True)
        tag_err = None
        if p.returncode == 0 and c["s360"]["on"] and self.state["status"] != "cancelled":
            self.state["stage"] = "Adding 360 metadata…"
            try:
                tag_360(out)
            except Exception as e:
                tag_err = "Couldn't add the 360 metadata: %s" % (getattr(e, "stderr", b"") or str(e))
        with self.lock:
            if self.state["status"] == "cancelled":
                if os.path.exists(out):
                    os.remove(out)
            elif tag_err:
                self.state["status"] = "error"
                self.state["error"] = tag_err
            elif p.returncode == 0:
                self.state["status"] = "done"
                self.state["done"] = self.state["duration"]
            else:
                self.state["status"] = "error"
                self.state["error"] = "Joining the chunks failed: " + p.stderr.decode(errors="replace")[-1500:]
            self.state["finished"] = time.time()
            self._cleanup_tmp()

    def _kill_all(self):
        for p in self.procs + self.auxes:
            if p.poll() is None:
                p.terminate()

    def _cleanup_tmp(self):
        if self.tmpdir:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None

    def cancel(self):
        with self.lock:
            if not self.running():
                return False
            self.state["status"] = "cancelled"
            self._kill_all()
            out = self.state.get("output")
        threading.Thread(target=self._after_cancel, args=(out,), daemon=True).start()
        return True

    def _after_cancel(self, out):
        for p in self.procs:
            p.wait()
        time.sleep(0.3)
        if out and os.path.exists(out):
            os.remove(out)
        with self.lock:
            self._cleanup_tmp()


def _shq(a):
    return a if re.fullmatch(r"[\w@%+=:,./-]+", a) else "'" + a.replace("'", "'\\''") + "'"


def video_start(cfg):
    """UTC epoch of the video's first frame (falls back to the track start)."""
    if cfg.get("start_utc") is not None:
        return cfg["start_utc"]
    if cfg.get("engine"):
        return cfg["engine"].track.t[0]
    return 0.0


def hud_setup(c, cfg):
    engine = cfg.get("engine")
    if not c["hud"] or engine is None:
        return None, None, None
    items, size = engine.layout(c["hud"])
    return (engine, items, size) if items else (None, None, None)


STAB_MAX_DEG = 5.0


class Stabilizer:
    """Gyro data + clock offset for the source video, and cached per-frame corrections."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.t = self.w = self.offset = None
        self.cache = {}

    def available(self):
        if gyro is None:
            return False, GYRO_ERR
        insv = self.cfg.get("insv")
        if not insv:
            return False, "Open a project to enable gyro stabilization."
        if not (os.path.exists(insv) or gyro.has_cached_imu(insv)):
            return False, "Camera file not found (%s). Insert the SD card once so its gyro data can be read." % insv
        return True, None

    def dev(self, smooth):
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        info = self.cfg["info"]
        with self.lock:
            if self.t is None:
                self.t, self.w = gyro.load_imu(self.cfg["insv"], log=lambda m: print(m, flush=True))
            if self.offset is None:
                cam = self.cfg["camera"]
                frames_fn = lambda yaw, start, dur: cam.lrv_frames({"yaw": yaw, "pitch": 0, "roll": 0, "hfov": 90}, 320, 180,
                                                                   start, dur, pix="gray")[0].astype(np.float32)
                self.offset = gyro.estimate_offset(frames_fn, os.path.basename(cam.insv) + "|lrv", self.t, self.w,
                                                   info["duration"], info["fps"], log=lambda m: print(m, flush=True))
            key = round(smooth, 1)
            if key not in self.cache:
                n = int(info["duration"] * info["fps"]) + 4
                self.cache[key] = gyro.corrections(self.t, self.w, self.offset, info["fps"], n, key, STAB_MAX_DEG)
            return self.cache[key]


def stab_margin(c):
    m = gyro.margin_for(c["W"], c["H"], c["main"]["hfov"], STAB_MAX_DEG)
    hf, vf = gyro.margin_fov(c["main"], c["W"], c["H"], m)
    return m, hf, vf


def stab_source_cmd(c, cam, t_in, dur, m, hf, vf, frames=None):
    """The main view with a margin (for the per-frame gyro warp), stitched straight from the camera file."""
    view = dict(c["main"], hfov=hf)
    files, g = views_graph(cam, [("mm", view, c["W"] + 2 * m, c["H"] + 2 * m)], 1)
    cmd = [FFMPEG, "-v", "error", "-nostdin"] + hwaccel_args() + ["-ss", "%.3f" % t_in]
    if dur:
        cmd += ["-t", "%.3f" % dur]
    cmd += ["-i", cam.insv]
    for f in files:
        cmd += ["-i", f]
    cmd += ["-filter_complex", g + ";[mm]format=yuv420p[v]", "-map", "[v]"]
    if frames:
        cmd += ["-frames:v", str(frames)]
    return cmd + ["-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]


def warp_yuv(buf, W, H, m, Hmat):
    """Warp one yuv420p margin frame to W x H with the inverse-map homography (chroma at half resolution)."""
    Wm, Hm = W + 2 * m, H + 2 * m
    a = np.frombuffer(buf, np.uint8)
    Y = a[:Wm * Hm].reshape(Hm, Wm)
    U = a[Wm * Hm:Wm * Hm * 5 // 4].reshape(Hm // 2, Wm // 2)
    V = a[Wm * Hm * 5 // 4:].reshape(Hm // 2, Wm // 2)
    fl = cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP
    Sc = np.array([[0.5, 0, -0.25], [0, 0.5, -0.25], [0, 0, 1.0]])
    Hc = Sc @ Hmat @ np.linalg.inv(Sc)
    y = cv2.warpPerspective(Y, Hmat, (W, H), flags=fl, borderMode=cv2.BORDER_REPLICATE)
    u = cv2.warpPerspective(U, Hc, (W // 2, H // 2), flags=fl, borderMode=cv2.BORDER_REPLICATE)
    v = cv2.warpPerspective(V, Hc, (W // 2, H // 2), flags=fl, borderMode=cv2.BORDER_REPLICATE)
    return y.tobytes() + u.tobytes() + v.tobytes()

