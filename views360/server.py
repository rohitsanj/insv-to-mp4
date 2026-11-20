"""360 Views web app: a small local HTTP server + the browser UI in web/index.html.

    python -m views360 [CAMERA_FILE] [--port 8360] [--out FOLDER] [--chunks N] [--no-browser]
    python -m views360 doctor            check that ffmpeg etc. are installed

The server only listens on 127.0.0.1. Renders and stitching happen here with ffmpeg; the browser
plays the camera's low-res preview and stitches it on the GPU for the live view.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__, camera, paths, projects, stitch360, system
from .render import (GYRO_ERR, HUD_ERR, Job, _num, _shq, clean_params, filter_graph, gyro, hud, hud_setup,  # noqa: F401
                     render360_cmd, render_cmd, sphere_graph, stab_margin, stab_source_cmd, video_start, warp_yuv)
from .system import FFMPEG

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = np = None

CHUNK = 1 << 20
VIDEO_CHUNK = 8 << 20
SYNC_LOCK = threading.Lock()


def make_handler(cfg, job):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _raw(self, limit=200 << 20):
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                raise ValueError("too large")
            return self.rfile.read(n) if n else b""

        def _body(self):
            raw = self._raw()
            return json.loads(raw) if raw else {}

        def _state(self):
            base = {"projects": projects.list_all(), "project": cfg.get("project"), "outdir": cfg["prefs"]["outdir"],
                    "prefs": cfg["prefs"], "hudError": HUD_ERR, "videoUrl": cfg.get("videoUrl", "/video"),
                    "version": __version__, "platform": sys.platform, "align": cfg.get("align"),
                    "encoders": {c: system.encoder(c) for c in ("h264", "hevc")}}
            cam = cfg.get("camera")
            if cam is None:
                return base
            settings = None
            sp = os.path.join(cfg["project_dir"], "settings.json")
            if os.path.exists(sp):
                try:
                    with open(sp) as f:
                        settings = json.load(f)
                except ValueError:
                    settings = None
            base.update({"master": cam.insv, "preview": cam.lrv, "info": cfg["info"], "model": cam.model,
                         "lens": cam.shader_params(), "pxPerDeg": cfg["pxPerDeg"], "sourceOnline": os.path.exists(cam.insv),
                         "settings": settings, "track": cfg.get("track"), "videoStart": cfg.get("start_utc"),
                         "stabError": cfg["stab"].available()[1]})
            return base

        def _projects(self, path, data):
            busy = lambda: self._json({"error": "Wait for the render to finish first."}, 409)
            try:
                pid = data.get("id")
                if path == "/api/projects/create":
                    if job.running():
                        return busy()
                    m = projects.create(data.get("name"), data.get("insv"), data.get("lrv") or None)
                    if m.get("lrv"):
                        projects.open_project(cfg, m["id"])
                    return self._json({"ok": True, "project": m, "opened": bool(m.get("lrv"))})
                if path == "/api/projects/open":
                    if job.running():
                        return busy()
                    return self._json({"ok": True, "project": projects.open_project(cfg, pid)})
                if path == "/api/projects/preview":           # (re)build a missing preview
                    m = projects.load(pid)
                    projects.build_preview(pid, camera.open_camera(m["insv"]))
                    return self._json({"ok": True})
                if path == "/api/projects/rename":
                    m = projects.rename(pid, data.get("name"))
                    if cfg.get("project") and cfg["project"]["id"] == pid:
                        cfg["project"]["name"] = m["name"]
                    return self._json({"ok": True, "project": m})
                if path == "/api/projects/delete":
                    if cfg.get("project") and cfg["project"]["id"] == pid:
                        if job.running():
                            return busy()
                        projects.close(cfg)
                    projects.delete(pid)
                    return self._json({"ok": True})
            except Exception as e:
                return self._json({"error": str(e)}, 400)
            self.send_error(404)

        def _pick(self, data):
            """The OS's own file dialog, shown on this computer (the app runs locally)."""
            kind = data.get("kind")
            prompt, exts, folder = {
                "insv": ("Choose the 360 camera recording (.insv)", ["insv"], False),
                "lrv": ("Choose the matching low-res preview (.lrv)", ["lrv", "mp4"], False),
                "outdir": ("Choose where rendered videos are saved", None, True),
            }.get(kind, ("Choose a file", None, False))
            start = data.get("from") or (cfg["prefs"]["outdir"] if folder else system.default_browse_dir())
            try:
                path = system.pick(prompt, start, folder=folder, exts=exts)
            except system.PickerUnavailable as e:
                return self._json({"unsupported": True, "error": str(e)})
            if not path:
                return self._json({"cancelled": True})
            res = {"path": path}
            if kind == "insv":
                res["lrv"] = camera.find_preview(path)
            return self._json(res)

        def _prefs(self, data):
            p = dict(cfg["prefs"])
            if "outdir" in data:
                d = os.path.abspath(os.path.expanduser(str(data["outdir"] or "")))
                if not os.path.isdir(d):
                    return self._json({"error": "Folder not found: %s" % d}, 400)
                p["outdir"] = d
            if "chunks" in data:
                p["chunks"] = int(_num(data["chunks"], 1, 8, 3))
            for k in ("tileUrl", "tileAttribution"):
                if k in data and isinstance(data[k], str) and data[k].strip():
                    p[k] = data[k].strip()[:300]
            if not re.fullmatch(r"https?://\S*\{z\}\S*\{x\}\S*\{y\}\S*", p["tileUrl"]):
                return self._json({"error": "Map tile URL needs {z}, {x} and {y}."}, 400)
            paths.save_prefs(p)
            cfg["prefs"] = paths.load_prefs()
            self._json({"ok": True, "prefs": cfg["prefs"]})

        def _align(self):
            """Measure this camera's lens alignment from its own footage (for cameras other than the X4)."""
            if cfg.get("align", {}).get("status") == "running":
                return self._json({"error": "Already aligning."}, 409)
            cam, pid = cfg["camera"], cfg["project"]["id"]
            if not os.path.exists(cam.insv):
                return self._json({"error": "Camera file not found (%s)." % cam.insv}, 409)
            cfg["align"] = {"status": "running", "log": ""}

            def run():
                try:
                    def log(m):
                        cfg["align"]["log"] = str(m)
                        print(m, flush=True)
                    base = stitch360.Lenses(cam.lenses.cal, cam.N, dict(stitch360.DEFAULT_FIT))
                    fit = stitch360.fit_alignment(cam.insv, base, cam.info["duration"], log)
                    m = projects.load(pid)
                    m["fit"] = {k: float(v) for k, v in fit.items()}
                    projects.save(pid, m)
                    if cfg.get("project") and cfg["project"]["id"] == pid and not job.running():
                        projects.open_project(cfg, pid)
                    cfg["align"] = {"status": "done", "fit": m["fit"]}
                except Exception as e:
                    cfg["align"] = {"status": "error", "error": str(e)}
            threading.Thread(target=run, daemon=True).start()
            self._json({"ok": True})

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                with open(os.path.join(paths.WEB_DIR, "index.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/video":
                self._serve_video()
            elif path == "/api/state":
                self._json(self._state())
            elif path == "/api/render":
                self._json(job.state)
            elif path == "/api/stab":
                self._stab()
            else:
                self.send_error(404)

        def _stab(self):
            if cfg.get("camera") is None:
                return self._json({"error": "No project is open."}, 409)
            q = parse_qs(urlparse(self.path).query)
            smooth = _num((q.get("smooth") or ["2"])[0], 0.3, 6, 2)
            try:
                dev = cfg["stab"].dev(smooth)
            except Exception as e:
                return self._json({"error": str(e)}, 409)
            body = np.ascontiguousarray(dev, "<f4").tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Fps", "%.6f" % cfg["info"]["fps"])
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/api/kml":
                if cfg.get("camera") is None:
                    return self._json({"error": "No project is open."}, 409)
                return self._kml()
            try:
                data = self._body()
            except ValueError:
                return self._json({"error": "bad request"}, 400)
            if path.startswith("/api/projects/"):
                return self._projects(path, data)
            if path == "/api/pick":
                return self._pick(data)
            if path == "/api/prefs":
                return self._prefs(data)
            if cfg.get("camera") is None:
                return self._json({"error": "No project is open."}, 409)
            if path == "/api/settings":
                sp = os.path.join(cfg["project_dir"], "settings.json")
                with open(sp + ".tmp", "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(sp + ".tmp", sp)
                self._json({"ok": True})
            elif path == "/api/render":
                c = clean_params(data, cfg["info"]["duration"])
                ok, msg = job.start(c, cfg)
                self._json({"ok": ok, "message": msg}, 200 if ok else 409)
            elif path == "/api/render/cancel":
                self._json({"ok": job.cancel()})
            elif path == "/api/still":
                self._still(data)
            elif path == "/api/hud":
                self._hud(data)
            elif path == "/api/autosync":
                self._autosync(data)
            elif path == "/api/align":
                if job.running():
                    return self._json({"error": "Wait for the render to finish first."}, 409)
                self._align()
            elif path == "/api/command":
                c = clean_params(data, cfg["info"]["duration"])
                _, items, size = hud_setup(c, cfg)
                out = os.path.join(cfg["prefs"]["outdir"], "OUTPUT.mp4")
                fifo = "STABILIZED_MAIN.yuv" if c["stab"]["on"] else None
                if c["s360"]["on"]:
                    cmd = " ".join(_shq(a) for a in render360_cmd(c, cfg["camera"], out))
                    cmd = "# 360 export (then 360 metadata is added)\n" + cmd
                    return self._json({"command": cmd})
                cmd = " ".join(_shq(a) for a in render_cmd(c, cfg["camera"], out, size, items, fifo))
                if fifo:
                    cmd = "# the gyro-stabilized main view is rendered by 360 Views and fed through a pipe\n" + cmd
                if items:
                    cmd = "# instrument frames are generated by 360 Views and piped into stdin\n" + cmd
                self._json({"command": cmd})
            else:
                self.send_error(404)

        def _kml(self):
            if hud is None:
                return self._json({"error": HUD_ERR}, 500)
            try:
                data = self._raw(100 << 20)
                name = self.headers.get("X-Filename", "track.kml")
                projects.store_track(cfg, data, name)
            except Exception as e:
                return self._json({"error": "Couldn't read that track: %s" % e}, 400)
            self._json({"track": cfg["track"], "videoStart": cfg.get("start_utc")})

        def _autosync(self, data):
            engine = cfg.get("engine")
            if engine is None:
                return self._json({"error": "Load a flight track first."}, 409)
            tr = engine.track
            if tr.bank is None:
                return self._json({"error": "This track has no bank data, so it can't be matched to the horizon."}, 409)
            try:
                from . import autosync
            except Exception as e:
                return self._json({"error": "Auto-sync unavailable: %s" % e}, 500)
            yaw = _num(data.get("yaw"), -360, 360, 180)
            with SYNC_LOCK:
                try:
                    t, a = autosync.horizon_angles(cfg["camera"], yaw, log=lambda m: print(m, flush=True))
                    r = autosync.find_offset(t, a, tr.t, tr.bank, video_start(cfg), cfg["info"]["duration"])
                except Exception as e:
                    return self._json({"error": "Auto-sync failed: %s" % e}, 500)
            if not r:
                return self._json({"error": "Couldn't find enough horizon in the forward view to sync."}, 422)
            self._json(r)

        def _hud(self, data):
            engine = cfg.get("engine")
            if engine is None:
                return self._json({"error": "No track loaded"}, 409)
            c = clean_params(data, cfg["info"]["duration"])
            if not c["hud"]:
                return self._json({"error": "Overlay off"}, 409)
            t = _num(data.get("time"), 0, cfg["info"]["duration"], 0)
            utc = video_start(cfg) + t + c["hud"]["offset"]
            try:
                items, size = engine.layout(c["hud"])
                img, d = engine.atlas(c["hud"], utc, items, size)
            except Exception as e:
                return self._json({"error": "Overlay failed: %s" % e}, 500)
            buf = io.BytesIO()
            img.save(buf, "PNG", compress_level=1)
            body = buf.getvalue()
            notes = []
            if c["hud"]["map"]["on"]:
                m = engine.map_layer(c["hud"]["map"]["w"], c["hud"]["map"]["h"], c["hud"]["map"]["scale"], c["hud"]["map"]["basemap"])
                if m.note:
                    notes.append(m.note)
                if abs(m.scale_nm - c["hud"]["map"]["scale"]) > 0.01:
                    notes.append("Map zoomed out to %.0f NM so the whole flight fits in memory." % m.scale_nm)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Layout", json.dumps([[n, ay, w, h] for n, ay, w, h, _x, _y in items]))
            self.send_header("X-Data", json.dumps({k: d[k] for k in ("t", "valid", "gs", "alt", "course", "bank", "pitch")}))
            self.send_header("X-Note", json.dumps(" ".join(notes)))
            self.end_headers()
            self.wfile.write(body)

        def _still(self, data):
            c = clean_params(data, cfg["info"]["duration"])
            t = _num(data.get("time"), 0, cfg["info"]["duration"], 0)
            engine, items, size = hud_setup(c, cfg)
            cam = cfg["camera"]
            if not os.path.exists(cam.insv):
                return self._json({"error": "Camera file not found (%s). Connect the SD card / drive." % cam.insv}, 409)
            if c["s360"]["on"]:          # the whole 360 frame (shown flat, like a world map)
                files, g = sphere_graph(cam, c, 1)
                cmd = [FFMPEG, "-v", "error", "-ss", "%.3f" % t, "-i", cam.insv]
                for f in files:
                    cmd += ["-i", f]
                cmd += ["-filter_complex", g + ";[v]scale=%d:-2[o]" % min(3840, c["s360"]["W"]), "-map", "[o]",
                        "-frames:v", "1", "-q:v", "2", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
                p = subprocess.run(cmd, capture_output=True)
                if p.returncode != 0 or not p.stdout:
                    return self._json({"error": p.stderr.decode(errors="replace")[-2000:]}, 500)
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(p.stdout)))
                self.end_headers()
                self.wfile.write(p.stdout)
                return
            tmps = []
            cmd = [FFMPEG, "-v", "error", "-ss", "%.3f" % t, "-i", cam.insv]
            try:
                stab = c["stab"]["on"]
                if stab:
                    try:
                        dev = cfg["stab"].dev(c["stab"]["smooth"])
                    except Exception as e:
                        return self._json({"error": "Stabilization unavailable: %s" % e}, 409)
                    m, hf, vf = stab_margin(c)
                    raw = subprocess.run(stab_source_cmd(c, cam, t, None, m, hf, vf, frames=1),
                                         capture_output=True).stdout
                    k = min(int(round(t * cfg["info"]["fps"])), len(dev) - 1)
                    yuv = np.frombuffer(warp_yuv(raw, c["W"], c["H"], m, gyro.homography(c["main"], c["W"], c["H"], m, dev[k])), np.uint8)
                    fd, p_main = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    tmps.append(p_main)
                    cv2.imwrite(p_main, cv2.cvtColor(yuv.reshape(c["H"] * 3 // 2, c["W"]), cv2.COLOR_YUV2BGR_I420))
                    cmd += ["-i", p_main]
                if items:
                    img, _ = engine.atlas(c["hud"], video_start(cfg) + t + c["hud"]["offset"], items, size)
                    fd, p_hud = tempfile.mkstemp(suffix=".png")
                    os.close(fd)
                    tmps.append(p_hud)
                    img.save(p_hud)
                    cmd += ["-i", p_hud]
                files, graph = filter_graph(c, cam, items, main_in="1:v" if stab else None, hud_in=2 if stab else 1,
                                            first_map_input=1 + int(stab) + int(bool(items)))
                for f in files:
                    cmd += ["-i", f]
                cmd += ["-filter_complex", graph, "-map", "[v]", "-frames:v", "1",
                        "-q:v", "2", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
                p = subprocess.run(cmd, capture_output=True)
            finally:
                for f in tmps:
                    os.remove(f)
            if p.returncode != 0 or not p.stdout:
                return self._json({"error": p.stderr.decode(errors="replace")[-2000:]}, 500)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(p.stdout)))
            self.end_headers()
            self.wfile.write(p.stdout)

        def _serve_video(self):
            if not cfg.get("preview"):
                return self.send_error(404)
            serve_range(self, cfg["preview"])

    return H


def serve_range(h, fp, cors=False):
    """Serve a file with HTTP range support (open-ended ranges are answered in chunks)."""
    size = os.path.getsize(fp)
    rng = h.headers.get("Range")
    start, end = 0, size - 1
    m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            if m.group(2):
                end = min(int(m.group(2)), size - 1)
            else:
                end = min(size - 1, start + VIDEO_CHUNK - 1)
        else:
            start = max(0, size - int(m.group(2)))
        if start > end:
            h.send_response(416)
            h.send_header("Content-Range", "bytes */%d" % size)
            h.end_headers()
            return
        h.send_response(206)
        h.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
    else:
        h.send_response(200)
    if cors:
        h.send_header("Access-Control-Allow-Origin", "*")
    h.send_header("Content-Type", "video/mp4")
    h.send_header("Accept-Ranges", "bytes")
    h.send_header("Content-Length", str(end - start + 1))
    h.end_headers()
    try:
        with open(fp, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                buf = f.read(min(CHUNK, left))
                if not buf:
                    break
                h.wfile.write(buf)
                left -= len(buf)
    except (BrokenPipeError, ConnectionResetError):
        pass


def make_video_handler(cfg):
    """The preview video lives on its own port: browsers allow only 6 connections per host:port, and a
    buffering <video> can hold all of them, which would stall the app's other requests."""
    class V(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if urlparse(self.path).path == "/video" and cfg.get("preview"):
                serve_range(self, cfg["preview"], cors=True)
            else:
                self.send_error(404)
    return V



def main(argv=None):
    ap = argparse.ArgumentParser(prog="360views", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("insv", nargs="?", help="optional: a camera recording to open (creates a project for it if needed)")
    ap.add_argument("--port", type=int, default=8360)
    ap.add_argument("--out", help="output folder (remembered)")
    ap.add_argument("--chunks", type=int, help="default number of parallel render pieces (remembered)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    old = os.path.dirname(paths.APP_DIR)      # earlier versions kept data next to the code
    moved = paths.migrate_legacy(old)
    if moved:
        print("Moved earlier data to %s: %s" % (paths.HOME, ", ".join(moved)), flush=True)
    os.makedirs(paths.PROJECTS_DIR, exist_ok=True)
    prefs = paths.load_prefs()
    if a.out:
        prefs["outdir"] = os.path.abspath(os.path.expanduser(a.out))
    if a.chunks:
        prefs["chunks"] = max(1, min(8, a.chunks))
    if a.out or a.chunks:
        paths.save_prefs(prefs)
    cfg = {"prefs": prefs, "camera": None, "project": None, "engine": None, "track": None, "info": None, "preview": None}
    pid = None
    if a.insv:
        src = os.path.abspath(a.insv)
        pid = next((m["id"] for m in projects.list_all() if m.get("insv") == src), None)
        if pid is None:
            pid = projects.create(os.path.splitext(os.path.basename(src))[0], src)["id"]
    else:
        pid = projects.last_opened()
    if pid:
        try:
            m = projects.open_project(cfg, pid)
            print("Opened project “%s”" % m["name"], flush=True)
        except Exception as e:
            print("Couldn't open the last project (%s); choose one in the app." % e, flush=True)
            projects.close(cfg)
    for err in (HUD_ERR, GYRO_ERR):
        if err:
            print(err, flush=True)
    job = Job(prefs["chunks"])
    try:
        vsrv = ThreadingHTTPServer((a.host, a.port + 1), make_video_handler(cfg))
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(cfg, job))
    except OSError as e:
        sys.exit("Port %d or %d is busy (%s). Is 360 Views already running? Try --port %d" % (a.port, a.port + 1, e, a.port + 10))
    threading.Thread(target=vsrv.serve_forever, daemon=True).start()
    cfg["videoUrl"] = "http://127.0.0.1:%d/video" % (a.port + 1)
    url = "http://127.0.0.1:%d/" % a.port
    print("360 Views %s running at %s  (Ctrl+C to stop)" % (__version__, url), flush=True)
    print("Data folder: %s" % paths.HOME, flush=True)
    if not a.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        job.cancel()
        print("\nStopped.")
