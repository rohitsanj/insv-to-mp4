"""Projects: a named recording (+ its preview, settings, flight track and lens alignment) kept in the data
folder. Camera files are only ever referenced, never copied, moved or modified."""
import json
import math
import os
import re
import shutil
import threading
import time

import numpy as np

from . import camera
from .paths import PROJECTS_DIR
from .render import HUD_ERR, Stabilizer, hud

CURRENT_FILE = os.path.join(PROJECTS_DIR, ".current")
TRACK_FILES = ("track.kml", "track.gpx", "track.csv")
_previews = {}          # project id -> {"progress": 0..1, "error": str|None}
_lock = threading.Lock()


def pdir(pid):
    if not re.fullmatch(r"[a-z0-9-]{1,64}", pid or ""):
        raise ValueError("bad project id")
    return os.path.join(PROJECTS_DIR, pid)


def load(pid):
    with open(os.path.join(pdir(pid), "project.json")) as f:
        return json.load(f)


def save(pid, meta):
    tmp = os.path.join(pdir(pid), "project.json.tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, os.path.join(pdir(pid), "project.json"))


def track_file(pid):
    for name in TRACK_FILES:
        p = os.path.join(pdir(pid), name)
        if os.path.exists(p):
            return p
    return None


def list_all():
    out = []
    if os.path.isdir(PROJECTS_DIR):
        for pid in os.listdir(PROJECTS_DIR):
            try:
                m = load(pid)
            except Exception:
                continue
            m["online"] = os.path.exists(m.get("insv", "")) and os.path.exists(m.get("lrv") or "")
            m["hasTrack"] = track_file(pid) is not None
            if pid in _previews:
                m["previewJob"] = dict(_previews[pid])
            out.append(m)
    return sorted(out, key=lambda m: -m.get("updated", 0))


def create(name, src, preview=None):
    """New project for a camera recording. If there's no preview file, one is built in the background
    (the project can be opened when it's done)."""
    name = (name or "").strip()[:80] or "Untitled"
    src = os.path.abspath(os.path.expanduser(src or ""))
    if not os.path.isfile(src):
        raise ValueError("Camera file not found: %s" % src)
    preview = os.path.abspath(os.path.expanduser(preview)) if preview else camera.find_preview(src)
    if preview and not os.path.isfile(preview):
        raise ValueError("Preview file not found: %s" % preview)
    cam = camera.open_camera(src, preview)            # validates: both lenses + calibration
    if preview and not cam.lrv_size:
        raise ValueError("Couldn't read the preview file %s" % preview)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "project"
    pid, n = slug, 2
    while os.path.exists(os.path.join(PROJECTS_DIR, pid)):
        pid = "%s-%d" % (slug, n)
        n += 1
    os.makedirs(pdir(pid))
    now = time.time()
    meta = {"id": pid, "name": name, "insv": src, "lrv": preview, "created": now, "updated": now,
            "model": cam.model, "duration": cam.info["duration"]}
    save(pid, meta)
    if not preview:
        build_preview(pid, cam)
    return meta


def build_preview(pid, cam):
    out = os.path.join(pdir(pid), "preview.mp4")
    with _lock:
        if pid in _previews and _previews[pid].get("error") is None and _previews[pid]["progress"] < 1:
            return
        _previews[pid] = {"progress": 0.0, "error": None}

    def run():
        def prog(f):
            _previews[pid]["progress"] = f
        try:
            camera.make_preview(cam, out, prog)
            meta = load(pid)
            meta["lrv"] = out
            save(pid, meta)
        except Exception as e:
            _previews[pid]["error"] = str(e)
    threading.Thread(target=run, daemon=True).start()


def rename(pid, name):
    m = load(pid)
    m["name"] = (name or "").strip()[:80] or m["name"]
    save(pid, m)
    return m


def delete(pid):
    shutil.rmtree(pdir(pid))          # project settings only; camera files are never touched


def store_track(cfg, data, name):
    """Save an uploaded track (KML/GPX/CSV) into the open project and load it."""
    if hud is None:
        raise RuntimeError(HUD_ERR)
    kind = hud.Track.sniff(data, name)
    tr = hud.Track(data, name)                        # parse before replacing anything
    d = cfg["project_dir"]
    for f in TRACK_FILES + ("track.name",):
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    path = os.path.join(d, "track." + kind)
    with open(path, "wb") as f:
        f.write(data)
    with open(os.path.join(d, "track.name"), "w") as f:
        f.write(name)
    _use_track(cfg, tr, name, path)


def _use_track(cfg, tr, name, path):
    cfg["track_path"] = path                          # the per-chunk instrument drawers read it from here
    cfg["engine"] = hud.HudEngine(tr, log=lambda m: print(m, flush=True))
    cfg["track"] = dict(tr.summary(), name=name)


def open_project(cfg, pid):
    """Point the whole app at a project's camera files, settings and flight track."""
    meta = load(pid)
    job = _previews.get(pid)
    if not meta.get("lrv") and job and not job.get("error"):
        raise ValueError("The preview for “%s” is still being made (%d%%)." % (meta["name"], job["progress"] * 100))
    if not os.path.exists(meta["insv"]):
        raise ValueError("Camera file not found: %s (is the memory card / drive connected?)" % meta["insv"])
    if not meta.get("lrv"):
        raise ValueError("This project has no preview yet — use “Make preview” in the project list.")
    cam = camera.open_camera(meta["insv"], meta.get("lrv"), fit=meta.get("fit"))
    if not cam.lrv_size:
        raise ValueError("Preview file not found: %s" % meta.get("lrv"))
    d = np.array([[0, 0, -1.0], [math.sin(math.radians(1)), 0, -math.cos(math.radians(1))]])
    u, v, _ = cam.lenses.project(0, d)
    cfg.update({"camera": cam, "info": cam.info, "insv": cam.insv, "preview": cam.lrv,
                "pxPerDeg": float(math.hypot(u[1] - u[0], v[1] - v[0])), "project": meta, "project_dir": pdir(pid),
                "engine": None, "track": None, "track_path": None, "start_utc": None})
    start = meta.get("start") or cam.info.get("creation_time")
    if start:
        from .track import parse_time
        try:
            cfg["start_utc"] = parse_time(start)
        except ValueError:
            pass
    cfg["stab"] = Stabilizer(cfg)
    tf = track_file(pid)
    if hud is not None and tf:
        name_f = os.path.join(pdir(pid), "track.name")
        name = open(name_f).read().strip() if os.path.exists(name_f) else os.path.basename(tf)
        try:
            with open(tf, "rb") as f:
                _use_track(cfg, hud.Track(f.read(), tf), name, tf)
        except Exception as e:
            print("Couldn't load the track for %s: %s" % (pid, e))
    meta["updated"] = time.time()
    save(pid, meta)
    with open(CURRENT_FILE, "w") as f:
        f.write(pid)
    return meta


def close(cfg):
    cfg.update({"camera": None, "project": None, "project_dir": None, "engine": None, "track": None})


def last_opened():
    try:
        return open(CURRENT_FILE).read().strip() or None
    except OSError:
        return None
