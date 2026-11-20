"""Where 360 Views keeps its data: projects, preferences and caches live in a per-user folder, never
inside the app's own folder, so the app can be installed anywhere (or updated) without losing work.

    macOS    ~/Library/Application Support/360 Views
    Linux    $XDG_DATA_HOME/360-views  (default ~/.local/share/360-views)
    Windows  %APPDATA%\\360 Views

Set VIEWS360_HOME to use a different folder.
"""
import json
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(APP_DIR, "web")


def _default_home():
    if os.environ.get("VIEWS360_HOME"):
        return os.path.abspath(os.path.expanduser(os.environ["VIEWS360_HOME"]))
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/360 Views")
    if os.name == "nt":
        return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "360 Views")
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "360-views")


HOME = _default_home()
PROJECTS_DIR = os.path.join(HOME, "projects")
CACHE_DIR = os.path.join(HOME, "cache")
MAP_DIR = os.path.join(CACHE_DIR, "maps")          # ffmpeg remap tables per view
TILE_DIR = os.path.join(CACHE_DIR, "tiles")        # map tiles
PREFS_PATH = os.path.join(HOME, "prefs.json")


def cache(name):
    """Path for a small cache file (gyro data, sync results…)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def default_outdir():
    for d in ("~/Movies", "~/Videos", "~"):
        d = os.path.expanduser(d)
        if os.path.isdir(d):
            return d
    return os.path.expanduser("~")


DEFAULT_PREFS = {
    "outdir": None,            # None = default_outdir()
    "chunks": 3,
    "tileUrl": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "tileAttribution": "© OpenStreetMap",
}


def load_prefs():
    p = dict(DEFAULT_PREFS)
    try:
        with open(PREFS_PATH) as f:
            p.update({k: v for k, v in json.load(f).items() if k in DEFAULT_PREFS})
    except (OSError, ValueError):
        pass
    if not p["outdir"]:
        p["outdir"] = default_outdir()
    return p


def save_prefs(p):
    os.makedirs(HOME, exist_ok=True)
    tmp = PREFS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({k: p.get(k) for k in DEFAULT_PREFS}, f, indent=2)
    os.replace(tmp, PREFS_PATH)


def migrate_legacy(old_dir):
    """Move data written by earlier versions (which kept everything next to the code) into HOME."""
    import glob
    import shutil
    moves = [("projects", PROJECTS_DIR), (".maps", MAP_DIR), ("tilecache", TILE_DIR)]
    moved = []
    for src, dst in moves:
        s = os.path.join(old_dir, src)
        if os.path.isdir(s) and not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(s, dst)
            moved.append(src)
    for pat in (".imu_*.npz", ".imuoff_*.txt", ".sync_*.npz"):
        for s in glob.glob(os.path.join(old_dir, pat)):
            dst = cache(os.path.basename(s).lstrip("."))
            if not os.path.exists(dst):
                shutil.move(s, dst)
                moved.append(os.path.basename(s))
    leftovers = [f for f in ("settings.json", "track.kml", "track.name", ".track_current.kml", ".proxy_progress.txt",
                             ".restitch.log") if os.path.exists(os.path.join(old_dir, f))]
    if leftovers:
        keep = os.path.join(HOME, "old-app-files")
        os.makedirs(keep, exist_ok=True)
        for f in leftovers:
            shutil.move(os.path.join(old_dir, f), os.path.join(keep, f))
            moved.append(f)
    return moved
