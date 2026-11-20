"""What this computer can do: where ffmpeg is, which hardware video encoder/decoder to use, native file
dialogs, and a health check (`python -m views360 doctor`). Everything platform-specific lives here."""
import os
import shutil
import socket
import subprocess
import sys
import time

_EXTRA_BIN = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]


def _which(name, env):
    p = os.environ.get(env)
    if p:
        return p
    found = shutil.which(name)
    if found:
        return found
    for d in _EXTRA_BIN:
        c = os.path.join(d, name)
        if os.access(c, os.X_OK):
            return c
    return None


FFMPEG = _which("ffmpeg", "VIEWS360_FFMPEG") or "ffmpeg"
FFPROBE = _which("ffprobe", "VIEWS360_FFPROBE") or "ffprobe"
IS_MAC = sys.platform == "darwin"
IS_WIN = os.name == "nt"

# Encoders in order of preference; the first one that actually works on this machine is used.
_ENCODERS = {
    "h264": ["h264_videotoolbox", "h264_nvenc", "h264_qsv", "h264_amf", "libx264"],
    "hevc": ["hevc_videotoolbox", "hevc_nvenc", "hevc_qsv", "hevc_amf", "libx265"],
}
_enc_cache = {}


def _encoder_works(enc):
    """Listed isn't enough (e.g. nvenc is compiled in but there's no NVIDIA GPU): try a tiny encode."""
    try:
        p = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-f", "lavfi", "-i", "color=c=gray:s=640x360:d=0.2",
                            "-pix_fmt", "yuv420p", "-c:v", enc, "-b:v", "2M", "-f", "null", "-"],
                           capture_output=True, timeout=30)
        return p.returncode == 0
    except Exception:
        return False


def encoder(codec):
    """Best working encoder for 'h264' or 'hevc'. Set VIEWS360_SOFTWARE=1 to force the CPU encoders."""
    if codec not in _enc_cache:
        cands = _ENCODERS[codec]
        if os.environ.get("VIEWS360_SOFTWARE"):
            cands = cands[-1:]
        _enc_cache[codec] = next((e for e in cands if _encoder_works(e)), cands[-1])
    return _enc_cache[codec]


def encoder_args(codec, mbps):
    enc = encoder(codec)
    args = ["-c:v", enc, "-b:v", "%dM" % mbps]
    if enc.startswith("lib"):              # CPU encoders: bound the bitrate and pick a sane speed
        args += ["-maxrate", "%dM" % int(mbps * 1.5), "-bufsize", "%dM" % (mbps * 2), "-preset", "fast"]
    if codec == "hevc":
        args += ["-tag:v", "hvc1"]         # plays in QuickTime/Safari/iOS
    return args + ["-pix_fmt", "yuv420p"]


def hwaccel_args():
    """Hardware decoding of the camera's HEVC streams when available (frames come back to the CPU)."""
    if os.environ.get("VIEWS360_SOFTWARE"):
        return []
    return ["-hwaccel", "videotoolbox"] if IS_MAC else ["-hwaccel", "auto"]


# ---------------------------------------------------------------- piping raw frames into ffmpeg

class FramePipe:
    """A second raw-video input for ffmpeg (stdin is taken by the instrument frames).
    Named pipe on macOS/Linux; a localhost socket on Windows (or when VIEWS360_TCP_PIPE=1)."""

    def __init__(self, tmpdir, name):
        self.tcp = IS_WIN or bool(os.environ.get("VIEWS360_TCP_PIPE")) or not hasattr(os, "mkfifo")
        if self.tcp:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
            s.close()
            self.url = "tcp://127.0.0.1:%d?listen=1" % self.port
        else:
            self.url = os.path.join(tmpdir, name)
            os.mkfifo(self.url)

    def open_writer(self, timeout=120):
        if not self.tcp:
            return open(self.url, "wb")
        end = time.time() + timeout
        while True:                         # ffmpeg starts listening once it reaches this input
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                sock.settimeout(None)
                return sock.makefile("wb")
            except OSError:
                if time.time() > end:
                    raise
                time.sleep(0.1)


# ---------------------------------------------------------------- native file dialogs

class PickerUnavailable(Exception):
    pass


def pick(prompt, start=None, folder=False, exts=None):
    """Show the OS's own open-file (or choose-folder) dialog on this computer. Returns a path, or None
    if cancelled. Raises PickerUnavailable when there's no dialog tool (the app then asks for a typed path)."""
    start = start or os.path.expanduser("~")
    if os.path.isfile(start):
        start = os.path.dirname(start)
    if not os.path.isdir(start):
        start = os.path.expanduser("~")
    if IS_MAC:
        q = lambda s: s.replace("\\", "").replace('"', "'")
        what = "choose folder" if folder else "choose file"
        script = ('tell application "System Events" to activate\n'
                  'set f to %s with prompt "%s" default location (POSIX file "%s")\n'
                  'POSIX path of f' % (what, q(prompt), q(start)))
        p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        return p.stdout.strip() or None if p.returncode == 0 else None
    if IS_WIN:
        if folder:
            ps = ("Add-Type -AssemblyName System.Windows.Forms;$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
                  "$d.Description='%s';$d.SelectedPath='%s';if($d.ShowDialog() -eq 'OK'){$d.SelectedPath}")
        else:
            ps = ("Add-Type -AssemblyName System.Windows.Forms;$d=New-Object System.Windows.Forms.OpenFileDialog;"
                  "$d.Title='%s';$d.InitialDirectory='%s';if($d.ShowDialog() -eq 'OK'){$d.FileName}")
        q = lambda s: s.replace("'", "''")
        p = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps % (q(prompt), q(start))],
                           capture_output=True, text=True)
        return p.stdout.strip() or None
    if shutil.which("zenity"):
        cmd = ["zenity", "--file-selection", "--title", prompt, "--filename", start.rstrip("/") + "/"]
        if folder:
            cmd.append("--directory")
        elif exts:
            cmd += ["--file-filter", " ".join("*." + e for e in exts), "--file-filter", "*"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.stdout.strip() or None
    if shutil.which("kdialog"):
        cmd = (["kdialog", "--getexistingdirectory", start] if folder else
               ["kdialog", "--getopenfilename", start, " ".join("*." + e for e in exts or [])])
        p = subprocess.run(cmd + ["--title", prompt], capture_output=True, text=True)
        return p.stdout.strip() or None
    raise PickerUnavailable("No file dialog available on this system; type or paste the path instead.")


def default_browse_dir():
    """Where memory cards show up."""
    for d in ("/Volumes", "/media/" + os.environ.get("USER", ""), "/run/media/" + os.environ.get("USER", "")):
        if os.path.isdir(d):
            return d
    return os.path.expanduser("~")


# ---------------------------------------------------------------- health check

def doctor():
    """Print what works and what's missing. Returns True when the essentials are present."""
    ok = True

    def line(good, what, detail=""):
        print("  %s %-26s %s" % ("✓" if good else ("✗" if good is False else "–"), what, detail))

    print("360 Views — system check")
    line(sys.version_info >= (3, 9), "Python 3.9+", sys.version.split()[0])
    ok &= sys.version_info >= (3, 9)
    for mod, why in (("numpy", "required"), ("PIL", "flight-data overlay"), ("cv2", "stabilization & auto-sync")):
        try:
            __import__(mod)
            line(True, mod)
        except Exception as e:
            line(False if mod == "numpy" else None, mod, "missing (%s): pip install -r requirements.txt" % why)
            ok &= mod != "numpy"
    try:
        v = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.split("\n")[0]
        line(True, "ffmpeg", "%s (%s)" % (v.split(" Copyright")[0], FFMPEG))
        filters = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        need = [f for f in ("remap", "maskedmerge", "mergeplanes") if " %s " % f not in filters]
        line(not need, "ffmpeg filters", "missing: " + ", ".join(need) if need else "remap, maskedmerge, mergeplanes")
        ok &= not need
        subprocess.run([FFPROBE, "-version"], capture_output=True, check=True)
        line(True, "ffprobe")
        for c in ("h264", "hevc"):
            e = encoder(c)
            line(True, "%s encoder" % c, e + ("  (CPU — slower)" if e.startswith("lib") else "  (hardware)"))
    except Exception:
        line(False, "ffmpeg / ffprobe", "not found — " + install_hint("ffmpeg"))
        ok = False
    from . import paths
    line(True, "data folder", paths.HOME)
    pk = "osascript" if IS_MAC else "PowerShell" if IS_WIN else (shutil.which("zenity") or shutil.which("kdialog") or None)
    line(bool(pk) or None, "file dialogs", pk or "none found (you can paste paths instead); install zenity")
    print("\nAll set." if ok else "\nSomething required is missing (see ✗ above).")
    return ok


def install_hint(tool):
    if IS_MAC:
        return "brew install %s" % tool
    if IS_WIN:
        return "winget install Gyan.FFmpeg"
    return "sudo apt install %s" % tool
