"""Mark an MP4 as a 360° (equirectangular) video for YouTube, Facebook and 360 players — no extra tools.

Writes Google's Spherical Video V1 metadata (an XMP `uuid` box in the video track, what Google's
spatial-media injector writes), then lets ffmpeg add the V2 boxes (sv3d/st3d) from it.
"""
import os
import shutil
import struct
import subprocess

from .system import FFMPEG

SPHERICAL_UUID = bytes.fromhex("ffcc8263f8554a938814587a02521fdd")
XMP = (b'<?xml version="1.0"?><rdf:SphericalVideo xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
       b'xmlns:GSpherical="http://ns.google.com/videos/1.0/spherical/">'
       b"<GSpherical:Spherical>true</GSpherical:Spherical><GSpherical:Stitched>true</GSpherical:Stitched>"
       b"<GSpherical:StitchingSoftware>360 Views</GSpherical:StitchingSoftware>"
       b"<GSpherical:ProjectionType>equirectangular</GSpherical:ProjectionType></rdf:SphericalVideo>")
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta"}


def _boxes(buf, start, end):
    """[(type, offset, size, header_len)] for the boxes in buf[start:end]."""
    out, p = [], start
    while p + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[p:p + 8])
        hl = 8
        if size == 1:
            size = struct.unpack(">Q", buf[p + 8:p + 16])[0]
            hl = 16
        elif size == 0:
            size = end - p
        if size < hl:
            break
        out.append((typ, p, size, hl))
        p += size
    return out


def _top_level(f, file_size):
    out, p = [], 0
    while p + 8 <= file_size:
        f.seek(p)
        size, typ = struct.unpack(">I4s", f.read(8))
        hl = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hl = 16
        elif size == 0:
            size = file_size - p
        if size < hl:
            raise ValueError("corrupt MP4 box at %d" % p)
        out.append((typ, p, size, hl))
        p += size
    return out


def _is_video_trak(moov, trak):
    _, off, size, hl = trak
    for typ, o, s, h in _boxes(moov, off + hl, off + size):
        if typ == b"mdia":
            for t2, o2, s2, h2 in _boxes(moov, o + h, o + s):
                if t2 == b"hdlr":
                    return moov[o2 + h2 + 8:o2 + h2 + 12] == b"vide"
    return False


def _shift_chunk_offsets(moov, start, end, delta):
    for typ, off, size, hl in _boxes(moov, start, end):
        if typ in CONTAINERS:
            _shift_chunk_offsets(moov, off + hl, off + size, delta)
        elif typ in (b"stco", b"co64"):
            n = struct.unpack(">I", moov[off + hl + 4:off + hl + 8])[0]
            p = off + hl + 8
            fmt, w = (">I", 4) if typ == b"stco" else (">Q", 8)
            for _ in range(n):
                v = struct.unpack(fmt, moov[p:p + w])[0] + delta
                if typ == b"stco" and v > 0xFFFFFFFF:
                    raise ValueError("file too large for a 32-bit chunk table")
                moov[p:p + w] = struct.pack(fmt, v)
                p += w


def inject_v1(path):
    """Add the V1 spherical XMP box to the first video track (replacing any existing one)."""
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        top = _top_level(f, file_size)
        mv = next((b for b in top if b[0] == b"moov"), None)
        if mv is None:
            raise ValueError("no moov box")
        _, moff, msize, mhl = mv
        if mhl != 8:
            raise ValueError("64-bit moov not supported")
        f.seek(moff)
        moov = bytearray(f.read(msize))
    traks = [b for b in _boxes(moov, 8, msize) if b[0] == b"trak"]
    trak = next((t for t in traks if _is_video_trak(moov, t)), None)
    if trak is None:
        raise ValueError("no video track")
    _, toff, tsize, thl = trak
    # drop an existing spherical uuid box, then append ours at the end of the trak
    for typ, o, s, h in reversed(_boxes(moov, toff + thl, toff + tsize)):
        if typ == b"uuid" and moov[o + h:o + h + 16] == SPHERICAL_UUID:
            del moov[o:o + s]
            tsize -= s
    box = struct.pack(">I4s", 8 + 16 + len(XMP), b"uuid") + SPHERICAL_UUID + XMP
    moov[toff + tsize:toff + tsize] = box
    new_tsize = tsize + len(box)
    struct.pack_into(">I", moov, toff, new_tsize)
    delta = len(moov) - msize
    struct.pack_into(">I", moov, 0, len(moov))
    mdat_after = any(b[0] == b"mdat" and b[1] > moff for b in top)
    if mdat_after and delta:
        _shift_chunk_offsets(moov, 8, len(moov), delta)
    tmp = path + ".tag.tmp"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(_Range(src, 0, moff), dst, 8 << 20)
        dst.write(moov)
        shutil.copyfileobj(_Range(src, moff + msize, file_size), dst, 8 << 20)
    os.replace(tmp, path)


class _Range:
    def __init__(self, f, start, end):
        self.f, self.left = f, end - start
        f.seek(start)

    def read(self, n=-1):
        n = self.left if n < 0 else min(n, self.left)
        b = self.f.read(n)
        self.left -= len(b)
        return b


def tag_360(path):
    """V1 (XMP) + V2 (sv3d/st3d) spherical metadata, so YouTube and players treat the file as 360."""
    inject_v1(path)
    tmp = path + ".v2.mp4"
    # ffmpeg reads V1 as spherical side data and writes V2 boxes from it (dropping V1, so it's re-added)
    p = subprocess.run([FFMPEG, "-v", "error", "-nostdin", "-y", "-i", path, "-map", "0", "-c", "copy",
                        "-strict", "unofficial", "-movflags", "+faststart", tmp], capture_output=True)
    if p.returncode != 0:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(p.stderr.decode(errors="replace")[-800:])
    os.replace(tmp, path)
    inject_v1(path)
