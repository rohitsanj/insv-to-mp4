"""Flight-data overlay (HUD) for 360 Views: moving map, attitude indicator and data panel.

The same renderer produces the browser preview and the frames piped into ffmpeg,
so what you see in the preview is what gets rendered.
"""
import bisect
import math
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from . import __version__
from .paths import TILE_DIR, load_prefs
from .track import M_PER_NM, Track, parse_time  # noqa: F401  (re-exported)

_PREFS = load_prefs()
TILE_URL = _PREFS["tileUrl"]
TILE_ATTRIBUTION = _PREFS["tileAttribution"]
TILE_UA = "360Views/%s (open-source 360 flight-video editor)" % __version__
EARTH_CIRC = 40075016.686


# ---------------------------------------------------------------- helpers

def merc(lat, lon):
    lat = max(-85.0, min(85.0, lat))
    s = math.sin(math.radians(lat))
    return (lon + 180.0) / 360.0, 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


_FONTS = {}
# first font that exists wins (macOS, Linux, Windows); Pillow's built-in font is the last resort
SANS_FONTS = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf",
              "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"]
MONO_FONTS = ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
              "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf", "C:/Windows/Fonts/consolab.ttf",
              "DejaVuSansMono-Bold.ttf"] + SANS_FONTS


def font(size, mono=False, bold=True):
    key = (int(size), mono, bold)
    if key not in _FONTS:
        f = None
        for path in (MONO_FONTS if mono else SANS_FONTS):
            try:
                f = ImageFont.truetype(path, max(6, int(size)))
                if bold:
                    try:
                        f.set_variation_by_name("Bold")
                    except Exception:
                        pass
                break
            except OSError:
                continue
        _FONTS[key] = f or ImageFont.load_default(max(6, int(size)))
    return _FONTS[key]


def rounded_mask(w, h, r, ss=4):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), r * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


# ---------------------------------------------------------------- moving map

class MapLayer:
    """Pre-rendered map background (optional OSM tiles + track line) for one widget size/scale."""

    def __init__(self, track, w, h, scale_nm, basemap, log=print):
        self.w, self.h = w, h
        lat0 = sum(track.lat) / len(track.lat)
        # normalized-mercator units per output pixel so the widget is `scale_nm` wide
        self.npp = scale_nm * M_PER_NM / w / (EARTH_CIRC * math.cos(math.radians(lat0)))
        pts = [merc(a, b) for a, b in zip(track.lat, track.lon)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        margin = math.hypot(w, h) / 2 + 4
        while True:
            self.x0 = min(xs) - margin * self.npp
            self.y0 = min(ys) - margin * self.npp
            bw = int(math.ceil((max(xs) - min(xs)) / self.npp + 2 * margin))
            bh = int(math.ceil((max(ys) - min(ys)) / self.npp + 2 * margin))
            if bw * bh <= 64_000_000:
                break
            self.npp *= 1.25  # flight too large for this scale: zoom out until the background fits in memory
        self.scale_nm = self.npp * w * EARTH_CIRC * math.cos(math.radians(lat0)) / M_PER_NM
        self.note = ""
        self.osm = basemap in ("osm", "osm-dim")
        if self.osm:
            try:
                bg = self._tiles(bw, bh, log)
                if basemap == "osm-dim":
                    bg = Image.blend(bg, Image.new("RGB", bg.size, (8, 12, 20)), 0.45)
            except Exception as e:  # network trouble: fall back to a plain background
                self.note = "Basemap unavailable (%s); showing track only." % e
                self.osm = False
                bg = Image.new("RGB", (bw, bh), (16, 22, 32))
        else:
            bg = Image.new("RGB", (bw, bh), (16, 22, 32))
        self.bg = bg.convert("RGBA")
        # track in background-pixel coordinates; drawn per frame so past and future can differ
        self.tt = track.t
        self.tx = [(x - self.x0) / self.npp for x, _ in pts]
        self.ty = [(y - self.y0) / self.npp for _, y in pts]
        self.lw = max(2, round(w / 120))
        self.mask = rounded_mask(w, h, max(6, min(w, h) // 14))
        self.icons = {}

    def _track_overlay(self, ox, oy, cw, ch, now, cur, past_rgb, future_rgb):
        """Track segments inside the crop box (ox, oy, cw, ch), past and future in different colors."""
        ss = 2
        pad = 4 * self.lw
        x0, y0, x1, y1 = ox - pad, oy - pad, ox + cw + pad, oy + ch + pad
        split = bisect.bisect_right(self.tt, now)
        layer = Image.new("RGBA", (cw * ss, ch * ss), (0, 0, 0, 0))
        dr = ImageDraw.Draw(layer)

        def runs(lo, hi, head=None, tail=None):
            """Polyline pieces that touch the crop box (each keeps one point beyond the edge)."""
            seq = ([head] if head else []) + [(self.tx[i], self.ty[i]) for i in range(lo, hi)] + ([tail] if tail else [])
            conv = lambda p: ((p[0] - ox) * ss, (p[1] - oy) * ss)
            out, run, prev = [], [], None
            for p in seq:
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                    if not run and prev is not None:
                        run.append(conv(prev))
                    run.append(conv(p))
                elif run:
                    run.append(conv(p))
                    out.append(run)
                    run = []
                prev = p
            if run:
                out.append(run)
            return [r for r in out if len(r) > 1]

        for run in runs(split, len(self.tt), head=cur):
            dr.line(run, fill=future_rgb + (200,), width=max(2, int(self.lw * ss * 0.75)), joint="curve")
        for run in runs(0, split, tail=cur):
            dr.line(run, fill=past_rgb + (255,), width=self.lw * ss, joint="curve")
        return layer.reduce(ss)

    def _tiles(self, bw, bh, log):
        z = max(0, min(17, int(math.ceil(math.log2(1.0 / (self.npp * 256))))))
        while True:
            n = 2 ** z
            tx0, ty0 = int(self.x0 * n), int(self.y0 * n)
            tx1, ty1 = int((self.x0 + bw * self.npp) * n), int((self.y0 + bh * self.npp) * n)
            count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
            if count <= 400 or z == 0:
                break
            z -= 1
        log("Fetching %d map tiles at zoom %d (OpenStreetMap)…" % (count, z))
        mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256), (16, 22, 32))
        jobs = [(x, y) for x in range(tx0, tx1 + 1) for y in range(ty0, ty1 + 1) if 0 <= y < n]
        with ThreadPoolExecutor(2) as ex:
            tiles = list(ex.map(lambda xy: (xy, _tile(z, xy[0] % n, xy[1])), jobs))
        if all(img is None for _, img in tiles):
            raise RuntimeError("no tiles could be downloaded")
        for (x, y), img in tiles:
            if img is not None:
                mosaic.paste(img, ((x - tx0) * 256, (y - ty0) * 256))
        k = n * 256  # mosaic px per normalized unit
        box = ((self.x0 * k - tx0 * 256), (self.y0 * k - ty0 * 256),
               ((self.x0 + bw * self.npp) * k - tx0 * 256), ((self.y0 + bh * self.npp) * k - ty0 * 256))
        return mosaic.resize((bw, bh), Image.LANCZOS, box=box)

    def icon(self, size):
        """Top-down high-wing aircraft (nose up): the wing is drawn over the fuselage."""
        if size not in self.icons:
            ss, s = 4, size
            S = s * ss
            im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            P = lambda pts: [(S / 2 + x * S, S / 2 + y * S) for x, y in pts]
            fill, ink, ow = (255, 210, 40, 255), (0, 0, 0, 255), max(2, S // 28)
            fuselage = P([(-0.045, -0.40), (0.045, -0.40), (0.075, -0.30), (0.075, -0.05), (0.04, 0.30), (0.025, 0.44),
                          (-0.025, 0.44), (-0.04, 0.30), (-0.075, -0.05), (-0.075, -0.30)])
            d.polygon(fuselage, fill=fill, outline=ink, width=ow)
            d.polygon(P([(-0.20, 0.30), (0.20, 0.30), (0.20, 0.40), (-0.20, 0.40)]), fill=fill, outline=ink, width=ow)  # stabilizer
            d.rounded_rectangle([S / 2 - 0.49 * S, S / 2 - 0.25 * S, S / 2 + 0.49 * S, S / 2 - 0.11 * S],
                                radius=0.03 * S, fill=fill, outline=ink, width=ow)  # high wing, on top of the fuselage
            d.ellipse([S / 2 - 0.035 * S, S / 2 - 0.47 * S, S / 2 + 0.035 * S, S / 2 - 0.40 * S], fill=ink)  # spinner
            self.icons[size] = im.resize((s, s), Image.LANCZOS)
        return self.icons[size]

    def render(self, d, e):
        w, h = self.w, self.h
        mx, my = merc(d["lat"], d["lon"])
        cx, cy = (mx - self.x0) / self.npp, (my - self.y0) / self.npp
        course = d["course"]
        past_rgb, future_rgb = _rgb(e.get("pastColor", "#ff40c8")), _rgb(e.get("futureColor", "#7fd8ff"))
        if e.get("orient") == "track":
            D = int(math.ceil(math.hypot(w, h))) + 2
            ox, oy = round(cx - D / 2), round(cy - D / 2)
            crop = self.bg.crop((ox, oy, ox + D, oy + D))
            crop.alpha_composite(self._track_overlay(ox, oy, D, D, d["t"], (cx, cy), past_rgb, future_rgb))
            crop = crop.rotate(course, resample=Image.BILINEAR)
            out = crop.crop(((D - w) // 2, (D - h) // 2, (D - w) // 2 + w, (D - h) // 2 + h))
            heading_up = 0.0
            north = -course
        else:
            ox, oy = round(cx - w / 2), round(cy - h / 2)
            out = self.bg.crop((ox, oy, ox + w, oy + h))
            out.alpha_composite(self._track_overlay(ox, oy, w, h, d["t"], (cx, cy), past_rgb, future_rgb))
            heading_up = course
            north = 0.0
        out = out.copy()
        s = max(16, min(w, h) // 11)
        ic = self.icon(s).rotate(-heading_up, resample=Image.BICUBIC)
        out.alpha_composite(ic, (w // 2 - s // 2, h // 2 - s // 2))
        dr = ImageDraw.Draw(out)
        # north indicator
        r = max(10, min(w, h) // 16)
        nx, ny = w - r - r // 2 - 4, r + r // 2 + 4
        dr.ellipse((nx - r, ny - r, nx + r, ny + r), fill=(0, 0, 0, 150))
        a = math.radians(north)
        pt = lambda rr, da: (nx + math.sin(a + da) * rr, ny - math.cos(a + da) * rr)
        dr.polygon([pt(r * 0.98, 0), pt(r * 0.62, -0.38), pt(r * 0.62, 0.38)], fill=(255, 80, 80, 255))
        dr.text((nx, ny), "N", font=font(r * 0.85), fill=(255, 255, 255, 255), anchor="mm")
        # scale label
        f2 = font(max(10, min(w, h) // 22))
        label = ("%.1f NM" % self.scale_nm).replace(".0 NM", " NM") + " across"
        dr.text((10, h - 8), label, font=f2, fill=(255, 255, 255, 230), anchor="ls", stroke_width=2, stroke_fill=(0, 0, 0, 200))
        if not d["valid"]:
            dr.text((w // 2, h // 2 + s), "NO DATA", font=f2, fill=(255, 90, 90, 255), anchor="mt", stroke_width=2, stroke_fill=(0, 0, 0, 255))
        # rounded frame
        out.putalpha(Image.composite(out.getchannel("A"), Image.new("L", (w, h), 0), self.mask))
        ImageDraw.Draw(out).rounded_rectangle((0, 0, w - 1, h - 1), max(6, min(w, h) // 14), outline=(255, 255, 255, 200), width=max(2, w // 160))
        if self.osm:
            fa = font(max(8, min(w, h) // 34), bold=False)
            ImageDraw.Draw(out).text((w - 10, h - 8), TILE_ATTRIBUTION, font=fa, fill=(255, 255, 255, 200), anchor="rs",
                                     stroke_width=1, stroke_fill=(0, 0, 0, 160))
        return out


def _rgb(hexstr):
    try:
        return tuple(int(hexstr[i:i + 2], 16) for i in (1, 3, 5))
    except (TypeError, ValueError):
        return (255, 64, 200)


def _tile(z, x, y):
    path = os.path.join(TILE_DIR, str(z), str(x), "%d.png" % y)
    if not os.path.exists(path):
        req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": TILE_UA})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
        except Exception:
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ---------------------------------------------------------------- attitude indicator

class ADI:
    SKY, GROUND = (58, 132, 214), (139, 94, 52)

    def __init__(self, size):
        self.size = size
        self.ss = 2
        S = size * self.ss
        self.S = S
        self.mask = Image.new("L", (S, S), 0)
        ImageDraw.Draw(self.mask).ellipse((1, 1, S - 2, S - 2), fill=255)
        self.ppd = (S / 2) / 28.0
        self.background = self._background()
        self.plane = self._plane()

    def _background(self):
        """Outside-in display: the sky/ground, pitch scale and bank scale never move."""
        S, r, c = self.S, self.S / 2, self.S / 2
        lw = max(2, S // 110)
        im = Image.new("RGBA", (S, S), self.SKY + (255,))
        d = ImageDraw.Draw(im)
        d.rectangle((0, c, S, S), fill=self.GROUND + (255,))
        d.line((0, c, S, c), fill=(255, 255, 255, 255), width=lw)
        f = font(S * 0.05)
        for k in (-20, -15, -10, -5, 5, 10, 15, 20):
            y = c - k * self.ppd
            hw = r * (0.22 if k % 10 == 0 else 0.10)
            d.line((c - hw, y, c + hw, y), fill=(255, 255, 255, 230), width=max(1, lw - 1))
            if k % 10 == 0:
                for sx in (-1, 1):
                    d.text((c + sx * (hw + r * 0.09), y), str(abs(k)), font=f, fill=(255, 255, 255, 230), anchor="mm")
        R = r * 0.86
        d.arc((c - R, c - R, c + R, c + R), 210, 330, fill=(255, 255, 255, 255), width=lw)
        for a in (0, -60, -45, -30, -20, -10, 10, 20, 30, 45, 60):
            L = 0.13 if a in (0, 30, -30, 60, -60) else 0.07
            ar = math.radians(a)
            d.line((c + math.sin(ar) * R, c - math.cos(ar) * R, c + math.sin(ar) * R * (1 + L), c - math.cos(ar) * R * (1 + L)),
                   fill=(255, 255, 255, 255), width=lw)
        out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        out.paste(im, (0, 0), self.mask)
        ImageDraw.Draw(out).ellipse((0, 0, S - 1, S - 1), outline=(20, 20, 20, 255), width=max(3, S // 40))
        return out

    def _plane(self):
        """Miniature aircraft and bank pointer as polygons around the centre (drawn level); rotated per frame."""
        r = self.S / 2
        wl, wt = r * 0.46, max(4, self.S // 45)
        wings = [[(sgn * r * 0.10, -wt / 2), (sgn * wl, -wt / 2), (sgn * wl, wt / 2), (sgn * r * 0.10, wt / 2)] for sgn in (-1, 1)]
        fin = [(-wt * 0.35, -wt * 3.2), (wt * 0.35, -wt * 3.2), (wt * 0.35, -wt), (-wt * 0.35, -wt)]
        R, t = r * 0.86, r * 0.075
        pointer = [(0, -R + t * 0.1), (-t, -R + t * 1.9), (t, -R + t * 1.9)]
        return {"aircraft": wings + [fin], "hub": wt * 1.1, "pointer": pointer}

    def render(self, d):
        S, r, c = self.S, self.S / 2, self.S / 2
        bank = d["bank"] if d["bank"] is not None else 0.0
        pitch = d["pitch"] if d["pitch"] is not None else 0.0
        p = max(-r * 0.7, min(r * 0.7, pitch * self.ppd))
        out = self.background.copy()
        dr = ImageDraw.Draw(out)
        org, blk = (255, 140, 30, 255), (0, 0, 0, 255)
        # right bank -> aircraft rotates clockwise (right wing down); pointer swings right along the fixed bank scale.
        # Aircraft: rotated about its own centre, then lifted by pitch. Pointer: rotated about the dial centre.
        a = math.radians(bank)
        ca, sa = math.cos(a), math.sin(a)
        rot = lambda pts, dy: [(c + x * ca - y * sa, c + x * sa + y * ca + dy) for x, y in pts]
        for poly in self.plane["aircraft"]:
            dr.polygon(rot(poly, -p), fill=org, outline=blk)
        h = self.plane["hub"]
        dr.ellipse((c - h, c - p - h, c + h, c - p + h), fill=org, outline=blk)
        dr.polygon(rot(self.plane["pointer"], 0), fill=org, outline=blk)
        if not d["valid"] or d["bank"] is None:
            msg = "NO DATA" if not d["valid"] else "NO ATTITUDE"
            ImageDraw.Draw(out).text((c, c + r * 0.4), msg, font=font(S * 0.08), fill=(255, 70, 70, 255), anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 255))
        return out.reduce(self.ss) if out.width == self.size * self.ss else out.resize((self.size, self.size), Image.LANCZOS)


# ---------------------------------------------------------------- data panel

class DataPanel:
    def __init__(self, w, h):
        self.w, self.h = w, h
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, h - 1), max(6, h // 10), fill=(0, 0, 0, 150), outline=(255, 255, 255, 90), width=max(1, h // 90))
        self.base = im
        self.lab = font(h * 0.13)
        self.val = font(h * 0.30, mono=True)
        self.unit = font(h * 0.13)

    def render(self, d):
        out = self.base.copy()
        dr = ImageDraw.Draw(out)
        w, h = self.w, self.h
        rows = [("GS", "%d" % round(d["gs"]) if d["valid"] else "---", "kt"),
                ("ALT", "{:,}".format(int(round(d["alt"] / 10.0) * 10)) if d["valid"] else "---", "ft")]
        pad = w * 0.07
        for i, (lab, val, unit) in enumerate(rows):
            yc = h * (0.29 + 0.44 * i)
            dr.text((pad, yc), lab, font=self.lab, fill=(170, 182, 200, 255), anchor="lm")
            ux = w - pad
            dr.text((ux, yc + h * 0.06), unit, font=self.unit, fill=(170, 182, 200, 255), anchor="rs")
            uw = dr.textlength(unit, font=self.unit)
            dr.text((ux - uw - w * 0.03, yc + h * 0.11), val, font=self.val, fill=(255, 255, 255, 255), anchor="rs")
        return out


# ---------------------------------------------------------------- engine

ELEMENTS = ("map", "adi", "data")


class HudEngine:
    def __init__(self, track, log=print):
        self.track = track
        self.log = log
        self.lock = threading.Lock()
        self.maps, self.adis, self.panels = {}, {}, {}

    def map_layer(self, w, h, scale, basemap):
        key = (w, h, round(scale, 3), basemap)
        with self.lock:
            if key not in self.maps:
                if len(self.maps) > 4:
                    self.maps.clear()
                self.maps[key] = MapLayer(self.track, w, h, scale, basemap, self.log)
            return self.maps[key]

    def _adi(self, size):
        with self.lock:
            if size not in self.adis:
                self.adis[size] = ADI(size)
            return self.adis[size]

    def _panel(self, w, h):
        with self.lock:
            if (w, h) not in self.panels:
                self.panels[(w, h)] = DataPanel(w, h)
            return self.panels[(w, h)]

    def layout(self, hud):
        """Atlas layout: list of (name, atlas_y, w, h, out_x, out_y); plus atlas size."""
        items, ay, aw = [], 0, 2
        for name in ELEMENTS:
            e = hud[name]
            if not e["on"]:
                continue
            w, h = e["w"], e["h"]
            if name == "adi":
                w = h = min(w, h)
            items.append((name, ay, w, h, e["x"], e["y"]))
            ay += h
            aw = max(aw, w)
        return items, (aw + aw % 2, max(2, ay + ay % 2))

    def values(self, utc):
        return self.track.at(utc)

    def atlas(self, hud, utc, items=None, size=None):
        if items is None:
            items, size = self.layout(hud)
        d = self.track.at(utc)
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        for name, ay, w, h, _x, _y in items:
            e = hud[name]
            if name == "map":
                piece = self.map_layer(w, h, e["scale"], e["basemap"]).render(d, e)
            elif name == "adi":
                piece = self._adi(w).render(d)
            else:
                piece = self._panel(w, h).render(d)
            img.paste(piece, (0, ay))
        return img, d
