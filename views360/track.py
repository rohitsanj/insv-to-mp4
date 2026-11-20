"""Flight/GPS tracks: one Track type, loaded from whatever the user has.

    KML  ForeFlight / Garmin Pilot / Sentry / Stratux exports (gx:Track), incl. pitch & bank when recorded
    GPX  any GPS logger, phone app, Garmin, etc. (<trkpt> with <time>)
    CSV  a header row with time + lat + lon columns; optional altitude, speed, course, pitch, bank
         (column names are matched loosely, e.g. "Timestamp", "Latitude", "AltMSL (ft)", "GndSpd", "Roll")

Groundspeed and course are derived from positions when the file doesn't include them.
Units inside Track: time = UTC epoch seconds, alt = feet, gs = knots, angles = degrees.
"""
import bisect
import csv
import io
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

GX = "{http://www.google.com/kml/ext/2.2}"
KML = "{http://www.opengis.net/kml/2.2}"
M_PER_NM = 1852.0
FT_PER_M = 3.28084
KT_PER_MPS = 3600 / M_PER_NM


def parse_time(s):
    """ISO-8601 time -> UTC epoch seconds (no zone = UTC). Also accepts 'YYYY-MM-DD HH:MM:SS' and epoch numbers."""
    s = str(s).strip()
    if re.fullmatch(r"\d{9,11}(\.\d+)?", s):
        return float(s)
    if re.fullmatch(r"\d{12,14}(\.\d+)?", s):          # epoch milliseconds
        return float(s) / 1000.0
    m = re.match(r"\s*(\d{4}-\d\d-\d\d)[T ](\d\d:\d\d:\d\d)(\.\d+)?\s*(Z|[+-]\d\d:?\d\d)?\s*$", s)
    if not m:
        raise ValueError("bad time: %r" % s)
    t = datetime.strptime(m.group(1) + "T" + m.group(2), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    t += float(m.group(3) or 0)
    tz = m.group(4) or "Z"
    if tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        hh, mm = int(tz[1:3]), int(tz[-2:])
        t -= sign * (hh * 3600 + mm * 60)
    return t


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371008.8 * math.asin(min(1.0, math.sqrt(a)))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.degrees(math.atan2(y, x)) % 360


class Track:
    """Time series of position (+ optional attitude). Build with Track(data, name) or the from_* helpers."""

    def __init__(self, data, name="track.kml"):
        if isinstance(data, str):
            data = data.encode()
        kind = self.sniff(data, name)
        loader = {"kml": self._kml, "gpx": self._gpx, "csv": self._csv}[kind]
        rows, self.meta = loader(data)
        self.format = kind
        self._build(rows)

    @staticmethod
    def sniff(data, name=""):
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext in ("kml", "gpx", "csv"):
            return ext
        head = data[:4096].lstrip().lower()
        if b"<gpx" in head:
            return "gpx"
        if head.startswith(b"<") or b"<kml" in head:
            return "kml"
        return "csv"

    # ------------------------------------------------------------ builders

    def _build(self, rows):
        """rows: dicts with t, lat, lon and optionally alt (ft), gs (kt), course, pitch, bank."""
        rows = sorted((r for r in rows if r.get("t") is not None and r.get("lat") is not None and r.get("lon") is not None),
                      key=lambda r: r["t"])
        dedup = []
        for r in rows:
            if dedup and r["t"] == dedup[-1]["t"]:
                continue
            dedup.append(r)
        if len(dedup) < 2:
            raise ValueError("No timestamped positions found in this file.")
        self.t = [r["t"] for r in dedup]
        self.lat = [r["lat"] for r in dedup]
        self.lon = [r["lon"] for r in dedup]
        col = lambda k: [r[k] for r in dedup] if all(r.get(k) is not None for r in dedup) else None
        self.alt = col("alt") or [r.get("alt") or 0.0 for r in dedup]
        self.gs = col("gs") or self._derive_gs()
        self.course = col("course") or self._derive_course()
        self.bank = col("bank")
        self.pitch = col("pitch")
        self.fields = sorted(k for k in ("alt", "gs", "course", "pitch", "bank") if col(k))

    @staticmethod
    def _kml(data):
        root = ET.fromstring(data)
        rows = []
        for trk in root.iter(GX + "Track"):
            w = [parse_time(e.text) for e in trk if _local(e.tag) == "when"]
            c = [e.text.split() for e in trk if _local(e.tag) == "coord"]
            if len(w) != len(c):
                continue
            arrays = {}
            # data arrays live inside the track, or (ForeFlight) in another Placemark with one value per point
            for scope in (trk, root):
                for sad in scope.iter(GX + "SimpleArrayData"):
                    vals = [v.text for v in sad if _local(v.tag) == "value"]
                    if len(vals) == len(w) and sad.get("name") not in arrays:
                        try:
                            arrays[sad.get("name")] = [float(v) for v in vals]
                        except (TypeError, ValueError):
                            pass
            for i, (t, xyz) in enumerate(zip(w, c)):
                get = lambda k: arrays[k][i] if k in arrays else None
                alt = get("altitude")
                if alt is None and len(xyz) > 2:
                    alt = float(xyz[2]) * FT_PER_M
                rows.append({"t": t, "lon": float(xyz[0]), "lat": float(xyz[1]), "alt": alt, "gs": get("speed_kts"),
                             "course": get("course"), "bank": get("bank"), "pitch": get("pitch")})
        if len(rows) < 2:
            raise ValueError("No timestamped track (gx:Track with <when>/<gx:coord>) found in this KML.")
        meta = {}
        for d in root.iter(KML + "Data"):
            v = d.find(KML + "value")
            if v is not None and v.text:
                meta[d.get("name")] = v.text.strip()
        return rows, meta

    @staticmethod
    def _gpx(data):
        root = ET.fromstring(data)
        rows = []
        for pt in root.iter():
            if _local(pt.tag) not in ("trkpt", "rtept"):
                continue
            r = {"lat": float(pt.get("lat")), "lon": float(pt.get("lon")), "t": None}
            for e in pt.iter():
                n, txt = _local(e.tag), (e.text or "").strip()
                if not txt:
                    continue
                try:
                    if n == "time":
                        r["t"] = parse_time(txt)
                    elif n == "ele":
                        r["alt"] = float(txt) * FT_PER_M
                    elif n == "speed":                    # GPX 1.0 / Garmin extensions: m/s
                        r["gs"] = float(txt) * KT_PER_MPS
                    elif n in ("course", "heading"):
                        r["course"] = float(txt) % 360
                    elif n in ("roll", "bank"):
                        r["bank"] = float(txt)
                    elif n == "pitch":
                        r["pitch"] = float(txt)
                except ValueError:
                    pass
            rows.append(r)
        meta = {}
        name = next((e.text for e in root.iter() if _local(e.tag) == "name" and e.text), None)
        if name:
            meta["flightTitle"] = name.strip()
        return rows, meta

    # CSV column matching: (field, patterns on the normalized header, unit conversion by header)
    _CSV_COLS = [
        ("t", [r"^(utc)?(time|timestamp|datetime|date_?time|gpstime|utctime|lcl_?time)"]),
        ("lat", [r"^lat"]),
        ("lon", [r"^(lon|lng|long)"]),
        ("alt", [r"^(alt|altitude|altmsl|gpsalt|ele|elevation|altgps|altb)"]),
        ("gs", [r"^(gs|gndspd|groundspeed|ground_speed|speed|spd)"]),
        ("course", [r"^(course|trk|track|crs|cog|heading|hdg)"]),
        ("pitch", [r"^pitch"]),
        ("bank", [r"^(bank|roll)"]),
    ]

    @classmethod
    def _csv(cls, data):
        text = data.decode("utf-8-sig", errors="replace")
        lines = [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
        try:
            dialect = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(io.StringIO("\n".join(lines)), dialect)
        rows_raw = list(reader)
        # find the header row (Garmin G1000 logs have a units row before it, etc.)
        hi = next((i for i, r in enumerate(rows_raw[:10]) if any(re.match(r"^\s*lat", c, re.I) for c in r)), None)
        if hi is None:
            raise ValueError("CSV needs a header row with latitude/longitude columns.")
        header = [re.sub(r"[^a-z_]", "", c.strip().lower().replace(" ", "_")) for c in rows_raw[hi]]
        raw_header = [c.lower() for c in rows_raw[hi]]
        idx = {}
        for field, pats in cls._CSV_COLS:
            for i, h in enumerate(header):
                if i not in idx.values() and any(re.match(p, h) for p in pats):
                    idx[field] = i
                    break
        date_i = next((i for i, h in enumerate(header) if h in ("date", "lcl_date", "utc_date")), None)
        if not {"t", "lat", "lon"} <= set(idx):
            raise ValueError("CSV needs time, latitude and longitude columns (found: %s)." % ", ".join(rows_raw[hi]))
        unit = lambda f: raw_header[idx[f]] if f in idx else ""
        alt_m = any(u in unit("alt") for u in ("(m)", "_m", " m", "meter")) or header[idx["alt"]] in ("ele", "elevation") if "alt" in idx else False
        gs_mps = any(u in unit("gs") for u in ("m/s", "mps"))
        gs_kmh = any(u in unit("gs") for u in ("km/h", "kph", "kmh"))
        gs_mph = "mph" in unit("gs")
        rows = []
        for r in rows_raw[hi + 1:]:
            try:
                ts = r[idx["t"]].strip()
                if date_i is not None and not re.match(r"\d{4}-\d\d-\d\d[T ]", ts) and not re.fullmatch(r"[\d.]+", ts):
                    ts = r[date_i].strip() + " " + ts
                row = {"t": parse_time(ts), "lat": float(r[idx["lat"]]), "lon": float(r[idx["lon"]])}
            except (ValueError, IndexError):
                continue
            for f in ("alt", "gs", "course", "pitch", "bank"):
                if f in idx:
                    try:
                        row[f] = float(r[idx[f]])
                    except (ValueError, IndexError):
                        row[f] = None
            if row.get("alt") is not None and alt_m:
                row["alt"] *= FT_PER_M
            if row.get("gs") is not None:
                row["gs"] *= KT_PER_MPS if gs_mps else (1 / 1.852 if gs_kmh else (0.868976 if gs_mph else 1))
            rows.append(row)
        return rows, {}

    # ------------------------------------------------------------ derived values

    def _derive_gs(self):
        n, out = len(self.t), []
        for i in range(n):
            a, b = max(0, i - 2), min(n - 1, i + 2)
            dt = self.t[b] - self.t[a]
            out.append(_haversine_m(self.lat[a], self.lon[a], self.lat[b], self.lon[b]) / dt * 3600 / M_PER_NM if dt > 0 else 0.0)
        return out

    def _derive_course(self):
        n, out = len(self.t), []
        for i in range(n):
            a, b = max(0, i - 1), min(n - 1, i + 1)
            out.append(_bearing(self.lat[a], self.lon[a], self.lat[b], self.lon[b]))
        return out

    def summary(self):
        return {"points": len(self.t), "start": self.t[0], "end": self.t[-1], "format": self.format,
                "hasAttitude": self.bank is not None and self.pitch is not None,
                "fields": self.fields, "tail": self.meta.get("tailNumber", ""),
                "title": self.meta.get("flightTitle", ""), "device": self.meta.get("GPSModelName", "")}

    def at(self, t):
        ts = self.t
        valid = ts[0] - 2 <= t <= ts[-1] + 2
        t = min(max(t, ts[0]), ts[-1])
        i = max(1, min(len(ts) - 1, bisect.bisect_right(ts, t)))
        t0, t1 = ts[i - 1], ts[i]
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        lin = lambda a: a[i - 1] + (a[i] - a[i - 1]) * f
        dc = (self.course[i] - self.course[i - 1] + 540) % 360 - 180
        out = {"t": t, "valid": valid, "lat": lin(self.lat), "lon": lin(self.lon), "alt": lin(self.alt),
               "gs": lin(self.gs), "course": (self.course[i - 1] + dc * f) % 360, "bank": None, "pitch": None}
        if self.bank is not None and self.pitch is not None:
            out["bank"], out["pitch"] = lin(self.bank), lin(self.pitch)
        return out
