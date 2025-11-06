# -*- coding: utf-8 -*-
"""
sun_utils.py  (NASA POWER + solar geometry utilities) 111
------------------------------------------------------
Provides:
  • Accurate solar geometry from (lat, lon, UTC datetime)
  • Optional hourly weather query from NASA POWER API
  • Empirical fallback model when offline or API unavailable
  • Lightweight local caching of responses

Dependencies:
  - requests  (for NASA POWER HTTP queries)
Optional:
  - astral >= 2  (for high-precision solar azimuth/elevation)

Coordinate convention:
  - x = East, y = North, z = Up (right-handed)
  - Azimuth measured clockwise from North
  - Altitude 0° = horizon, +90° = zenith
"""

from __future__ import annotations
import os, json, math, hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POWER_BASE = "https://power.larc.nasa.gov/api/temporal/hourly/point"
POWER_PARAMS = ["DNI", "DHI", "GHI", "CLOUD_AMT", "T2M"]
CACHE_DIR = os.path.expanduser("~/.iris_cache/power")
os.makedirs(CACHE_DIR, exist_ok=True)

# Optional environment variables:
#   IRIS_OFFLINE=1        → force offline mode (skip HTTP)
#   IRIS_POWER_TIMEOUT=8  → request timeout seconds
#   IRIS_POWER_NORETRY=1  → disable retry on failure

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SunPose:
    """Solar azimuth and altitude angles (deg)."""
    azimuth_deg: float
    altitude_deg: float

@dataclass
class SunProfile:
    """Full solar profile including irradiance and metadata."""
    timestamp_utc: str
    lat: float
    lon: float
    pose: SunPose
    direction: Tuple[float, float, float]  # (x=East, y=North, z=Up)
    dni_wm2: float
    dhi_wm2: Optional[float] = None
    ghi_wm2: Optional[float] = None
    cloud: Optional[float] = None          # cloud cover %
    t2m_c: Optional[float] = None          # air temperature °C
    source: str = "fallback"               # "power_api" | "fallback"
    cache_hit: bool = False

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def ensure_utc(dt: datetime) -> datetime:
    """Guarantee that datetime is timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def day_key(dt_utc: datetime) -> str:
    """YYYYMMDD string for caching / API query."""
    return dt_utc.strftime("%Y%m%d")

# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------
def _solar_position_fallback(lat_deg: float, lon_deg: float, when_utc: datetime) -> SunPose:
    """
    Simplified analytical solar position (low-accuracy fallback).
    Error <1–2° for most latitudes; sufficient for visualization.
    """
    def _deg(x): return x * 180.0 / math.pi
    def _rad(x): return x * math.pi / 180.0

    d = (when_utc - datetime(2000,1,1,tzinfo=timezone.utc)).total_seconds()/86400.0
    L = (280.46 + 0.9856474*d) % 360
    g = _rad((357.528 + 0.9856003*d) % 360)
    lam = _rad(L + 1.915*math.sin(g) + 0.02*math.sin(2*g))
    eps = _rad(23.439 - 0.0000004*d)

    alpha = math.atan2(math.cos(eps)*math.sin(lam), math.cos(lam))
    delta = math.asin(math.sin(eps)*math.sin(lam))

    # Approximate local hour angle (ignores equation-of-time)
    H = _rad(((when_utc.hour + when_utc.minute/60 + when_utc.second/3600)*15 + lon_deg) - _deg(alpha))
    lat = _rad(lat_deg)

    alt = math.asin(math.sin(lat)*math.sin(delta) + math.cos(lat)*math.cos(delta)*math.cos(H))
    az  = math.atan2(-math.sin(H)*math.cos(delta),
                     math.cos(lat)*math.sin(delta)-math.sin(lat)*math.cos(delta)*math.cos(H))
    return SunPose(azimuth_deg=( _deg(az)+360 )%360, altitude_deg=_deg(alt))

def solar_position(lat_deg: float, lon_deg: float, when_utc: datetime) -> SunPose:
    """Use Astral if available; otherwise fallback."""
    when_utc = ensure_utc(when_utc)
    try:
        from astral import solar
        alt = solar.elevation(when_utc, lat_deg, lon_deg)
        az  = solar.azimuth(when_utc, lat_deg, lon_deg)
        return SunPose(azimuth_deg=az % 360, altitude_deg=alt)
    except Exception:
        return _solar_position_fallback(lat_deg, lon_deg, when_utc)

def to_sun_dir(azimuth_deg: float, altitude_deg: float) -> Tuple[float, float, float]:
    """Convert (azimuth, altitude) → 3D direction vector (x=E, y=N, z=Up)."""
    az, alt = map(math.radians, (azimuth_deg, altitude_deg))
    dx = math.cos(alt) * math.sin(az)
    dy = math.cos(alt) * math.cos(az)
    dz = math.sin(alt)
    return (dx, dy, dz)

# ---------------------------------------------------------------------------
# NASA POWER API utilities
# ---------------------------------------------------------------------------
def _cache_path(lat: float, lon: float, date_yyyymmdd: str) -> str:
    """Deterministic cache file name for given location/date."""
    key = f"{lat:.4f}_{lon:.4f}_{date_yyyymmdd}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"power_{h}.json")

def _load_cache(path: str) -> Optional[dict]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _save_cache(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass

def fetch_weather_power(lat: float, lon: float, dt_utc: datetime) -> Optional[Dict[str, Any]]:
    """
    Query NASA POWER hourly data for the given UTC datetime.
    Returns dictionary {DNI, DHI, GHI, CLOUD_AMT, T2M} or None on failure.
    """
    if os.getenv("IRIS_OFFLINE", "0") == "1":
        return None
    dt_utc = ensure_utc(dt_utc)
    date_key = day_key(dt_utc)
    cache_file = _cache_path(lat, lon, date_key)

    # --- try cache first ---
    blob = _load_cache(cache_file)
    if blob is None:
        import requests
        timeout = int(os.getenv("IRIS_POWER_TIMEOUT", "8"))
        params = ",".join(POWER_PARAMS)
        url = (f"{POWER_BASE}?parameters={params}&community=RE"
               f"&longitude={lon:.4f}&latitude={lat:.4f}"
               f"&start={date_key}&end={date_key}&format=JSON")
        tries = 1 if os.getenv("IRIS_POWER_NORETRY","0")=="1" else 2
        blob = None
        for _ in range(tries):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                blob = r.json()
                _save_cache(cache_file, blob)
                break
            except Exception:
                blob = None
        if blob is None:
            return None

    # --- parse hourly record (nearest hour) ---
    try:
        data = blob["properties"]["parameter"]
        key = dt_utc.strftime("%Y%m%d%H")
        candidates = [key,
                      (dt_utc - timedelta(hours=1)).strftime("%Y%m%d%H"),
                      (dt_utc + timedelta(hours=1)).strftime("%Y%m%d%H")]

        def pick(series: dict) -> Optional[float]:
            for k in candidates:
                if k in series and series[k] is not None:
                    try: return float(series[k])
                    except Exception: return None
            return None

        out = {p: pick(data.get(p, {})) for p in POWER_PARAMS}
        return out
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Irradiance estimation (fallback)
# ---------------------------------------------------------------------------
def estimate_dni(altitude_deg: float, weather: Optional[Dict[str, Any]]) -> float:
    """
    Estimate direct normal irradiance (DNI, W/m²).
    Priority:
      1. Use weather['DNI'] if available
      2. Otherwise use empirical clear-sky model:
           AM = 1 / max(0.1, sin(alt))
           dni = 1367 * exp(-0.14*AM) * sin(alt)
    """
    if weather and isinstance(weather.get("DNI"), (int, float)):
        dni = float(weather["DNI"])
        return max(0.0, min(1200.0, dni))

    alt_rad = math.radians(max(-5.0, altitude_deg))
    s = max(0.0, math.sin(alt_rad))
    if s <= 0:
        return 0.0
    AM = 1.0 / max(0.1, s)
    dni = 1367.0 * math.exp(-0.14 * AM) * s
    return max(0.0, min(1200.0, dni))

# ---------------------------------------------------------------------------
# High-level profile builder
# ---------------------------------------------------------------------------
def solar_profile(lat: float, lon: float, when_utc: datetime) -> SunProfile:
    """
    Compute complete solar profile for given location/time:
      1. Get solar azimuth/altitude/direction
      2. Try NASA POWER (hourly); if missing, fallback model
      3. Return SunProfile dataclass
    """
    when_utc = ensure_utc(when_utc)
    pose = solar_position(lat, lon, when_utc)
    direction = to_sun_dir(pose.azimuth_deg, pose.altitude_deg)

    weather = fetch_weather_power(lat, lon, when_utc)
    source = "fallback" if weather is None else "power_api"

    dni = estimate_dni(pose.altitude_deg, weather)
    dhi = weather.get("DHI") if weather else None
    ghi = weather.get("GHI") if weather else None
    cloud = weather.get("CLOUD_AMT") if weather else None
    t2m = weather.get("T2M") if weather else None

    return SunProfile(
        timestamp_utc=when_utc.isoformat(),
        lat=float(lat), lon=float(lon),
        pose=pose, direction=direction,
        dni_wm2=float(dni),
        dhi_wm2=(float(dhi) if isinstance(dhi,(int,float)) else None),
        ghi_wm2=(float(ghi) if isinstance(ghi,(int,float)) else None),
        cloud=(float(cloud) if isinstance(cloud,(int,float)) else None),
        t2m_c=(float(t2m) if isinstance(t2m,(int,float)) else None),
        source=source, cache_hit=(weather is not None)
    )

# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def solar_position_dict(lat: float, lon: float, when_utc: datetime) -> Dict[str, float]:
    p = solar_position(lat, lon, when_utc)
    dx, dy, dz = to_sun_dir(p.azimuth_deg, p.altitude_deg)
    return {"azimuth_deg": p.azimuth_deg,
            "altitude_deg": p.altitude_deg,
            "direction": [dx, dy, dz]}

def solar_profile_dict(lat: float, lon: float, when_utc: datetime) -> Dict[str, Any]:
    """Return solar profile as JSON-serializable dict."""
    return asdict(solar_profile(lat, lon, when_utc))

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: Tampa, FL on 2025-07-05 14:30 UTC
    ts = datetime(2025,7,5,14,30,0,tzinfo=timezone.utc)
    lat, lon = 27.95, -82.46
    profile = solar_profile(lat, lon, ts)
    print(json.dumps(solar_profile_dict(lat, lon, ts), indent=2))
