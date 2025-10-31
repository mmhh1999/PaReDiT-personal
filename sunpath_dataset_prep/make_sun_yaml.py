"""
make_sun_yaml.py
------------------------------------------------
Create a directional light configuration (YAML)
for IRIS relighting, based on real geographic
location and UTC time using sun_utils.py.

Usage:
    python make_sun_yaml.py
"""

from datetime import datetime, timezone
import yaml, os, json
from iris.sunpath_dataset_prep.sun_utils import solar_profile_dict


def make_sun_yaml(lat: float, lon: float, iso_utc: str, out_yaml: str):
    """
    Generate a YAML config describing a 'directional' (sun) light
    using the solar direction and irradiance from sun_utils.
    """
    # Parse timestamp
    dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
    # Compute solar parameters
    prof = solar_profile_dict(lat, lon, dt)
    dx, dy, dz = prof["direction"]
    dni = float(prof["dni_wm2"])  # direct normal irradiance (W/m²)

    # IRIS / Mitsuba 'directional' light definition
    light_cfg = {
        "lights": [
            {
                "type": "directional",
                # Direction in world coordinates (x=East, y=North, z=Up)
                "direction": [dx, dy, dz],
                # Irradiance intensity in W/m² (from DNI)
                "irradiance": dni,
                "name": f"sun_{iso_utc}"
            }
        ],
        # Optional: disable learned emitters or indoor lights
        "emitter_scale": 0.0
    }

    os.makedirs(os.path.dirname(out_yaml), exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(light_cfg, f, sort_keys=False)

    # Save the computed metadata alongside for traceability
    with open(out_yaml.replace(".yaml", ".json"), "w", encoding="utf-8") as f:
        json.dump(prof, f, indent=2)
    print("✅ Saved:", out_yaml)


if __name__ == "__main__":
    # Example: Tampa, FL at 2025-07-05 14:30 UTC
    make_sun_yaml(
        lat=27.95,
        lon=-82.46,
        iso_utc="2025-07-05T14:30:00Z",
        out_yaml="configs/scannetpp/bathroom2/relight_sun.yaml"
    )
