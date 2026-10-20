"""
citimon_map.py - Lightweight, precomputed SVG geographic map preview for Citimon locations.

Provides a compact (~105 x 70 px) overview map of Slovenia with Citimon stations
plotted as background dots and the selected station highlighted.
Includes graceful fallback for locations with missing/null coordinates.
"""
import os
import json
import math
import base64
from typing import Dict, Optional, Tuple, Any
from dash import html

# SVG canvas dimensions
SVG_WIDTH = 105.0
SVG_HEIGHT = 70.0
SVG_PADDING = 3.5

# Bounding box of Slovenia with margin
MIN_LON, MAX_LON = 13.3831, 16.6079
MIN_LAT, MAX_LAT = 45.4258, 46.8767
MID_LAT = (MIN_LAT + MAX_LAT) / 2.0
COS_LAT = math.cos(math.radians(MID_LAT))

W_SPAN = (MAX_LON - MIN_LON) * COS_LAT
H_SPAN = MAX_LAT - MIN_LAT
SCALE = min((SVG_WIDTH - 2 * SVG_PADDING) / W_SPAN, (SVG_HEIGHT - 2 * SVG_PADDING) / H_SPAN)

X_OFFSET = SVG_PADDING + (SVG_WIDTH - 2 * SVG_PADDING - (W_SPAN * SCALE)) / 2.0
Y_OFFSET = SVG_PADDING + (SVG_HEIGHT - 2 * SVG_PADDING - (H_SPAN * SCALE)) / 2.0


def normalize_location_key(key: str) -> str:
    """Normalize location key for resilient matching (lowercase, hyphens to underscores)."""
    return str(key).strip().lower().replace("-", "_")


def project_coord(lon: float, lat: float) -> Tuple[float, float]:
    """Project WGS84 lon, lat to SVG viewport coordinates."""
    x = X_OFFSET + ((lon - MIN_LON) * COS_LAT) * SCALE
    y = SVG_HEIGHT - (Y_OFFSET + ((lat - MIN_LAT) * SCALE))
    return round(x, 1), round(y, 1)


def _load_boundary() -> str:
    """Load and project Slovenia boundary to SVG path d string."""
    boundary_paths = [
        os.path.join(os.path.dirname(__file__), "assets", "slovenia_boundary.json"),
        os.path.join("assets", "slovenia_boundary.json"),
    ]
    ring = []
    for p in boundary_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    ring = json.load(f)
                break
            except Exception:
                pass

    if not ring:
        # Fallback simplified bounding polygon
        ring = [
            [13.5, 45.5], [14.0, 45.4], [14.8, 45.5], [15.3, 45.6], [15.7, 45.8],
            [16.2, 46.5], [16.6, 46.8], [15.8, 46.7], [15.0, 46.5], [14.2, 46.4],
            [13.6, 46.4], [13.4, 45.8], [13.5, 45.5]
        ]

    projected = [project_coord(pt[0], pt[1]) for pt in ring]
    return "M " + " L ".join(f"{x},{y}" for x, y in projected) + " Z"


def _load_locations() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[float, float]]]:
    """
    Load location coordinates from assets or data directory.
    Returns (raw_dict, valid_projected_coords_dict).
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "assets", "citimon_locations.json"),
        os.path.join("assets", "citimon_locations.json"),
        os.path.join("data", "citimon_locations.json"),
    ]
    raw_dict = {}
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw_dict = json.load(f)
                break
            except Exception:
                pass

    valid_coords = {}
    for loc_id, meta in raw_dict.items():
        if not isinstance(meta, dict):
            continue
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        if lat is not None and lon is not None:
            try:
                f_lat, f_lon = float(lat), float(lon)
                px, py = project_coord(f_lon, f_lat)
                valid_coords[loc_id] = (px, py, f_lat, f_lon)
                # Also store under normalized key
                norm_k = normalize_location_key(loc_id)
                if norm_k not in valid_coords:
                    valid_coords[norm_k] = (px, py, f_lat, f_lon)
            except (ValueError, TypeError):
                pass

    return raw_dict, valid_coords


# Precompute boundary and coordinate index
_BOUNDARY_PATH_D = _load_boundary()
_RAW_LOCATIONS_META, _PROJECTED_COORDS = _load_locations()


def get_location_coordinates(location: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Return (svg_x, svg_y, lat, lon) or None."""
    if not location:
        return None
    loc_str = str(location).strip()
    if loc_str in _PROJECTED_COORDS:
        return _PROJECTED_COORDS[loc_str]
    norm_k = normalize_location_key(loc_str)
    return _PROJECTED_COORDS.get(norm_k)


def _build_svg_string(selected_location: Optional[str] = None) -> str:
    """Generate SVG XML string for a specific selected location."""
    active_coord = get_location_coordinates(selected_location)

    # Base container styling
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(SVG_WIDTH)} {int(SVG_HEIGHT)}" '
        f'width="{int(SVG_WIDTH)}" height="{int(SVG_HEIGHT)}" '
        f'style="background-color: #f8fafc; border-radius: 4px; display: block;">'
    ]

    if active_coord is None:
        # Fallback state: faint map silhouette + centered "No coordinates" badge
        svg_parts.append(
            f'<path d="{_BOUNDARY_PATH_D}" fill="#e2e8f0" stroke="#cbd5e1" stroke-width="0.8" opacity="0.4" />'
        )
        svg_parts.append(
            '<rect x="10" y="24" width="85" height="22" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#e2e8f0" stroke-width="0.8"/>'
        )
        svg_parts.append(
            '<text x="52.5" y="38" text-anchor="middle" font-size="8.5" '
            'font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif" '
            'fill="#64748b" font-weight="500">No coordinates</text>'
        )
    else:
        # Full map state: solid Slovenia outline
        svg_parts.append(
            f'<path d="{_BOUNDARY_PATH_D}" fill="#e2e8f0" stroke="#94a3b8" stroke-width="0.75" />'
        )

        # Background dots for other valid Citimon locations
        # Deduplicate projected coordinate points
        seen_pts = set()
        bg_dots = []
        for loc_id, (px, py, _, _) in _PROJECTED_COORDS.items():
            pt_key = (px, py)
            if pt_key not in seen_pts:
                seen_pts.add(pt_key)
                bg_dots.append(f'<circle cx="{px}" cy="{py}" r="1.5" fill="#64748b" opacity="0.4" />')
        svg_parts.append("".join(bg_dots))

        # Highlighted active marker
        ax, ay, alat, alon = active_coord
        loc_display = str(selected_location or "")
        tooltip = f"{loc_display} ({alat:.4f}&#176; N, {alon:.4f}&#176; E)"
        svg_parts.append(
            f'<g>'
            f'<title>{tooltip}</title>'
            f'<circle cx="{ax}" cy="{ay}" r="4.5" fill="#dc2626" fill-opacity="0.25" />'
            f'<circle cx="{ax}" cy="{ay}" r="2.8" fill="#dc2626" stroke="#ffffff" stroke-width="1.2" />'
            f'</g>'
        )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def _svg_to_data_uri(svg_str: str) -> str:
    """Encode SVG string into standard base64 data URI."""
    encoded = base64.b64encode(svg_str.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


# Precompute all SVGs into memory at startup
_PRECOMPUTED_DATA_URIS: Dict[str, str] = {}
_FALLBACK_DATA_URI = _svg_to_data_uri(_build_svg_string(None))

for loc_id in _RAW_LOCATIONS_META.keys():
    uri = _svg_to_data_uri(_build_svg_string(loc_id))
    _PRECOMPUTED_DATA_URIS[loc_id] = uri
    _PRECOMPUTED_DATA_URIS[normalize_location_key(loc_id)] = uri


def get_map_data_uri(location: Optional[str]) -> str:
    """Return precomputed data URI for location with instant O(1) lookup."""
    if not location:
        return _FALLBACK_DATA_URI
    loc_str = str(location).strip()
    if loc_str in _PRECOMPUTED_DATA_URIS:
        return _PRECOMPUTED_DATA_URIS[loc_str]
    norm_k = normalize_location_key(loc_str)
    if norm_k in _PRECOMPUTED_DATA_URIS:
        return _PRECOMPUTED_DATA_URIS[norm_k]
    # On the fly generation if an unindexed location with valid coords is passed
    return _svg_to_data_uri(_build_svg_string(location))


def render_citimon_map_component(location: Optional[str]) -> html.Div:
    """Render the Dash HTML component containing the map preview."""
    data_uri = get_map_data_uri(location)
    return html.Div(
        children=[
            html.Img(
                src=data_uri,
                alt=f"Map preview for {location}" if location else "Map preview",
                style={
                    "width": f"{int(SVG_WIDTH)}px",
                    "height": f"{int(SVG_HEIGHT)}px",
                    "display": "block",
                    "borderRadius": "4px",
                }
            )
        ],
        id="location-map-preview",
        title=f"Location: {location}" if location else "Location preview",
        style={
            "width": f"{int(SVG_WIDTH)}px",
            "height": f"{int(SVG_HEIGHT)}px",
            "border": "1px solid #d0d7de",
            "borderRadius": "5px",
            "backgroundColor": "#f8fafc",
            "overflow": "hidden",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.05)",
            "flexShrink": "0",
        }
    )


def is_citimon_project(project_id: Optional[str]) -> bool:
    """Return True if project is Citimon (case-insensitive)."""
    return bool(project_id and str(project_id).strip().lower() == "citimon")
