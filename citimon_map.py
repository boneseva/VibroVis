"""
citimon_map.py - Lightweight SVG geographic map preview for Citimon.

Provides a compact (~105 x 70 px) overview map of Slovenia. Every Citimon microlocation is a
background dot; the selected recorder's microlocations (optionally narrowed to the microlocations
selected in the filter) are highlighted.

Coordinates come from the Citimon registry (config/citimon_registry.json, see citimon_registry.py).
A selection may be a recorder ID ("C08"), a microlocation ID ("C08-P1") or any historical source-folder
name ("2026_03-C08"). Selections without coordinates (e.g. microlocation "Unassigned") show a
"No coordinates" placeholder. Without a registry, assets/citimon_locations.json is used per folder name.
"""
import os
import json
import math
import base64
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple, Any
from dash import html

import citimon_registry
import multimon_locations

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


def _load_boundary_ring() -> List[List[float]]:
    """Slovenia boundary as [lon, lat] pairs."""
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
    return ring


def _load_boundary() -> str:
    """Load and project Slovenia boundary to SVG path d string."""
    projected = [project_coord(pt[0], pt[1]) for pt in _load_boundary_ring()]
    return "M " + " L ".join(f"{x},{y}" for x, y in projected) + " Z"


_BOUNDARY_PATH_D = _load_boundary()


def _load_assets_locations() -> Dict[str, Tuple[float, float, float, float]]:
    """Fallback without a registry: per-folder coordinates from assets/citimon_locations.json."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "assets", "citimon_locations.json"),
        os.path.join("assets", "citimon_locations.json"),
        os.path.join("data", "citimon_locations.json"),
    ]
    raw_dict: Dict[str, Any] = {}
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    raw_dict = json.load(f)
                break
            except Exception:
                pass
    coords = {}
    for loc_id, meta in raw_dict.items():
        if not isinstance(meta, dict):
            continue
        try:
            f_lat, f_lon = float(meta.get("latitude")), float(meta.get("longitude"))
        except (TypeError, ValueError):
            continue
        px, py = project_coord(f_lon, f_lat)
        coords[loc_id] = (px, py, f_lat, f_lon)
        coords.setdefault(normalize_location_key(loc_id), (px, py, f_lat, f_lon))
    return coords


def _registry():
    try:
        return citimon_registry.load_registry()
    except Exception:
        return None


def is_multimon_project(project_id: Optional[str]) -> bool:
    return bool(project_id and str(project_id).strip().lower() == "multimon")


def _project_key(project: Optional[str]) -> Optional[str]:
    """Cache key for _svg_for: 'Multimon' or None (Citimon, the historical default)."""
    return "Multimon" if is_multimon_project(project) else None


def _multimon():
    try:
        return multimon_locations.load()
    except Exception:
        return None


def _all_points(project: Optional[str] = None) -> Dict[str, Tuple[float, float, float, float]]:
    """Every map point of a project: key -> (svg_x, svg_y, lat, lon).

    Citimon (default): microlocation IDs from the registry (folder names without a registry).
    Multimon: position keys from config/multimon_locations.json."""
    if is_multimon_project(project):
        mm = _multimon()
        if mm is None:
            return {}
        return {k: (*project_coord(lon, lat), lat, lon) for k, (lat, lon) in mm.points.items()}
    reg = _registry()
    if reg is None:
        return _ASSETS_COORDS
    pts = {}
    for mid, m in reg.microlocations.items():
        px, py = project_coord(float(m["longitude"]), float(m["latitude"]))
        pts[mid] = (px, py, float(m["latitude"]), float(m["longitude"]))
    return pts


def resolve_selection(location: Optional[str], microlocations: Optional[Iterable[str]] = None,
                      project: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """(display name, point keys to highlight) for a recorder / microlocation / folder name.

    ``microlocations`` (the microlocation filter) narrows the highlighted points; None means no filter.
    For Multimon the location's positions are returned (its microlocations have no coordinates of their own).
    """
    if not location:
        return None, []
    loc = str(location).strip()
    if is_multimon_project(project):
        mm = _multimon()
        return loc, (mm.resolve(loc) if mm is not None else [])
    reg = _registry()
    if reg is None:
        key = loc if loc in _ASSETS_COORDS else normalize_location_key(loc)
        return loc, ([key] if key in _ASSETS_COORDS else [])
    if loc in reg.microlocations:
        recorder, keys = reg.microlocations[loc]["recorder"], [loc]
    else:
        recorder, keys = reg.resolve_display(loc)
        if recorder is None:
            return loc, []
    if microlocations is not None:
        # Filter values are microlocation IDs (or folder names in caches built before the registry).
        chosen = set()
        for m in microlocations:
            m = str(m)
            info = None if m in reg.microlocations else reg.folder_info(m)
            chosen.add(info.microlocation if info else m)
        keys = [k for k in keys if k in chosen]
    return recorder, keys


def get_location_coordinates(location: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """(svg_x, svg_y, lat, lon) of the first point for a selection, or None when it has no coordinates."""
    _, keys = resolve_selection(location)
    pts = _all_points()
    return pts.get(keys[0]) if keys else None


@lru_cache(maxsize=512)
def _svg_for(display: Optional[str], keys: Tuple[str, ...], project: Optional[str] = None) -> str:
    pts = _all_points(project)
    multimon = is_multimon_project(project)
    active = [(k, pts[k]) for k in keys if k in pts]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(SVG_WIDTH)} {int(SVG_HEIGHT)}" '
        f'width="{int(SVG_WIDTH)}" height="{int(SVG_HEIGHT)}" '
        f'style="background-color: #f8fafc; border-radius: 4px; display: block;">'
    ]

    if not active:
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
        svg_parts.append(
            f'<path d="{_BOUNDARY_PATH_D}" fill="#e2e8f0" stroke="#94a3b8" stroke-width="0.75" />'
        )
        # Background dots for all points (deduplicated by projected position)
        seen_pts = set()
        bg_dots = []
        for px, py, _, _ in pts.values():
            if (px, py) not in seen_pts:
                seen_pts.add((px, py))
                bg_dots.append(f'<circle cx="{px}" cy="{py}" r="1.5" fill="#64748b" opacity="0.4" />')
        svg_parts.append("".join(bg_dots))

        # Highlighted markers for the selection
        reg = None if multimon else _registry()
        for key, (ax, ay, alat, alon) in active:
            owner = None if multimon else display          # Multimon keys already read "<location> · <position>"
            if not owner and reg is not None and key in reg.microlocations:
                owner = reg.microlocations[key]["recorder"]  # several recorders selected: label each by its own
            label = key if not owner or key == owner else f"{owner} &#183; {key}"
            label = label.replace(multimon_locations.SEPARATOR.strip(), "&#183;")
            tooltip = f"{label} ({alat:.4f}&#176; N, {alon:.4f}&#176; E)"
            svg_parts.append(
                f'<g>'
                f'<title>{tooltip}</title>'
                f'<circle cx="{ax}" cy="{ay}" r="4.5" fill="#dc2626" fill-opacity="0.25" />'
                f'<circle cx="{ax}" cy="{ay}" r="2.8" fill="#dc2626" stroke="#ffffff" stroke-width="1.2" />'
                f'</g>'
            )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


def _build_svg_string(selected_location: Optional[str] = None,
                      microlocations: Optional[Iterable[str]] = None, project: Optional[str] = None) -> str:
    """SVG XML string for a selection (recorder, microlocation or folder name)."""
    display, keys = resolve_selection(selected_location, microlocations, project)
    return _svg_for(display, tuple(keys), _project_key(project))


def _svg_to_data_uri(svg_str: str) -> str:
    """Encode SVG string into standard base64 data URI."""
    encoded = base64.b64encode(svg_str.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


_ASSETS_COORDS = _load_assets_locations()

# Precompute one data URI per recorder (or per folder name without a registry) at startup
_PRECOMPUTED_DATA_URIS: Dict[str, str] = {}
_FALLBACK_DATA_URI = _svg_to_data_uri(_svg_for(None, ()))
_reg_at_start = _registry()
for _key in (_reg_at_start.recorders if _reg_at_start is not None else _ASSETS_COORDS):
    _PRECOMPUTED_DATA_URIS[_key] = _svg_to_data_uri(_build_svg_string(_key))


def _multi_selection(locations: Iterable[str], microlocations: Optional[Iterable[str]],
                     project: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """Union of highlighted points for several selected recorders (display None = label per recorder)."""
    keys: List[str] = []
    for loc in locations:
        _, loc_keys = resolve_selection(loc, microlocations, project)
        keys.extend(k for k in loc_keys if k not in keys)
    return None, keys


def get_map_data_uri(location: Any, microlocations: Optional[Iterable[str]] = None,
                     project: Optional[str] = None) -> str:
    """Data URI for a selection (one location or a list of them); precomputed per single Citimon recorder
    when no microlocation filter narrows it."""
    pkey = _project_key(project)
    if isinstance(location, (list, tuple, set)):
        locs = [str(l) for l in location if l]
        if len(locs) > 1:
            micro = list(microlocations) if microlocations is not None else None
            display, keys = _multi_selection(locs, micro, project)
            return _svg_to_data_uri(_svg_for(display, tuple(keys), pkey))
        location = locs[0] if locs else None
    if not location:
        return _FALLBACK_DATA_URI
    loc_str = str(location).strip()
    if pkey is None and microlocations is None and loc_str in _PRECOMPUTED_DATA_URIS:
        return _PRECOMPUTED_DATA_URIS[loc_str]
    return _svg_to_data_uri(_build_svg_string(loc_str, microlocations, project))


def render_citimon_map_component(location: Any,
                                 microlocations: Optional[Iterable[str]] = None,
                                 project: Optional[str] = None) -> html.Div:
    """Render the Dash HTML component containing the map preview.

    ``location`` is the selected recorder (or a historical folder name); ``microlocations`` is the
    microlocation filter value (None = all microlocations of the recorder)."""
    data_uri = get_map_data_uri(location, microlocations, project)
    if isinstance(location, (list, tuple, set)):
        location = ", ".join(str(l) for l in location if l) or None
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
        title=(f"Location: {location}. " if location else "") + "Click to choose locations on the map",
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


def has_map(project_id: Optional[str]) -> bool:
    """Projects with a location map: Citimon (registry) and Multimon (config/multimon_locations.json)."""
    return is_citimon_project(project_id) or (is_multimon_project(project_id) and _multimon() is not None)


# ---- Larger map for choosing locations (opened by clicking the preview) ------------------------------------------

def _point_recorders(project: Optional[str] = None) -> Dict[str, str]:
    """Map point key -> recorder (location value). Without a registry the key is the folder/location itself."""
    if is_multimon_project(project):
        mm = _multimon()
        return dict(mm.owners) if mm is not None else {}
    reg = _registry()
    if reg is None:
        return {k: k for k in _ASSETS_COORDS}
    return {mid: m["recorder"] for mid, m in reg.microlocations.items()}


def picker_figure(selected: Iterable[str], available: Iterable[str], project: Optional[str] = None):
    """Plotly figure of every map point for choosing locations.

    Selected recorders are red, other recorders with data in the current project are blue and clickable,
    points without data are faint grey. customdata holds the recorder (the location-dropdown value).
    """
    import plotly.graph_objects as go
    selected_set = {str(s) for s in (selected or [])}
    available_set = {str(a) for a in (available or [])}
    pts = _all_points(project)
    owners = _point_recorders(project)
    multimon = is_multimon_project(project)
    ring = _load_boundary_ring()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p[0] for p in ring], y=[p[1] for p in ring], mode="lines", fill="toself",
                             fillcolor="#e2e8f0", line=dict(color="#94a3b8", width=1),
                             hoverinfo="skip", showlegend=False))
    groups = {"none": ([], "#cbd5e1", 7), "avail": ([], "#2563eb", 11), "sel": ([], "#dc2626", 13)}
    for key, (_, _, lat, lon) in sorted(pts.items()):
        rec = owners.get(key, key)
        state = "sel" if rec in selected_set else ("avail" if rec in available_set else "none")
        label = key if (multimon or key == rec) else f"{rec} &#183; {key}"
        groups[state][0].append((lon, lat, rec, label + ("" if state != "none" else " (no data)")))
    names = {"none": "No data", "avail": "Not selected", "sel": "Selected"}
    for state in ("none", "avail", "sel"):
        rows, color, size = groups[state]
        if not rows:
            continue
        fig.add_trace(go.Scatter(
            x=[r[0] for r in rows], y=[r[1] for r in rows], mode="markers", name=names[state],
            customdata=[r[2] for r in rows], text=[r[3] for r in rows], hovertemplate="%{text}<extra></extra>",
            marker=dict(size=size, color=color, line=dict(color="#ffffff", width=1.5)),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1)),
        ))
    fig.update_layout(
        margin=dict(l=4, r=4, t=4, b=4), dragmode="select", clickmode="event", uirevision="citimon-picker",
        paper_bgcolor="#f8fafc", plot_bgcolor="#f8fafc", showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="right", x=1.0, bgcolor="rgba(255,255,255,0.7)"),
        xaxis=dict(visible=False, range=[MIN_LON, MAX_LON]),
        yaxis=dict(visible=False, range=[MIN_LAT, MAX_LAT], scaleanchor="x", scaleratio=1 / COS_LAT),
    )
    return fig


def picker_toggle(current: Iterable[str], available: Iterable[str], click_data=None, selected_data=None) -> List[str]:
    """New location selection after a map click (toggles that recorder) or a box/lasso selection (adds all)."""
    available_set = {str(a) for a in (available or [])}
    result = [str(c) for c in (current or [])]
    if selected_data and selected_data.get("points"):
        for p in selected_data["points"]:
            rec = p.get("customdata")
            if rec in available_set and rec not in result:
                result.append(rec)
        return result
    if click_data and click_data.get("points"):
        rec = click_data["points"][0].get("customdata")
        if rec in available_set:
            if rec in result:
                result.remove(rec)
            else:
                result.append(rec)
    return result
