"""Map rendering helpers for Health Desert Streamlit app."""

from __future__ import annotations

from typing import Iterable, Optional

import folium
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap
from folium.features import DivIcon
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

# Lightly saturated traffic-light palette
COLOR_GOOD = "#22c55e"
COLOR_WARN = "#eab308"
COLOR_BAD = "#ef4444"
COLOR_BORDER = "#111827"
COLOR_SELECTED = "#ffffff"


def _normalized(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([0.5] * len(series), index=series.index)
    min_val, max_val = numeric.min(), numeric.max()
    if min_val == max_val:
        return pd.Series([0.5] * len(series), index=series.index)
    return (numeric - min_val) / (max_val - min_val)


def _color_for_value(value: float | None, higher_is_worse: bool) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "#374151"  # Neutral gray
    scaled = float(np.clip(value, 0.0, 1.0))
    if not higher_is_worse:
        scaled = 1 - scaled
    if scaled <= 0.33:
        return COLOR_GOOD
    if scaled <= 0.66:
        return COLOR_WARN
    return COLOR_BAD


def _layer_config(layer_name: str) -> tuple[str, bool, str]:
    mapping = {
        "Risk score": ("risk_score", True, "Risk score"),
        "Facilities": ("facilities_per_10k", False, "Facilities / 10k"),
        "Population": ("population_density", False, "Population density"),
        "Towers": ("towers_per_10k", False, "Towers / 10k"),
        "SHAP": ("shap_importance", True, "SHAP importance"),
    }
    return mapping[layer_name]


def render_map(
    geo_df: pd.DataFrame,
    layer_name: str,
    depth_level: int,
    selected_lga_uid: str | None,
    ranked_lgas: pd.DataFrame,
    state_filter: str,
    height: int = 620,
) -> Optional[str]:
    """Render Folium map and return clicked LGA uid, if any."""

    if geo_df.empty:
        return None

    # Ensure identifier is string for stable comparisons
    geo_df = geo_df.copy()
    geo_df["lga_uid"] = geo_df["lga_uid"].astype(str)
    column, higher_is_worse, label = _layer_config(layer_name)
    if column not in geo_df.columns:
        geo_df[column] = np.nan

    geo_df["__norm_value"] = _normalized(geo_df[column])
    color_lookup = {
        row.lga_uid: _color_for_value(row.__norm_value, higher_is_worse)
        for row in geo_df[["lga_uid", "__norm_value"]].itertuples()
    }

    # Build color legend
    colormap = LinearColormap([COLOR_GOOD, COLOR_WARN, COLOR_BAD], vmin=0, vmax=1)
    colormap.caption = label

    center_lat = geo_df.geometry.centroid.y.mean()
    center_lon = geo_df.geometry.centroid.x.mean()
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="cartodbdark_matter")
    Fullscreen(position="topleft").add_to(folium_map)

    def style_function(feature: dict) -> dict:
        uid = str(feature["properties"].get("lga_uid"))
        base_color = color_lookup.get(uid, "#374151")
        border_color = COLOR_SELECTED if selected_lga_uid and uid == str(selected_lga_uid) else COLOR_BORDER
        weight = 3 if selected_lga_uid and uid == str(selected_lga_uid) else 1
        return {
            "fillColor": base_color,
            "color": border_color,
            "weight": weight,
            "fillOpacity": 0.75,
        }

    tooltip_fields = [field for field in ["lga_name", "state_name", column, "risk_score"] if field in geo_df.columns]
    tooltip_aliases = []
    for field in tooltip_fields:
        if field == "lga_name":
            tooltip_aliases.append("LGA:")
        elif field == "state_name":
            tooltip_aliases.append("State:")
        elif field == "risk_score":
            tooltip_aliases.append("Risk score:")
        elif field == "facilities_per_10k":
            tooltip_aliases.append("Facilities / 10k:")
        elif field == "towers_per_10k":
            tooltip_aliases.append("Towers / 10k:")
        elif field == "population_density":
            tooltip_aliases.append("Population density:")
        elif field == "shap_importance":
            tooltip_aliases.append("SHAP importance:")
        else:
            tooltip_aliases.append(f"{field}:")

    folium.GeoJson(
        geo_df,
        name="LGAs",
        style_function=style_function,
        highlight_function=lambda _: {"weight": 3, "color": "#ffffff", "fillOpacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, sticky=False),
    ).add_to(folium_map)

    colormap.add_to(folium_map)

    # Ranked markers for top 10
    for rank, row in enumerate(ranked_lgas.head(10).itertuples(), start=1):
        if hasattr(row, "lga_lat") and hasattr(row, "lga_lon"):
            try:
                lat = float(row.lga_lat)
                lon = float(row.lga_lon)
            except (TypeError, ValueError):
                continue
            folium.Marker(
                location=[lat, lon],
                icon=DivIcon(
                    icon_size=(24, 24),
                    icon_anchor=(12, 12),
                    html=f"<div style='font-size:12px;font-weight:700;color:#f8fafc;'>{rank:02d}</div>",
                ),
            ).add_to(folium_map)

    # Zoom to state filter if applicable
    bounds = geo_df.total_bounds
    if state_filter != "All Nigeria" and len(bounds) == 4:
        minx, miny, maxx, maxy = bounds
        folium_map.fit_bounds([[miny, minx], [maxy, maxx]])

    map_result = st_folium(folium_map, use_container_width=True, height=height, returned_objects=["last_object_clicked"])
    click = map_result.get("last_object_clicked") if isinstance(map_result, dict) else None
    if click and isinstance(click, dict):
        props = click.get("properties", {})
        clicked_uid = props.get("lga_uid") or props.get("lga_name")
        if clicked_uid is not None:
            return str(clicked_uid)
    return None
