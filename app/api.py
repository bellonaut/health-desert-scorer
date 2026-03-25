from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from bridge import _NaNSafeEncoder, _json_default, _sanitize_json_tree, build_payload
from data_api import latest_year, load_backend_data

HTML_PATH = APP_DIR / "health_desert_ui.html"
CSS_PATH = APP_DIR / "health_desert_ui.css"
JS_PATH = APP_DIR / "health_desert_ui.js"
SW_PATH = APP_DIR / "sw.js"
ASSETS_DIR = APP_DIR / "assets"
STATIC_DIR = APP_DIR / "static"
LANDING_PATH = STATIC_DIR / "index.html"

app = FastAPI(title="Health Desert NG API")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@lru_cache(maxsize=1)
def _load_frames():
    return load_backend_data(boundary_resolution="low", is_mobile=True)


def _json_response(payload: dict) -> Response:
    body = json.dumps(
        _sanitize_json_tree(payload),
        cls=_NaNSafeEncoder,
        default=_json_default,
        allow_nan=False,
    )
    return Response(content=body, media_type="application/json")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(LANDING_PATH)


@app.get("/ng")
@app.get("/ng/")
def nigeria_dashboard() -> FileResponse:
    return FileResponse(HTML_PATH)


@app.get("/health_desert_ui.html")
def health_desert_ui() -> FileResponse:
    return FileResponse(HTML_PATH)


@app.get("/health_desert_ui.css")
def health_desert_css() -> FileResponse:
    return FileResponse(CSS_PATH, media_type="text/css")


@app.get("/health_desert_ui.js")
def health_desert_js() -> FileResponse:
    return FileResponse(JS_PATH, media_type="application/javascript")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    return FileResponse(SW_PATH, media_type="application/javascript")


@app.get("/api/data")
def data(
    year: str | None = Query(default=None),
    focus: str = Query(default="All risk"),
    state: str | None = Query(default=None),
    lga: str | None = Query(default=None),
    depth: int = Query(default=0),
    compare: str | None = Query(default=None),
    mobile: str | None = Query(default=None),
) -> Response:
    geo_df, shap_df = _load_frames()
    compare_lgas = [uid for uid in str(compare or "").split(",") if uid]
    is_mobile = str(mobile or "").lower() in {"1", "true", "yes"}
    payload = build_payload(
        geo_df,
        shap_df,
        {
            "hd_year": year or latest_year(geo_df),
            "hd_state_filter": state or "All Nigeria",
            "hd_focus": focus,
            "hd_depth": depth,
            "hd_compare_lgas": compare_lgas,
            "hd_selected_lga": lga,
            "hd_is_mobile": is_mobile,
            "hd_parent_app_path": "/ng",
        },
    )
    return _json_response(payload)
