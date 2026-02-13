"""Build an injected HTML snapshot for accessibility checks."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from bridge import build_payload, inject_data_to_html
from data_api import load_backend_data


def main() -> None:
    geo_df, shap_df = load_backend_data(boundary_resolution="low", is_mobile=True)
    payload = build_payload(geo_df, shap_df, {})
    html = inject_data_to_html(Path("app/health_desert_ui.html"), payload)
    out_dir = Path("build")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embedded_ui.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
