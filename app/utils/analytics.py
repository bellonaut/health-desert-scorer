"""Analytics helpers for user testing instrumentation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS_DIR = Path("user_testing") / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def log_event(session_id: str | None, persona: str, event_type: str, details: dict[str, Any] | None = None) -> None:
    """Append a structured event to logs and session file."""
    if not session_id:
        return

    payload = {
        "timestamp": time.time(),
        "session_id": session_id,
        "persona": persona,
        "event_type": event_type,
        "details": details or {},
    }

    log_path = LOG_DIR / "app_events.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    session_path = SESSIONS_DIR / f"{session_id}.json"
    if session_path.exists():
        try:
            existing = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {"session_id": session_id, "persona": persona, "events": []}
    else:
        existing = {"session_id": session_id, "persona": persona, "events": []}

    existing["persona"] = persona
    existing.setdefault("events", []).append(payload)
    session_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
