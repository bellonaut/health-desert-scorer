from __future__ import annotations

import socket
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

_server: HTTPServer | None = None
_serve_dir: Path | None = None
_port: int = 8600
_write_lock = threading.Lock()


def _find_free_port(start: int = 8600) -> int:
    for port in range(start, start + 100):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found")


def start_file_server(serve_dir: Path) -> int:
    global _server, _serve_dir, _port

    if _server is not None:
        return _port

    _serve_dir = serve_dir
    _port = _find_free_port()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(serve_dir), **kwargs)

        def log_message(self, *args) -> None:
            pass

    _server = HTTPServer(("127.0.0.1", _port), Handler)
    thread = threading.Thread(target=_server.serve_forever, daemon=True)
    thread.start()
    return _port


def write_and_get_url(html: str, filename: str = "app.html") -> str:
    """Write rendered HTML to the serve dir and return its localhost URL."""
    if _serve_dir is None:
        raise RuntimeError("Call start_file_server first")

    path = _serve_dir / filename
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with _write_lock:
        tmp_path.write_text(html, encoding="utf-8")
        tmp_path.replace(path)
    return f"http://127.0.0.1:{_port}/{filename}"
