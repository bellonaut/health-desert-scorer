"""User-safe error handling helpers for the Streamlit wrapper."""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "app_errors.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def safe_execute(operation_name: str = "Operation", reset_callback: Optional[Callable[[], None]] = None):
    """Decorator for safe error handling with optional reset action."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError as exc:
                logger.error("%s - File not found: %s", operation_name, exc)
                st.error(
                    """
                    Data not available.

                    The required data file could not be found. This may happen when the data build
                    has not been run yet or files were moved. Try refreshing or rebuilding data.
                    """
                )
                if st.button("Retry", key=f"retry_{operation_name}"):
                    st.rerun()
                return None
            except ValueError as exc:
                logger.error("%s - Invalid value: %s", operation_name, exc)
                st.error(
                    """
                    Invalid input.

                    The current filters produced invalid data. Try resetting filters or
                    switching to a different year/state.
                    """
                )
                if reset_callback is not None and st.button("Reset filters", key=f"reset_{operation_name}"):
                    reset_callback()
                    st.rerun()
                return None
            except Exception as exc:  # pragma: no cover - guardrail for unexpected errors
                logger.exception("%s - Unexpected error: %s", operation_name, exc)
                st.error(
                    """
                    Something went wrong.

                    An unexpected error occurred. This has been logged for investigation.
                    """
                )
                if st.button("Refresh", key=f"refresh_{operation_name}"):
                    st.rerun()
                return None

        return wrapper

    return decorator


def show_system_status(data_last_updated: str | None = None, model_version: list[str] | None = None) -> None:
    """Display system status in the Streamlit sidebar."""
    with st.sidebar:
        st.markdown("---")
        with st.expander("System status"):
            st.caption("Data status")
            st.write("Online" if data_last_updated else "Unknown")
            if data_last_updated:
                st.caption(f"Last data update: {data_last_updated}")
            st.caption("Model version")
            if model_version:
                st.write(", ".join(model_version))
            else:
                st.write("Unknown")
