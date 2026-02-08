#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

echo "==> Nigeria Health Desert Risk Scorer demo"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required but was not found on PATH."
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "==> Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

echo "==> Activating virtual environment"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Installing dependencies"
python -m pip install --upgrade pip
python -m pip install -r "${PROJECT_ROOT}/requirements.txt"

echo "==> Creating mock DHS data"
python "${PROJECT_ROOT}/scripts/create_mock_dhs.py"

echo "==> Downloading open datasets"
python "${PROJECT_ROOT}/scripts/download_open_data.py"

echo "==> Building features"
python -m src.data.build_features

echo "==> Training models"
python -m src.models.train_models

echo "==> Launching Streamlit app"
echo "Open http://localhost:8501 in your browser."
streamlit run "${PROJECT_ROOT}/app/app.py"
