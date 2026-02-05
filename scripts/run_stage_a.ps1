<# Stage A baseline runner for Windows #>
$ErrorActionPreference = "Stop"

Write-Host "Tip: activate your virtual env first, e.g. .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Yellow

function Run-Step {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host "== $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$Label failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python not found on PATH."
    exit 1
}

Run-Step "Create mock DHS clusters" { python scripts/create_mock_dhs.py }
Run-Step "Download open data" { python scripts/download_open_data.py }
Run-Step "Build LGA features (Stage A)" { python -m src.data.build_features }

Write-Host "Stage A pipeline completed." -ForegroundColor Green
