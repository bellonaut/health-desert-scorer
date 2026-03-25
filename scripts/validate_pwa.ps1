Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Wait-Url {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = $null

    while ((Get-Date) -lt $deadline) {
        try {
            return Invoke-RestMethod -Uri $Url -TimeoutSec 10 -UseBasicParsing
        } catch {
            $lastError = $_
            Start-Sleep -Seconds 1
        }
    }

    throw "Timed out waiting for $Label at $Url. Last error: $lastError"
}

function Wait-Http {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Label,
        [int]$TimeoutSeconds = 120
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = $null

    while ((Get-Date) -lt $deadline) {
        try {
            return Invoke-WebRequest -Uri $Url -TimeoutSec 10 -UseBasicParsing
        } catch {
            $lastError = $_
            Start-Sleep -Seconds 1
        }
    }

    throw "Timed out waiting for $Label at $Url. Last error: $lastError"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$appDir = Join-Path $repoRoot 'app'
$tmpDir = Join-Path $repoRoot 'tmp'
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null

$python = (Get-Command python).Source
$apiOutLog = Join-Path $tmpDir 'validate_api.out.log'
$apiErrLog = Join-Path $tmpDir 'validate_api.err.log'
$streamlitOutLog = Join-Path $tmpDir 'validate_streamlit.out.log'
$streamlitErrLog = Join-Path $tmpDir 'validate_streamlit.err.log'

$apiProc = $null
$streamlitProc = $null

try {
    Write-Step "Starting API server on 127.0.0.1:8601"
    $apiProc = Start-Process -FilePath $python `
        -ArgumentList '-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', '8601' `
        -WorkingDirectory $appDir `
        -RedirectStandardOutput $apiOutLog `
        -RedirectStandardError $apiErrLog `
        -PassThru

    Write-Step "Starting Streamlit on 127.0.0.1:8501"
    $streamlitProc = Start-Process -FilePath $python `
        -ArgumentList '-m', 'streamlit', 'run', 'app/app.py', '--server.headless', 'true', '--server.port', '8501' `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $streamlitOutLog `
        -RedirectStandardError $streamlitErrLog `
        -PassThru

    Write-Step "Validation 1: API payload shape"
    $apiPayload = Wait-Url -Url 'http://127.0.0.1:8601/api/data?year=2024&focus=All+risk' -Label 'API'
    Write-Host ("status: 200")
    Write-Host ("top keys: " + (($apiPayload.PSObject.Properties.Name | Sort-Object) -join ', '))
    Write-Host ("map keys: " + (($apiPayload.map.PSObject.Properties.Name | Sort-Object) -join ', '))
    Write-Host ("lgas: " + $apiPayload.lgas.Count)

    Write-Step "Validation 2: Compare API payload against embedded __INITIAL_DATA__ snapshot"
    & $python (Join-Path $repoRoot 'scripts/build_embedded_html.py') | Write-Host

    $comparisonOutput = @'
import json
import re
from pathlib import Path

root = Path(r"__REPO_ROOT__")
html = (root / "build" / "embedded_ui.html").read_text(encoding="utf-8")
match = re.search(r'window\.__INITIAL_DATA__\s*=\s*(\{.*?\});</script>', html, re.S)
if not match:
    raise SystemExit("Could not extract __INITIAL_DATA__ from build/embedded_ui.html")
embedded_payload = json.loads(match.group(1))

import requests
api_payload = requests.get("http://127.0.0.1:8601/api/data?year=2024&focus=All+risk", timeout=30).json()

print("embedded top keys:", ", ".join(sorted(embedded_payload.keys())))
print("embedded map keys:", ", ".join(sorted(embedded_payload["map"].keys())))
print("shape matches:", sorted(api_payload.keys()) == sorted(embedded_payload.keys()))
print("map shape matches:", sorted(api_payload["map"].keys()) == sorted(embedded_payload["map"].keys()))
print("embedded lgas:", len(embedded_payload["lgas"]))
'@ -replace '__REPO_ROOT__', ($repoRoot -replace '\\', '\\')
    $comparisonOutput | & $python -

    Write-Step "Validation 3: Standalone HTML and asset wiring"
    $standaloneHtml = Wait-Http -Url 'http://127.0.0.1:8601/health_desert_ui.html?year=2024&focus=All+risk' -Label 'Standalone HTML'
    $manifest = Wait-Url -Url 'http://127.0.0.1:8601/static/manifest.json' -Label 'Manifest'
    $sw = Wait-Http -Url 'http://127.0.0.1:8601/static/sw.js' -Label 'Service worker'

    Write-Host ("standalone html status: " + [int]$standaloneHtml.StatusCode)
    Write-Host ("manifest linked: " + $standaloneHtml.Content.Contains('rel="manifest"'))
    Write-Host ("service worker registration present: " + $standaloneHtml.Content.Contains('serviceWorker.register'))
    Write-Host ("manifest name: " + $manifest.name)
    Write-Host ("manifest start_url: " + $manifest.start_url)
    Write-Host ("sw has cacheFirst: " + $sw.Content.Contains('cacheFirst'))
    Write-Host ("sw has staleWhileRevalidate: " + $sw.Content.Contains('staleWhileRevalidate'))

    Write-Step "Validation 4: Manual browser-only checks still required"
    Write-Host "Open http://127.0.0.1:8601/health_desert_ui.html?year=2024&focus=All%20risk and confirm the map renders."
    Write-Host "Run Lighthouse PWA against http://127.0.0.1:8601/?pwa=1 from Chrome DevTools or npx lighthouse."
    Write-Host "This script validates the server-side and static wiring only."

    Write-Step "Logs"
    Write-Host "API stdout log: $apiOutLog"
    Write-Host "API stderr log: $apiErrLog"
    Write-Host "Streamlit stdout log: $streamlitOutLog"
    Write-Host "Streamlit stderr log: $streamlitErrLog"
} finally {
    foreach ($proc in @($apiProc, $streamlitProc)) {
        if ($null -ne $proc) {
            try {
                if (-not $proc.HasExited) {
                    Stop-Process -Id $proc.Id -Force
                }
            } catch {
                # Ignore cleanup races.
            }
        }
    }
}
