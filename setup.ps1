# ============================================================================
# setup.ps1 - One-command Windows setup for price-predictor
# ============================================================================
#
# WHAT THIS DOES
#   1. Ensures uv (the Python package manager) is installed.
#   2. Installs Python 3.13 via uv if needed.
#   3. Runs `uv sync` to build the venv from the committed uv.lock.
#   4. Scaffolds a .env from .env.example on first run.
#   5. Checks your API keys are filled in, then offers to launch the web app.
#
# HOW TO RUN (from a PowerShell window, in the repo root):
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# Re-runnable and idempotent - safe to run again any time.
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Work from this script's own directory (the repo root) regardless of cwd.
Set-Location -Path $PSScriptRoot

function Write-Step($msg)  { Write-Host "`n== $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "   OK: $msg" -ForegroundColor Green }
function Write-Warn2($msg) { Write-Host "   WARN: $msg" -ForegroundColor Yellow }

# ----------------------------------------------------------------------------
# 1. Ensure uv is installed
# ----------------------------------------------------------------------------
Write-Step "Checking for uv..."
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Ok "uv already installed ($(uv --version))."
} else {
    Write-Warn2 "uv not found. Installing via the official installer..."
    # Official Astral installer (public, no corporate mirror).
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    # The installer adds uv to PATH for new shells; add it to THIS session too.
    $uvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path (Join-Path $uvBin "uv.exe")) {
        $env:Path = "$uvBin;$env:Path"
    }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installed but not on PATH. Close and reopen PowerShell, then re-run this script."
    }
    Write-Ok "uv installed ($(uv --version))."
}

# ----------------------------------------------------------------------------
# 2. Ensure Python 3.13 is available (uv manages it - no manual install)
# ----------------------------------------------------------------------------
Write-Step "Ensuring Python 3.13 is available..."
uv python install 3.13
Write-Ok "Python 3.13 ready."

# ----------------------------------------------------------------------------
# 3. Build the environment from the committed lockfile
# ----------------------------------------------------------------------------
Write-Step "Installing dependencies (uv sync)..."
try {
    uv sync
    Write-Ok "Dependencies installed."
} catch {
    Write-Warn2 "uv sync failed. The most common cause on Windows is TA-Lib."
    Write-Host "   TA-Lib is a C library. Recent versions ship prebuilt Windows"
    Write-Host "   wheels, but if resolution failed, install the Microsoft C++"
    Write-Host "   Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/)"
    Write-Host "   and re-run this script. Full error is above."
    throw
}

# ----------------------------------------------------------------------------
# 4. Scaffold .env on first run
# ----------------------------------------------------------------------------
Write-Step "Checking .env..."
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Ok "Created .env from .env.example."
} else {
    Write-Ok ".env already exists (left untouched)."
}

# ----------------------------------------------------------------------------
# 5. Verify API keys are filled in
# ----------------------------------------------------------------------------
Write-Step "Checking API keys in .env..."
$envText = Get-Content ".env" -Raw
$placeholders = @()
if ($envText -match "your_groq_key_here")   { $placeholders += "GROQ_API_KEY" }
if ($envText -match "your_gemini_key_here") { $placeholders += "GEMINI_API_KEY" }

if ($placeholders.Count -gt 0) {
    Write-Warn2 "These keys still have placeholder values: $($placeholders -join ', ')"
    Write-Host ""
    Write-Host "   Get free keys here, then paste them into .env:" -ForegroundColor Yellow
    Write-Host "     Groq   : https://console.groq.com/keys"
    Write-Host "     Gemini : https://aistudio.google.com/app/apikey"
    Write-Host ""
    Write-Host "   After editing .env, run:  uv run price-predictor-web" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Setup finished, but the app needs real API keys before it can predict."
    exit 0
}
Write-Ok "API keys look set."

# ----------------------------------------------------------------------------
# 6. Offer to launch
# ----------------------------------------------------------------------------
Write-Step "All set."
$answer = Read-Host "Launch the web app now? (y/n)"
if ($answer -eq "y" -or $answer -eq "Y") {
    Write-Host "Starting... your browser should open shortly. Ctrl+C to stop." -ForegroundColor Cyan
    uv run price-predictor-web
} else {
    Write-Host "When ready, run:  uv run price-predictor-web" -ForegroundColor Cyan
}
