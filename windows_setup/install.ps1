# ============================================================================
#  install.ps1  --  ONE-TIME setup, run by the developer on the user's laptop
# ============================================================================
#
#  Prep this laptop so the user only ever double-clicks a desktop icon.
#  Run this ONCE, from inside the cloned repo, in a PowerShell window:
#
#      powershell -ExecutionPolicy Bypass -File .\windows_setup\install.ps1
#
#  What it does:
#    1. Verifies git is installed (needed for auto-updates on every launch).
#    2. Installs uv if missing, provisions Python 3.13.
#    3. Runs `uv sync` (builds the environment; installs TA-Lib etc.).
#    4. Checks a .env exists  (YOU add it by hand with your API keys).
#    5. Creates a "Stock Price Predictor" desktop shortcut -> windows_setup\launch.bat
#
#  After this, hand the laptop back. The user just clicks the icon.
# ============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Repo root = parent of this script's folder (windows_setup\).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -Path $RepoRoot

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "   OK: $m" -ForegroundColor Green }
function Warn($m) { Write-Host "   WARN: $m" -ForegroundColor Yellow }

# ----------------------------------------------------------------------------
# 1. git (required - the launcher pulls updates every time)
# ----------------------------------------------------------------------------
Step "Checking for git..."
if (Get-Command git -ErrorAction SilentlyContinue) {
    Ok "git present ($((git --version)))."
} else {
    throw "git is not installed. Install Git for Windows (https://git-scm.com/download/win), then re-run this script."
}

# ----------------------------------------------------------------------------
# 2. uv (install if missing) + Python 3.13
# ----------------------------------------------------------------------------
Step "Checking for uv..."
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Ok "uv present ($(uv --version))."
} else {
    Warn "uv not found. Installing via the official Astral installer..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $uvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path (Join-Path $uvBin "uv.exe")) { $env:Path = "$uvBin;$env:Path" }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv installed but not on PATH yet. Close and reopen PowerShell, then re-run this script."
    }
    Ok "uv installed ($(uv --version))."
}

Step "Ensuring Python 3.13..."
uv python install 3.13
Ok "Python 3.13 ready."

# ----------------------------------------------------------------------------
# 3. Build the environment
# ----------------------------------------------------------------------------
Step "Installing dependencies (uv sync)... first run can take a few minutes."
try {
    uv sync
    Ok "Dependencies installed."
} catch {
    Warn "uv sync failed. On Windows this is almost always TA-Lib (a C library)."
    Write-Host "   Recent TA-Lib ships prebuilt Windows wheels, but if it failed,"
    Write-Host "   install Microsoft C++ Build Tools:"
    Write-Host "   https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Host "   then re-run this script. Full error is above."
    throw
}

# ----------------------------------------------------------------------------
# 4. .env check (developer adds this by hand)
# ----------------------------------------------------------------------------
Step "Checking .env..."
$envPath = Join-Path $RepoRoot ".env"
if (Test-Path $envPath) {
    Ok ".env present."
} else {
    Warn "No .env found in the repo root."
    Write-Host "   Create it (copy .env.example -> .env) and paste your GROQ_API_KEY"
    Write-Host "   and GEMINI_API_KEY. The app can't predict without it."
    Write-Host "   You can do this now, or before the first launch."
}

# ----------------------------------------------------------------------------
# 4b. Region check -- NSE geo-blocks foreign IPs.
#     Outside India, the jugaad + nse_bhavcopy price tiers hit NSE and fail
#     (403/timeout), wasting 1-2 minutes per fetch before falling back to
#     yfinance. Setting PRICE_CHAIN=yfinance skips the dead tiers so data
#     fetches take ~2s instead of minutes.
# ----------------------------------------------------------------------------
Step "Region check (price data source)..."
$outside = Read-Host "   Is this laptop OUTSIDE India (USA/EU/etc.)? NSE blocks foreign IPs. [y/N]"
if ($outside -match '^(y|Y|yes|YES)$') {
    if (Test-Path $envPath) {
        $lines = Get-Content $envPath
        if ($lines -match '^\s*PRICE_CHAIN\s*=') {
            $lines = $lines -replace '^\s*PRICE_CHAIN\s*=.*', 'PRICE_CHAIN=yfinance'
        } else {
            $lines += 'PRICE_CHAIN=yfinance'
        }
        Set-Content -Path $envPath -Value $lines
        Ok "Set PRICE_CHAIN=yfinance in .env (skips NSE tiers blocked outside India)."
    } else {
        Warn "No .env yet -- when you create it, add this line:  PRICE_CHAIN=yfinance"
    }
} else {
    Ok "Keeping the India default chain (jugaad -> nse_bhavcopy -> yfinance)."
}

# ----------------------------------------------------------------------------
# 5. Desktop shortcut -> launch.bat
# ----------------------------------------------------------------------------
Step "Creating desktop shortcut..."
$launcher = Join-Path $PSScriptRoot "launch.bat"
$desktop  = [Environment]::GetFolderPath("Desktop")
$lnkPath  = Join-Path $desktop "Stock Price Predictor.lnk"

$wsh = New-Object -ComObject WScript.Shell
$sc  = $wsh.CreateShortcut($lnkPath)
$sc.TargetPath       = $launcher
$sc.WorkingDirectory = $RepoRoot
$sc.WindowStyle      = 1
$sc.Description       = "Launch Stock Price Predictor (updates + opens in browser)"
$sc.Save()
Ok "Desktop shortcut created: 'Stock Price Predictor'."

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
Write-Host "`n== Setup complete." -ForegroundColor Green
Write-Host "   The user can now double-click the 'Stock Price Predictor' icon on the Desktop."
Write-Host "   It pulls the latest release, syncs, and opens the app in the browser."
Write-Host ""
Write-Host "   Reminder: this laptop follows the 'release' branch. Promote vetted"
Write-Host "   code with:  git checkout release; git merge main; git push origin release"
