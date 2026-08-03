<#
  probe_foundry_npu.ps1  --  ONE-OFF feasibility test (NOT part of launch).

  PURPOSE
  =======
  Answer three questions that ONLY your Snapdragon X laptop can answer, before
  we invest in wiring Foundry Local into the app:

    Gate 1  Does the Foundry catalog offer an NPU (QNN) build of a good-at-JSON
            model (Qwen / Phi / DeepSeek) for THIS machine?
    Gate 2  Is it actually faster than the current Ollama-CPU qwen3:8b tier?
    Gate 3  Does it produce VALID JSON (what the prediction pipeline needs)?

  It changes NOTHING in the app. It only installs Foundry Local (if missing),
  lists models, and runs one timed test prompt. Safe to delete afterwards.

  USAGE
  =====
    powershell -ExecutionPolicy Bypass -File .\windows_setup\probe_foundry_npu.ps1

    # or force a specific model id you saw in the NPU list:
    powershell -ExecutionPolicy Bypass -File .\windows_setup\probe_foundry_npu.ps1 -Model qwen2.5-7b-instruct-qnn-npu

  WHAT TO SEND BACK
  =================
    Copy the whole terminal output. The "NPU CATALOG" block + the "RESULT"
    block at the end are what I need to make the call.
#>

param(
  [string]$Model = "",
  [int]$MaxTokens = 256
)

$ErrorActionPreference = "Continue"
function Info($m) { Write-Host "[*] $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "[OK] $m" -ForegroundColor Green }
function Warn($m) { Write-Host "[!] $m" -ForegroundColor Yellow }
function Err($m)  { Write-Host "[X] $m" -ForegroundColor Red }

# ---- Capture EVERYTHING to one shareable file -------------------------------
# The whole console session is teed to diagnostics\foundry_probe_<ts>.txt so
# you can drag ONE file into chat instead of copy-pasting the terminal.
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Split-Path -Parent $scriptDir
$diagDir   = Join-Path $repoRoot "diagnostics"
if (-not (Test-Path $diagDir)) { New-Item -ItemType Directory -Path $diagDir | Out-Null }
$stamp     = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss") + "Z"
$OutFile   = Join-Path $diagDir "foundry_probe_$stamp.txt"
try { Start-Transcript -Path $OutFile -Force | Out-Null } catch { Warn "Transcript unavailable: $($_.Exception.Message)" }

function Finish($code) {
  try { Stop-Transcript | Out-Null } catch {}
  Write-Host ""
  Write-Host "Saved full report to:" -ForegroundColor Green
  Write-Host "    $OutFile" -ForegroundColor Green
  Write-Host "Drag THAT file into the chat with Thor. That's all I need." -ForegroundColor Green
  exit $code
}

Write-Host "==================================================================="
Write-Host " Foundry Local NPU feasibility probe  (Snapdragon X / Hexagon NPU)"
Write-Host "==================================================================="

# ---- 0. Hardware sanity ------------------------------------------------------
Info "Machine:"
(Get-CimInstance Win32_Processor).Name | ForEach-Object { Write-Host "    CPU : $_" }
"{0:N0} GB RAM" -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB) | ForEach-Object { Write-Host "    RAM : $_" }
Get-CimInstance Win32_VideoController | ForEach-Object { Write-Host "    GPU : $($_.Name)" }
Write-Host ""

# ---- 1. Install Foundry Local (winget) --------------------------------------
if (-not (Get-Command foundry -ErrorAction SilentlyContinue)) {
  Warn "foundry CLI not found. Installing Microsoft Foundry Local via winget..."
  winget install --id Microsoft.FoundryLocal --accept-source-agreements --accept-package-agreements
  # winget can fail behind the corp proxy; retry once with sysproxy.
  if (-not (Get-Command foundry -ErrorAction SilentlyContinue)) {
    Warn "First attempt didn't put 'foundry' on PATH. Trying with Walmart sysproxy..."
    $env:HTTP_PROXY  = "http://sysproxy.wal-mart.com:8080"
    $env:HTTPS_PROXY = "http://sysproxy.wal-mart.com:8080"
    winget install --id Microsoft.FoundryLocal --accept-source-agreements --accept-package-agreements
  }
  if (-not (Get-Command foundry -ErrorAction SilentlyContinue)) {
    Err "foundry still not on PATH. Close/reopen PowerShell and re-run, or install manually from https://learn.microsoft.com/azure/ai-foundry/foundry-local/get-started"
    Finish 1
  }
}
Ok ("foundry present: " + (foundry --version 2>&1 | Select-Object -First 1))
Write-Host ""

# ---- 2. First `model list` downloads the EPs for this hardware ---------------
Info "Listing models (first run downloads execution providers for your hardware)..."
foundry model list | Out-Host
Write-Host ""

# ---- 2b. NPU driver check (QNN needs Hexagon driver >= 30.0.140.0) -----------
Write-Host "----------------------- NPU / ACCELERATOR DRIVERS ------------------" -ForegroundColor Magenta
Get-CimInstance Win32_PnPSignedDriver 2>$null |
  Where-Object { $_.DeviceName -match "NPU|Hexagon|Neural|Adreno|Qualcomm" } |
  Select-Object DeviceName, DriverVersion, DriverDate |
  Format-Table -AutoSize | Out-Host
Write-Host "--------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host ""

# ---- 3. GATE 1: ALL hardware variants (NPU / GPU / CPU) for good models ------
# v0.10.2 has no --filter; --variants lists every hardware-specific build.
Write-Host "------------------ HARDWARE VARIANTS (--variants) ------------------" -ForegroundColor Magenta
$variants = foundry model list --variants 2>&1
$variants | Out-Host
Write-Host "--------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "[*] NPU/GPU-accelerated variants found (if any):" -ForegroundColor Cyan
$accel = $variants | Select-String -Pattern "npu|qnn|gpu|cuda|webgpu|dml|directml"
if ($accel) { $accel | Out-Host } else { Warn "None -- every variant is CPU on this box." }
Write-Host ""

# ---- 4. Pick a model to time -------------------------------------------------
# Default to a good-at-JSON 7-8B that IS in the catalog. Override with -Model.
if (-not $Model) { $Model = "qwen2.5-7b" }
Ok "Testing model: $Model  (override with -Model <id> from the variants list above)"
Write-Host ""

# ---- 5. Load model + find the OpenAI endpoint port ---------------------------
Info "Downloading + loading (one-time model download may be several GB)..."
foundry model download $Model | Out-Host
foundry model load $Model | Out-Host

$status = (foundry service status 2>&1) -join "`n"
$port = [regex]::Match($status, "127\.0\.0\.1:(\d{3,5})").Groups[1].Value
if (-not $port) { $port = [regex]::Match($status, ":(\d{3,5})").Groups[1].Value }
if (-not $port) { $port = "5273" }  # fallback; your run showed 127.0.0.1:58380
$endpoint = "http://127.0.0.1:$port/v1/chat/completions"
Info "Service status:"; Write-Host $status
Info "Using endpoint: $endpoint"
Write-Host ""

# ---- 6. GATE 2 + 3: timed JSON generation -----------------------------------
$body = @{
  model    = $Model
  messages = @(
    @{ role = "system"; content = "You are a JSON API. Reply with ONLY a JSON object, no prose." },
    @{ role = "user";   content = "Give a mock 1-day stock prediction for TCS as JSON with keys: direction (up/down), confidence (0-1), target_price (number), rationale (short string)." }
  )
  max_tokens = $MaxTokens
  stream = $false
} | ConvertTo-Json -Depth 6

Warn ">>> WATCH: open Task Manager -> Performance. Watch the NPU *and* GPU graphs"
Warn "    while this runs. Whichever spikes is where the model actually ran"
Warn "    (if only CPU moves, Foundry is CPU-only on this box -- same as Ollama)."
Write-Host ""
Info "Sending timed request..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
  $resp = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
} catch {
  Err "Request failed: $($_.Exception.Message)"
  Warn "If it's a connection error, run 'foundry service status' and re-run with the right port."
  Finish 1
}
$sw.Stop()

$text = $resp.choices[0].message.content
$ctoks = $resp.usage.completion_tokens
$secs  = [math]::Round($sw.Elapsed.TotalSeconds, 1)
$tps   = if ($ctoks -and $secs -gt 0) { [math]::Round($ctoks / $secs, 1) } else { "n/a" }

$validJson = $false
try { $null = ($text | ConvertFrom-Json); $validJson = $true } catch { $validJson = $false }

Write-Host ""
Write-Host "----------------------------- RESULT -------------------------------" -ForegroundColor Magenta
Write-Host "  Model            : $Model"
Write-Host "  Wall time        : $secs s"
Write-Host "  Completion tokens: $ctoks"
Write-Host "  Throughput       : $tps tok/s   (compare: Ollama-CPU qwen3:8b ~5-15)"
Write-Host "  Valid JSON       : $validJson"
Write-Host "  Ran on           : check which Task Manager graph spiked (NPU / GPU / CPU)"
Write-Host "--------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "Raw model output:" -ForegroundColor DarkGray
Write-Host $text
Write-Host ""

try { Stop-Transcript | Out-Null } catch {}
Write-Host ""
Ok "Saved full report to:"
Write-Host "    $OutFile" -ForegroundColor Green
Write-Host "Drag THAT file into the chat with Thor. That's all I need." -ForegroundColor Green