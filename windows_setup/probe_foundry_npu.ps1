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
    exit 1
  }
}
Ok ("foundry present: " + (foundry --version 2>&1 | Select-Object -First 1))
Write-Host ""

# ---- 2. First `model list` downloads the EPs for this hardware ---------------
Info "Listing models (first run downloads the QNN execution provider for your NPU -- can take a minute)..."
foundry model list | Out-Host
Write-Host ""

# ---- 3. GATE 1: which models have an NPU build on THIS box -------------------
Write-Host "------------------------- NPU CATALOG ------------------------------" -ForegroundColor Magenta
$npu = foundry model list --filter device=npu 2>&1
$npu | Out-Host
Write-Host "--------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host ""

# ---- 4. Pick a model to time --------------------------------------------------
if (-not $Model) {
  # Prefer a good-at-JSON family from the NPU list: qwen -> phi -> deepseek.
  foreach ($pat in @("qwen","phi","deepseek")) {
    $hit = $npu | Select-String -Pattern $pat | Select-Object -First 1
    if ($hit) {
      # crude token grab: first whitespace-delimited word that contains the pattern
      $Model = ($hit.ToString() -split "\s+" | Where-Object { $_ -match $pat } | Select-Object -First 1)
      if ($Model) { break }
    }
  }
}
if (-not $Model) {
  Err "No Qwen/Phi/DeepSeek NPU build detected in the catalog for this machine."
  Warn "Gate 1 = FAIL. Send me the NPU CATALOG block above; we'll pick another route."
  exit 0
}
Ok "Testing model: $Model"
Write-Host ""

# ---- 5. Load model + find the OpenAI endpoint port ---------------------------
Info "Downloading + loading (one-time model download may be several GB)..."
foundry model download $Model | Out-Host
foundry model load $Model | Out-Host

$status = (foundry service status 2>&1) -join "`n"
$port = [regex]::Match($status, ":(\d{3,5})").Groups[1].Value
if (-not $port) { $port = "5273" }  # common Foundry default; adjust if wrong
$endpoint = "http://localhost:$port/v1/chat/completions"
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

Warn ">>> WATCH: open Task Manager -> Performance -> NPU while this runs. If the"
Warn "    NPU graph spikes, the model is truly on the Hexagon (not CPU fallback)."
Write-Host ""
Info "Sending timed request..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
  $resp = Invoke-RestMethod -Uri $endpoint -Method Post -ContentType "application/json" -Body $body -TimeoutSec 300
} catch {
  Err "Request failed: $($_.Exception.Message)"
  Warn "If it's a connection error, run 'foundry service status' and re-run with the right port."
  exit 1
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
Write-Host "  NPU used?        : check the Task Manager NPU graph you watched"
Write-Host "--------------------------------------------------------------------" -ForegroundColor Magenta
Write-Host ""
Write-Host "Raw model output:" -ForegroundColor DarkGray
Write-Host $text
Write-Host ""
Ok "Done. Copy this whole output back to Thor."
