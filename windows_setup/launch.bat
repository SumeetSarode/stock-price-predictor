@echo off
setlocal

REM ====================================================================
REM  Stock Price Predictor - LAUNCHER  (the everyday clickable file)
REM ====================================================================
REM  Double-click this (or the "Stock Price Predictor" desktop shortcut) to:
REM    1. Pull the latest vetted code from the RELEASE branch.
REM    2. Sync dependencies (instant when nothing changed).
REM    3. Start the app - your browser opens automatically.
REM
REM  Keep the black window open while you use the app.
REM  Close it (or press Ctrl+C) to stop.
REM ====================================================================

title Stock Price Predictor

REM -- Which branch to follow. 'release' = only versions the developer
REM    has vetted reach this laptop. Change to "main" for raw latest
REM    (not recommended - a broken push would break the app here).
set BRANCH=release

REM -- Make sure uv is findable even in a fresh shell (installer puts it here).
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

REM -- Jump to the repo root (this file lives in windows_setup\).
cd /d "%~dp0.."

echo.
echo   Stock Price Predictor
echo   =====================
echo.

REM -- 1. Update code. Graceful: if offline, just run what we already have.
where git >nul 2>nul
if errorlevel 1 (
    echo   [skip] git not found - cannot check for updates. Starting current version.
) else (
    echo   Checking for updates on "%BRANCH%"...
    git fetch origin %BRANCH% >nul 2>nul
    if errorlevel 1 (
        echo   [offline?] Could not reach the server - starting the version you have.
    ) else (
        REM  reset --hard makes local code an exact mirror of the release branch.
        REM  It only touches tracked files - your .env and your saved history
        REM  (stored in your home folder) are never affected.
        git reset --hard origin/%BRANCH% >nul 2>nul
        if errorlevel 1 (
            echo   [warn] Update step hiccuped - starting the version you have.
        ) else (
            echo   Up to date.
        )
    )
)

REM -- 2. Sync dependencies. Cheap no-op when nothing changed.
echo   Preparing... (the very first run can take a few minutes)
uv sync >nul 2>nul
if errorlevel 1 (
    echo   [warn] Dependency sync had an issue - trying to start anyway.
)

REM -- 2b. Refresh the full NSE stock search index (runs every launch).
REM    We download NSE's official listed-equity master (EQUITY_L.csv). This
REM    is a STATIC file on NSE's archives CDN - reachable worldwide, incl.
REM    outside India (unlike NSE's dynamic price APIs, which do geo-block).
REM    Safety: tight timeout (~5s to fail if truly blocked/offline), the
REM    builder writes NOTHING on failure and rejects a suspiciously small
REM    response, and the git reset above already restored the shipped index
REM    - so a bad/offline fetch can never leave you with a broken search
REM    box. Fully non-fatal.
echo   Refreshing the stock search list from NSE... ^(a few seconds; skipped if offline^)
uv run python scripts\build_search_index.py --fetch-nse
if errorlevel 1 (
    echo   [info] Stock list refresh skipped/failed - using the built-in list.
)

REM -- 3. Launch. Scheduler ON so predictions auto-grade while the app is open,
REM    which keeps your history meaningful over time.
set WEB_ENABLE_SCHEDULER=true

echo.
echo   Starting... your browser will open in a moment.
echo   (Keep this window open while you use the app. Close it to stop.)
echo.

uv run price-predictor-web

echo.
echo   App stopped. You can close this window.
pause
endlocal
