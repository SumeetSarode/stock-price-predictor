# Stock Price Predictor — Windows Setup Guide

Step-by-step instructions to install **Stock Price Predictor** on a
non-technical person's Windows laptop, so that afterward they only ever
**double-click a desktop icon** to use it.

- **Who does this:** you (the developer / the technical person). Do it once.
- **How long:** about 20-40 minutes (most of it is the first `uv sync`).
- **What they get:** a "Stock Price Predictor" icon on their Desktop. Clicking
  it pulls your latest vetted code, then opens the app in their browser.
- **Cost:** free. It runs entirely on their laptop — no hosting, no server.

> Prefer the short version? The quick overview lives in
> [`README.md`](README.md). This file is the detailed walkthrough.

---

## Before you start — what you'll need

| Thing | Why | Where |
|-------|-----|-------|
| The laptop, ~40 min | You're setting it up in person | — |
| Internet connection | Downloads git, uv, Python, dependencies | — |
| A Groq API key (free) | Primary LLM for predictions | https://console.groq.com/keys |
| A Gemini API key (free) | Fallback LLM | https://aistudio.google.com/app/apikey |
| The repo URL (public) | To clone the code | your GitHub repo |

Grab the two API keys first (steps in **Phase 4** below) so you're not
juggling browser tabs mid-install.

---

## Phase 1 — Install Git

Git is what lets the app auto-update itself later.

1. Go to **https://git-scm.com/download/win** — the download starts automatically.
2. Run the installer. **Click "Next" through every screen** (all the defaults
   are fine). Click "Install", then "Finish".
3. Verify it worked: press **Start**, type `powershell`, open **Windows PowerShell**,
   and run:
   ```powershell
   git --version
   ```
   You should see something like `git version 2.xx.x`. If you get an error,
   close PowerShell, reopen it, and try again (the installer needs a fresh window).

---

## Phase 2 — Clone the repository

We'll put the app in the user's Documents folder.

1. In the same PowerShell window, run these one at a time:
   ```powershell
   cd $HOME\Documents
   git clone <PASTE-YOUR-PUBLIC-REPO-URL-HERE> price_predictor
   cd price_predictor
   ```
   Replace `<PASTE-YOUR-PUBLIC-REPO-URL-HERE>` with your actual repo URL
   (e.g. `https://github.com/you/stock-price-predictor.git`).

2. Confirm you're in the right place:
   ```powershell
   ls
   ```
   You should see files like `pyproject.toml`, `README.md`, and a
   `windows_setup` folder.

---

## Phase 3 — Create the `release` branch

The laptop follows a branch called **`release`** — this is your safety net.
You develop on `main` (where things can be half-finished), and only code you
**promote to `release`** ever reaches this laptop. So a broken commit on `main`
can never break their app.

Run this **once**:

```powershell
git checkout -b release
git push -u origin release
git checkout main
```

That creates `release`, publishes it, and switches you back to `main`.

> **Don't want a safety net?** You can skip this phase and later set
> `BRANCH=main` in `windows_setup\launch.bat`. Not recommended — if you ever
> push a broken commit to `main`, their next launch breaks with no warning.

---

## Phase 4 — Add your API keys (the `.env` file)

The app needs LLM keys to make predictions. You add them here, by hand.
They live only on this laptop and are **never** committed to git.

### 4a. Get a free Groq key
1. Go to **https://console.groq.com/keys** and sign in (Google login works).
2. Click **Create API Key**, give it any name, and **copy the key**
   (it starts with `gsk_...`). You won't be able to see it again, so copy it now.

### 4b. Get a free Gemini key
1. Go to **https://aistudio.google.com/app/apikey** and sign in.
2. Click **Create API key** and **copy it**.

### 4c. Create the `.env` file
In PowerShell (still inside the `price_predictor` folder):

```powershell
copy .env.example .env
notepad .env
```

Notepad opens. Find these two lines and replace the placeholder text with
your real keys:

```
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
```

becomes (example):

```
GROQ_API_KEY=gsk_abc123realkeyhere
GEMINI_API_KEY=AIzaSyRealKeyHere
```

**Save** (Ctrl+S) and **close** Notepad. Leave everything else in the file
as-is.

---

## Phase 5 — Run the installer

This installs `uv` (the Python manager), Python 3.13, all dependencies, and
creates the desktop shortcut. Run it once:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows_setup\install.ps1
```

Watch the output. It prints an `OK:` line for each step. The longest step is
"Installing dependencies (uv sync)" — the first time it can take **a few
minutes**. Let it finish.

When it's done you'll see:
```
== Setup complete.
   The user can now double-click the 'Stock Price Predictor' icon on the Desktop.
```

> If it fails on TA-Lib, see **Troubleshooting** at the bottom.

---

## Phase 6 — Test it yourself (before handing over)

1. Go to the Desktop. You should see a **Stock Price Predictor** icon.
2. **Double-click it.** A black window opens (that's the app running — normal),
   and after a moment your browser opens to the app at
   `http://127.0.0.1:8000`.
   - Among the startup lines you'll see **"Refreshing the stock search
     list from NSE..."** — that's the app pulling the full list of ~2,300+
     NSE-listed stocks so search is always current. It takes a few seconds,
     and if the laptop is offline it's skipped harmlessly (see the FAQ).
3. Search for a stock (e.g. `RELIANCE`), generate a prediction, and confirm it
   appears.
4. Click **History** — your prediction should be listed.
5. **Close the black window** to stop the app. Reopen it via the icon and check
   **History** again — your prediction should **still be there**. (It's saved
   permanently in their user folder; see the FAQ.)

If all that works, you're done.

---

## Offline fallback with Ollama (mostly automatic)

By default the app uses free hosted AI (Gemini + Groq). Those have daily
limits; when they run out, predictions would normally pause until the quota
resets. To keep the app working **even when the free quotas are exhausted**,
it falls back to a local AI model that runs on the laptop itself — no
internet, no quota.

**The launcher sets this up for you.** Every time the desktop icon is
double-clicked, it automatically:

1. Checks whether Ollama is installed — and if not, tries to install it
   silently via `winget`.
2. Starts the Ollama server if it isn't already running.
3. Pulls the configured model (`qwen3:8b`) if it isn't downloaded yet
   (a one-time ~5 GB download; you'll see progress in the black window).

After the first successful launch, the fallback is ready and later launches
just confirm it in a second. Every step is **non-fatal** — if Ollama can't be
set up (no winget, offline, etc.), the app still starts on the hosted AI.

> **Hardware note:** the local model runs on the CPU and needs ~5 GB free
> disk + ~8 GB free RAM. On a 16 GB laptop it works but is slower than the
> hosted models — which is fine, because it only kicks in as a last resort.

> **If auto-install doesn't work** (some machines lack `winget`): install
> Ollama by hand from https://ollama.com/download, then just relaunch — the
> launcher pulls the model on its own from there.

> **To turn it off:** open `.env` in Notepad and delete the
> `,ollama_chat/qwen3:8b` at the end of the `CHAIN_AGENTIC` line. Save. The
> launcher then skips all Ollama setup.

---

## Phase 7 — Hand it over

Tell the non-technical user just three things:

1. **To open it:** double-click the **Stock Price Predictor** icon on the Desktop.
   The browser opens automatically.
2. **To close it:** close the black window (or click its X).
3. That's it. It updates itself every time they open it.

They never need PowerShell, git, or anything technical again.

---

## Shipping updates later (you, from your own machine)

Develop and test on `main` as usual. When something is ready, promote it to
`release` — that's the only thing that reaches their laptop:

```powershell
git checkout release
git merge main
git push origin release
git checkout main
```

The next time they double-click the icon, they get your update. No action
needed on their end.

### New `.env` settings ship automatically (keys stay safe)

When you update `.env.example` and promote to `release`, the launcher syncs
the change into the user's existing `.env` on their next launch:

- **New keys are added.** Any key they don't have yet (e.g. a brand-new
  `OLLAMA_API_BASE`) is appended with its documented default.
- **App-managed keys are updated.** The model chains — `CHAIN_AGENTIC` and
  `PAID_AGENTIC` — are "owned" by the app: if you change their value in
  `.env.example` (e.g. add an Ollama fallback tail), the user's line is
  updated to match. **This is how model/chain changes reach every laptop.**
- **Their API keys and personal tweaks are NEVER touched.** Secrets,
  `PRICE_CHAIN=yfinance`, custom ports — all left byte-for-byte as-is.
- **Idempotent.** Once synced, later launches do nothing.

So to push a model/chain change to every deployed laptop: edit `CHAIN_AGENTIC`
in `.env.example`, promote to `release`, done. Their next launch picks it up.

> **Want another key to auto-update too?** Add it to the `MANAGED_KEYS` set
> in `scripts/sync_env.py`. Keep that list tiny and NEVER add secrets or
> user/geo-specific settings — those must stay user-owned.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `git --version` says "not recognized" | Close and reopen PowerShell. If still failing, reinstall Git for Windows and pick "Add to PATH" during install. |
| `install.ps1` fails on **TA-Lib** | TA-Lib is a C library. Install the free **Microsoft C++ Build Tools** (https://visualstudio.microsoft.com/visual-cpp-build-tools/ — during install, check "Desktop development with C++"), then re-run `install.ps1`. |
| "uv installed but not on PATH" | Close PowerShell, open a new one, and re-run `install.ps1`. |
| Icon opens the black window but **no browser** | Manually open a browser and go to `http://127.0.0.1:8000`. |
| "Port 8000 already in use" | Another program is using that port. Edit `.env`, add a line `WEB_PORT=8010`, save, relaunch. |
| Predictions say "no API key" / fail instantly | The `.env` keys are wrong or missing. Re-check Phase 4 (no quotes, no spaces around `=`). |
| App won't update | Confirm the laptop has internet, and that you actually pushed to the `release` branch (Phase 3 + "Shipping updates"). |
| Running `launch.bat` shows "cannot check for updates" | It's just offline — the app still starts with the last version it has. |
| "Stock list refresh skipped/failed - using the built-in list" | Harmless. The laptop couldn't reach NSE (offline, or a temporary NSE hiccup), so search falls back to the ~2,300-stock list that ships with the app. It'll refresh on the next launch that has a connection. |
| **News headlines won't load** / "couldn't reach the news service" | News comes from GDELT (a free external service). Run `uv run python scripts\check_news.py` in the app folder for a plain-English diagnosis. Usually it's a corporate network/VPN blocking GDELT — set `HTTPS_PROXY` in `.env` and relaunch. **Predictions still work without news** — news only enriches them, it never blocks a prediction. |

---

## FAQ

**Which stocks can I search?**
Every equity listed on the NSE — roughly 2,300+ names, not just the big ones.
The list refreshes automatically from NSE's official records **on every
launch** (you'll see a "Refreshing the stock search list" line in the black
window), so newly listed stocks and renames show up without any manual step.
If the laptop is offline or NSE is briefly unreachable, the refresh is skipped
and search uses the copy that shipped with the app — nothing breaks, and it
tries again next launch. (Note: this refresh downloads a *static list file*
from NSE's archive, which works worldwide including outside India — unlike
NSE's live price feed, which is India-only and is why out-of-India laptops set
`PRICE_CHAIN=yfinance`.)

**Where is my prediction history stored?**
In a small database at `C:\Users\<their-username>\.price_predictor\app.db`.
It lives in the user folder, **outside** the app code, so updates never erase
it. Predictions and starred stocks persist across launches and updates.

**Is anything sent to the internet?**
Only what's needed to work: stock/news data is fetched from public sources,
and prediction reasoning is generated via the Groq/Gemini APIs using your keys.
The app itself runs locally on the laptop — there is no public website and no
one else can access it.

**Does the laptop need to stay on?**
Only while they're using it. Open the icon when they want predictions; close
the window when done. (The app also grades past predictions in the background,
but only while it's open.)

**Can I move the app folder later?**
Yes, but you'd need to recreate the desktop shortcut (just re-run
`install.ps1` from the new location). The history in the user folder is
unaffected.

**How do I completely uninstall it?**
Delete the `price_predictor` folder, delete the desktop shortcut, and (if you
want history gone too) delete the `C:\Users\<username>\.price_predictor` folder.
