# Windows setup — deploy Stock Price Predictor to a non-technical user's laptop

This folder makes the app run on someone else's Windows laptop as a
**free, self-updating, click-to-open** tool. They only ever double-click a
desktop icon. You (the developer) do a one-time setup.

## How it works

- The repo is **cloned once** onto their laptop.
- A desktop icon (`launch.bat`) runs every time they open the app. It:
  1. pulls the latest **vetted** code from the `release` branch,
  2. syncs dependencies (instant when nothing changed),
  3. starts the app — their browser opens automatically.
- **Updates are "on next open":** whatever you've promoted to `release`
  is what they get the next time they click the icon.
- **History persists for free.** Predictions and the watchlist live in
  SQLite at `C:\Users\<them>\.price_predictor\app.db` — in their home
  folder, *outside the repo*. `git pull` never touches it, so a prediction
  made today is still in History next week.

---

## One-time setup (you, on their laptop)

1. **Install Git for Windows** (if not already):
   https://git-scm.com/download/win

2. **Clone the repo** somewhere permanent, e.g. their Documents:
   ```powershell
   cd $HOME\Documents
   git clone <public-repo-url> price_predictor
   cd price_predictor
   ```

3. **Create the release branch** (once, from your machine or theirs) so the
   laptop has a vetted branch to follow:
   ```powershell
   git checkout -b release
   git push -u origin release
   git checkout main
   ```

4. **Add the `.env`** with your API keys (you do this by hand):
   ```powershell
   copy .env.example .env
   notepad .env    # paste GROQ_API_KEY and GEMINI_API_KEY, save
   ```
   Free keys: Groq https://console.groq.com/keys  ·
   Gemini https://aistudio.google.com/app/apikey

5. **Run the installer** (installs uv + Python 3.13, builds the env,
   creates the desktop shortcut):
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\windows_setup\install.ps1
   ```

6. **Hand it back.** They double-click **"Stock Price Predictor"** on the Desktop.
   Done — they never see a terminal again.

---

## Your update workflow (afterwards)

Develop on `main` freely. When something is solid and tested, **promote it
to `release`** — that's the only thing that reaches their laptop:

```powershell
git checkout release
git merge main
git push origin release
git checkout main
```

Next time they open the app, they get it.

> **Prefer raw latest instead?** Edit `windows_setup\launch.bat` and set
> `BRANCH=main`. Not recommended — a broken push would break their app with
> no safety net.

---

## Gotchas worth knowing

- **The black console window IS the running server.** Keep it open
  using the app; close it (or Ctrl+C) to stop. That's the on/off switch.
- **Laptop must be awake + app open** for the nightly grading pass to run
  (it grades predictions in the background while the app is running).
- **Offline is handled.** If the laptop can't reach the internet at launch,
  the update step is skipped and the last-known version starts anyway.
- **Database schema changes are the one real risk.** The app auto-creates
  tables but does *not* auto-migrate. If you ever change the shape of a
  table, old local data could break. Be deliberate with schema changes on
  `release`.
- **TA-Lib** is the only tricky dependency on Windows. Modern wheels make
  it painless, but if `uv sync` ever fails on it, install the Microsoft
  C++ Build Tools and re-run `install.ps1` (the script says this too).

---

## Files here

| File | Who runs it | When |
|------|-------------|------|
| `install.ps1` | You (developer) | Once, during setup |
| `launch.bat`  | The user (via desktop icon) | Every time they open the app |
| `README.md`   | You | Reference |
