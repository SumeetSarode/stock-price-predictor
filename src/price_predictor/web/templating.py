"""Shared Jinja2Templates instance.

Both `routes/pages.py` and `routes/api.py` render templates; keeping a
single configured instance here is DRY and gives us one place to register
Jinja globals (like the cache-busting ``asset()`` helper below).
"""
from __future__ import annotations

from fastapi.templating import Jinja2Templates

from price_predictor.web.settings import settings

templates = Jinja2Templates(directory=str(settings.templates_dir))


def asset(path: str) -> str:
    """Return a ``/static/...`` URL with a cache-busting ``?v=<mtime>`` suffix.

    The version is the file's last-modified time (whole seconds), so the
    query string changes ONLY when the file actually changes. Browsers then
    re-fetch exactly when they must and cache aggressively otherwise — no
    more "hard-refresh to see my CSS change" papercuts.

    Falls back to a bare URL if the file can't be stat'd (e.g. missing).
    """
    rel = path.lstrip("/")
    full = settings.static_dir / rel
    try:
        version = int(full.stat().st_mtime)
    except OSError:
        return f"/static/{rel}"
    return f"/static/{rel}?v={version}"


templates.env.globals["asset"] = asset
