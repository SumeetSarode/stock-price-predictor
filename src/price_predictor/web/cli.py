"""CLI entry point — `price-predictor-web` command.

Starts uvicorn against the FastAPI app and, by default, pops the
user's browser open to the local URL. Both behaviors are configurable
via env vars (see WebSettings).

Wired via pyproject.toml::

    [project.scripts]
    price-predictor-web = "price_predictor.web.cli:main"
"""
from __future__ import annotations

import threading
import time
import webbrowser

import uvicorn
from loguru import logger

from price_predictor.web.settings import settings


def _open_browser_when_ready(url: str, delay_seconds: float = 1.2) -> None:
    """Open the URL in the default browser after a short delay.

    Runs in a daemon thread so the main uvicorn loop isn't blocked.
    The delay gives uvicorn time to actually bind the port — opening
    too early gets a 'site unreachable' page.

    Failure here is non-fatal: if the OS can't find a browser, we just
    log and let the user navigate manually.
    """
    def _do_open() -> None:
        time.sleep(delay_seconds)
        try:
            webbrowser.open_new_tab(url)
        except Exception as exc:  # pragma: no cover — OS-dependent
            logger.warning("Could not auto-open browser: {}", exc)

    threading.Thread(target=_do_open, daemon=True).start()


def main() -> None:
    """Boot the web app.

    Pulls host / port / browser-auto-open from WebSettings (env-driven).
    Prints a friendly banner so the user knows what just happened.
    """
    url = f"http://{settings.host}:{settings.port}"

    print()
    print("  📈  Stock Price Predictor")
    print(f"      Local-first · v0.1.0-dev · {url}")
    print("      Press Ctrl+C to stop")
    print()

    if settings.auto_open_browser:
        _open_browser_when_ready(url)

    uvicorn.run(
        "price_predictor.web.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
