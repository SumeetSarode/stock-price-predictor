"""
Loguru setup for the price predictor app.

Call setup_logging() ONCE at app startup (e.g., in cli.py main()).
After that, import `logger` from loguru anywhere and use it.
"""
import sys

from loguru import logger

from config.settings import settings


def setup_logging() -> None:
    """Configure loguru with three handlers: console, full log, errors-only log."""
    # Drop loguru's default handler
    logger.remove()

    # Console handler — colored, level from settings
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Full log file — DEBUG+, rotates at 10 MB, keeps last 5
    logger.add(
        settings.logs_dir / "predictor.log",
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{extra} | "
            "{message}"
        ),
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )

    # Errors-only file — ERROR+, rotates at 10 MB, keeps last 10
    logger.add(
        settings.logs_dir / "errors.log",
        level="ERROR",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{extra} | "
            "{message}\n{exception}"
        ),
        rotation="10 MB",
        retention=10,
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        f"Logging initialized | level={settings.log_level} | dir={settings.logs_dir}"
    )


def get_stock_logger(ticker: str):
    """Return a logger pre-bound with stock context.

    Usage:
        log = get_stock_logger("RELIANCE")
        log.info("Fetching prices")  # → log line tagged with stock=RELIANCE
    """
    return logger.bind(stock=ticker)


if __name__ == "__main__":
    setup_logging()

    logger.debug("Debug — file only (not console at INFO level)")
    logger.info("Info — should appear")
    logger.warning("Warning — should appear in yellow")
    logger.error("Error — should appear in red, also in errors.log")

    # Test stock binding
    reliance_log = get_stock_logger("RELIANCE")
    reliance_log.info("This log line is tagged with stock=RELIANCE")
