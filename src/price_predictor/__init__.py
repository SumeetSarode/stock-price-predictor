"""Public API root.

The CLI entry point is wired via [project.scripts] in pyproject.toml:
    price-predictor = "price_predictor:main"
"""
from price_predictor.cli.main import main

__all__ = ["main"]
