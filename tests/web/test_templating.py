"""Unit tests for the cache-busting asset() template helper."""
from __future__ import annotations

import re

from price_predictor.web.templating import asset, templates


def test_asset_appends_version_for_real_file():
    # components.css definitely exists in the static dir.
    url = asset("styles/components.css")
    assert url.startswith("/static/styles/components.css?v=")
    version = url.split("?v=")[1]
    assert version.isdigit()


def test_asset_strips_leading_slash():
    assert asset("/styles/base.css").startswith("/static/styles/base.css?v=")


def test_asset_missing_file_no_version():
    url = asset("styles/does-not-exist-xyz.css")
    assert url == "/static/styles/does-not-exist-xyz.css"
    assert "?v=" not in url


def test_asset_version_changes_with_mtime(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace

    import price_predictor.web.templating as mod

    f = tmp_path / "thing.css"
    f.write_text("a{}")
    # Swap the whole settings reference (static_dir may be a read-only
    # property, so patching the attribute directly won't work).
    monkeypatch.setattr(mod, "settings", SimpleNamespace(static_dir=tmp_path))

    v1 = mod.asset("thing.css").split("?v=")[1]
    assert re.fullmatch(r"\d+", v1)
    # Bump mtime forward and confirm the version tracks it.
    os.utime(f, (2_000_000_000, 2_000_000_000))
    v2 = mod.asset("thing.css").split("?v=")[1]
    assert v2 == "2000000000"


def test_asset_registered_as_jinja_global():
    assert templates.env.globals.get("asset") is asset
