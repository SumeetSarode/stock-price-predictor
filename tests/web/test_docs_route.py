"""Tests for the in-app 'How it works' walkthrough wiring.

The docs/ walkthrough (how_it_works.html + report/ chapters + assets)
is served read-only at /docs and linked from the global nav. These
tests lock in:
  - the mount serves the hub page and its chapters/assets,
  - relative links inside the hub resolve under /docs,
  - the nav renders a link to it,
  - a missing docs/ dir degrades gracefully (no crash, no mount).
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from price_predictor.web.app import create_app
from price_predictor.web.settings import settings


class TestDocsServed:
    def test_hub_page_is_served(self, client):
        resp = client.get("/docs/how_it_works.html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        # A stable phrase from the hero — proves it's the real doc.
        assert "How we make" in resp.text

    def test_report_chapter_is_served(self, client):
        # The hub links to report/all_in_one.html via a relative href;
        # that path must resolve under the /docs mount.
        resp = client.get("/docs/report/all_in_one.html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_report_asset_is_served(self, client):
        # Relative <link href="report/assets/styles.css"> in the hub.
        resp = client.get("/docs/report/assets/styles.css")
        assert resp.status_code == 200
        assert "css" in resp.headers["content-type"]

    def test_relative_links_in_hub_resolve_under_docs(self, client):
        # The hub uses relative hrefs (report/...) NOT absolute (/docs/...),
        # so it stays self-contained when opened as a file too. Served at
        # /docs/how_it_works.html, the browser resolves them under /docs/.
        # Verify both the href form and the resolved target exist.
        hub = client.get("/docs/how_it_works.html").text
        assert 'href="report/all_in_one.html"' in hub
        assert client.get("/docs/report/all_in_one.html").status_code == 200

    def test_api_keys_guide_is_served(self, client):
        resp = client.get("/docs/api_keys.html")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        # Both required keys must be named in the guide.
        assert "GEMINI_API_KEY" in resp.text
        assert "GROQ_API_KEY" in resp.text

    def test_hub_links_to_api_keys_guide(self, client):
        hub = client.get("/docs/how_it_works.html").text
        # Relative link so it resolves under /docs and works as a file too.
        assert 'href="api_keys.html"' in hub


class TestNavLink:
    def test_home_nav_links_to_walkthrough(self, client):
        home = client.get("/").text
        assert "/docs/how_it_works.html" in home
        assert "How it works" in home

    def test_nav_link_opens_new_tab_accessibly(self, client):
        # WCAG: a link that opens a new tab must announce that to AT users.
        home = client.get("/").text
        assert 'target="_blank"' in home
        assert 'rel="noopener"' in home
        assert "opens in a new tab" in home


class TestGracefulWhenDocsMissing:
    def test_missing_docs_dir_does_not_crash_app(self, monkeypatch, tmp_path):
        # Point docs_dir at a non-existent path. The app must still build
        # (docs are optional) and the /docs mount simply won't exist.
        missing = tmp_path / "no_docs_here"
        monkeypatch.setattr(
            type(settings), "docs_dir",
            property(lambda self: missing),
        )
        assert not missing.exists()

        app = create_app()
        with TestClient(app) as c:
            # App is alive...
            assert c.get("/").status_code == 200
            # ...but the walkthrough isn't mounted → 404, not a 500/crash.
            assert c.get("/docs/how_it_works.html").status_code == 404
