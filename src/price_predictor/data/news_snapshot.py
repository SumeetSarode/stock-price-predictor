"""NewsSnapshot - disk-backed point-in-time GDELT cache for backtest replay.

WHY THIS EXISTS
===============
The whole reason `predict(as_of=...)` is interesting is *honest replay*:
re-running a backtest months later should produce the exact same
prediction. GDELT's archive is mostly stable, but two things spoil that
guarantee if we re-query live every time:

  1. GDELT can index articles retroactively. An article *published* on
     2024-01-10 might not appear in our query results until GDELT
     *sees* it on 2024-02-01. A backtest run today for as_of=2024-01-15
     would include that article; a real predictor on 2024-01-15
     wouldn't have. The post-filter `published_at <= as_of` closes
     most of this leak, but only if the article was indexed in time
     to appear in the response at all.

  2. GDELT's response shape can drift over months. Even for the
     "same" query at the "same" date range, the article ordering and
     truncation behaviour is not bit-stable.

So we snapshot once. Subsequent backtest runs read from disk. Identical
inputs => identical outputs. The whole calibration pipeline becomes
reproducible.

DESIGN
======
Mirrors PredictionStore's "one file per (logical key)" approach because
it solves the same problems (atomic writes, easy archival, no DB).

  Layout: {root}/{as_of YYYY-MM-DD}/{key}.json
  Key   : sha256(lang|lookback_days|query)[:16]
  Body  : {"query": str, "lang": str, "as_of": iso, "lookback_days": int,
           "fetched_at": iso, "articles": [...]}

The `query` is also stored verbatim inside the JSON so a human
grepping the cache can find what was fetched without needing to
recompute the hash.

LIVE MODE
=========
Live `predict()` does NOT use this store. Live calls hit GDELT
directly (the whole point is to see the latest news as it happens).
The snapshot is strictly for backtest reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from price_predictor.data.news import NewsFetchError, fetch_news
from price_predictor.data.news_providers import GoogleNewsRssProvider
from price_predictor.data.news_providers import NewsFetchError as RssFetchError

# Hash truncation length. 16 hex chars = 64 bits = collision probability
# is negligible at our scale (< 1M cached queries per as_of). Filenames
# stay short and human-scannable.
_KEY_LEN = 16

# Strip anything not [a-z0-9-] from the lang code (defence-in-depth
# against weird LLM-supplied values that shouldn't reach the FS).
_UNSAFE_LANG_CHARS = re.compile(r"[^a-z0-9\-]")


class NewsSnapshotError(Exception):
    """Raised when snapshot persistence operations fail unrecoverably.

    Examples: corrupted JSON on disk, unwriteable root, schema drift
    on load. Distinct from NewsFetchError so callers can distinguish
    "GDELT call failed" from "our cache is broken".
    """


def _safe_lang(lang: str) -> str:
    """Sanitise a lang code for filename use.

    GDELT uses 3-letter ISO codes ('eng', 'spa', ...). Anything else
    is suspicious; we strip it rather than reject so a typo doesn't
    crash the whole pipeline.
    """
    cleaned = _UNSAFE_LANG_CHARS.sub("", lang.lower())
    return cleaned or "unk"


def _hash_key(
    query: str, lang: str, lookback_days: int, *, exact_phrase: bool = True,
) -> str:
    """Deterministic short key for a (query, lang, lookback) tuple.

    The triple uniquely identifies a logical fetch. Two backtest runs
    asking for the same logical thing on the same as_of must hit the
    same file -- otherwise we re-fetch needlessly and lose
    reproducibility.

    `exact_phrase` distinguishes an exact-phrase company fetch from a
    loose-token sector fetch of the *same* string. It's only mixed into
    the key when False, so existing exact-phrase snapshots keep their
    original hashes (no cache invalidation).
    """
    suffix = "" if exact_phrase else "|loose"
    payload = f"{lang}|{lookback_days}|{query}{suffix}".encode()
    return hashlib.sha256(payload).hexdigest()[:_KEY_LEN]


class NewsSnapshot:
    """Filesystem-backed snapshot store for GDELT news fetches.

    Not thread-safe for concurrent writes of the SAME (query, as_of,
    lookback) triple -- last-writer-wins. In practice harmless: if two
    parallel backtest calls race on the same key, both compute
    identical content (deterministic GDELT response post-filtered to
    the same as_of), so whichever lands wins and the other is wasted
    bandwidth, not corruption.
    """

    def __init__(self, root: Path | str):
        """Args:
            root: Directory under which snapshots are stored. Created
                if missing. Must be writable.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.debug(f"NewsSnapshot rooted at {self.root}")

    # ─────────────────────────────────────────────────────────
    # Path computation
    # ─────────────────────────────────────────────────────────
    def _day_dir(self, as_of: date) -> Path:
        return self.root / as_of.isoformat()

    def path_for(
        self,
        query: str,
        as_of: date,
        lookback_days: int,
        *,
        lang: str = "eng",
        exact_phrase: bool = True,
    ) -> Path:
        """Where this (query, as_of, lookback, lang) tuple lives on disk.

        Useful for tests and for callers that want to check existence
        before triggering a fetch.
        """
        safe_lang = _safe_lang(lang)
        key = _hash_key(query, safe_lang, lookback_days, exact_phrase=exact_phrase)
        return self._day_dir(as_of) / f"{key}_{safe_lang}_{lookback_days}d.json"

    # ─────────────────────────────────────────────────────────
    # Read / write
    # ─────────────────────────────────────────────────────────
    def _save(
        self,
        path: Path,
        *,
        query: str,
        lang: str,
        as_of: date,
        lookback_days: int,
        df: pd.DataFrame,
    ) -> None:
        """Atomic write of one snapshot.

        Stores the full article rows as a list of dicts. tz-aware
        timestamps are serialised as ISO strings so JSON round-trips
        cleanly through pandas; loaders convert them back to UTC.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        articles: list[dict] = []
        for _, row in df.iterrows():
            article = {k: (v.isoformat() if hasattr(v, "isoformat") else v)
                       for k, v in row.to_dict().items()}
            articles.append(article)

        payload = {
            "query": query,
            "lang": lang,
            "as_of": as_of.isoformat(),
            "lookback_days": lookback_days,
            "fetched_at": datetime.now(UTC).isoformat(),
            "article_count": len(articles),
            "articles": articles,
        }

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(payload, tmp, indent=2, sort_keys=True)
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, path)
        except OSError as e:
            raise NewsSnapshotError(
                f"Failed to write snapshot to {path}: {e}"
            ) from e
        logger.debug(f"saved news snapshot: {path} ({len(articles)} articles)")

    def _load(self, path: Path) -> pd.DataFrame:
        """Read a snapshot back as a DataFrame matching fetch_news()'s shape.

        Reconstructs the published_at column as tz-aware UTC timestamps
        so downstream code (post-filters, agent prompt builders) sees
        the exact same dtype it would from a live fetch.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise NewsSnapshotError(
                f"Cannot load snapshot {path}: {e}"
            ) from e

        articles = payload.get("articles", [])
        if not articles:
            return pd.DataFrame(
                columns=["title", "url", "published_at", "source", "language"]
            )

        df = pd.DataFrame(articles)
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        return df

    # ─────────────────────────────────────────────────────────
    # The one-shot public API
    # ─────────────────────────────────────────────────────────
    async def _rss_fallback(
        self,
        query: str,
        start: date,
        end: date,
        as_of: date,
        *,
        lang: str,
        exact_phrase: bool,
        gdelt_err: NewsFetchError,
    ) -> pd.DataFrame:
        """Live-only fallback when GDELT fails. Re-raises gdelt_err unless a
        fresh, non-empty RSS result is available.

        Look-ahead guard: RSS returns RECENT news only, so it is used ONLY
        when the window's end is within the freshness horizon. For an older
        (backtest) window we re-raise the GDELT error rather than fabricate
        'current' news for a past date.

        Empty RSS also re-raises: GDELT ERRORED (not 'found nothing'), so we
        must not cache an empty result as truth -- that would permanently
        neuter this (query, as_of) after a transient throttle.
        """
        from price_predictor.config.settings import settings

        if not settings.news_rss_fallback_enabled:
            raise gdelt_err

        provider = GoogleNewsRssProvider(
            freshness_days=settings.news_rss_freshness_days
        )
        end_dt = pd.Timestamp(as_of, tz="UTC").to_pydatetime()
        if not provider.coverage.can_serve_end(end_dt):
            logger.warning(
                f"news RSS fallback SKIPPED (window ends {as_of}, older than "
                f"{settings.news_rss_freshness_days}d horizon -- backtest, not "
                f"live); re-raising GDELT error for query={query!r}"
            )
            raise gdelt_err

        logger.info(
            f"news RSS fallback: GDELT failed ({gdelt_err}); trying "
            f"{provider.name} for query={query!r} as_of={as_of}"
        )
        try:
            df = await provider.fetch(
                query, start.isoformat(), end.isoformat(),
                lang=lang, exact_phrase=exact_phrase,
            )
        except (ValueError, RssFetchError) as rss_err:
            logger.warning(
                f"news RSS fallback ALSO failed ({rss_err}); re-raising "
                f"original GDELT error for query={query!r}"
            )
            raise gdelt_err from rss_err

        if df.empty:
            logger.warning(
                f"news RSS fallback returned no articles for query={query!r}; "
                f"re-raising GDELT error (won't cache empty as truth)"
            )
            raise gdelt_err

        logger.info(
            f"news RSS fallback SUCCEEDED: {len(df)} article(s) from "
            f"{provider.name} for query={query!r}"
        )
        return df

    async def get_or_fetch(
        self,
        query: str,
        as_of: date,
        lookback_days: int,        *,
        lang: str = "eng",
        exact_phrase: bool = True,
    ) -> pd.DataFrame:
        """Return articles for (query, as_of, lookback). Cache-or-fetch.

        Cache hit  : load from disk, return immediately.
        Cache miss : call fetch_news() with end=as_of, post-filter to
                     drop any rows with published_at > as_of (defence
                     against GDELT's seendate-vs-publishdate skew),
                     persist, return.

        Args:
            query: Free-text GDELT query (e.g., "Reliance Industries").
            as_of: Trading date the snapshot is anchored to. The window
                   queried is [as_of - lookback_days, as_of].
            lookback_days: How many days of news to pull back from as_of.
            lang: ISO 3-letter language code, default "eng".

        Returns:
            DataFrame with the same shape as fetch_news() returns.
            May be empty if the original query genuinely matched
            nothing -- empty is a valid cached result, NOT a cache miss.

        Raises:
            NewsSnapshotError: cache file is corrupted or unwriteable.
            NewsFetchError:    cache miss AND the live fetch failed.
                               Caller decides whether to degrade.
        """
        path = self.path_for(query, as_of, lookback_days, lang=lang,
                              exact_phrase=exact_phrase)
        if path.exists():
            logger.debug(f"news snapshot HIT: {path}")
            return self._load(path)

        logger.info(
            f"news snapshot MISS: query={query!r} as_of={as_of} "
            f"lookback={lookback_days}d lang={lang}; fetching from GDELT"
        )
        end = as_of
        start = as_of - pd.Timedelta(days=lookback_days).to_pytimedelta()

        try:
            df = await fetch_news(
                query, start.isoformat(), end.isoformat(), lang=lang,
                exact_phrase=exact_phrase,
            )
        except ValueError:
            # Caller's bad input -- no fallback would help.
            raise
        except NewsFetchError as gdelt_err:
            # GDELT failed (typically a 429 burst). Try the live RSS fallback,
            # but ONLY for recent windows -- a backtest must never be served
            # 'current' news for a past date (look-ahead). If the fallback is
            # disabled, out-of-horizon, or also fails/empty, re-raise the
            # original GDELT error so the caller decides whether to degrade.
            df = await self._rss_fallback(
                query, start, end, as_of, lang=lang,
                exact_phrase=exact_phrase, gdelt_err=gdelt_err,
            )

        # Belt-and-braces: drop any article published AFTER as_of.
        # GDELT filters by seendate, not publishdate, so a late-indexed
        # article can slip through the [start..end] window. This
        # post-filter is the second line of defence.
        #
        # Defensive parse: upstream fetch_news may return published_at
        # as either Timestamps OR strings depending on the code path,
        # so normalize to tz-aware UTC datetimes BEFORE comparing.
        # This also gives _save() a clean dtype to serialise from.
        if not df.empty and "published_at" in df.columns:
            df = df.copy()
            df["published_at"] = pd.to_datetime(
                df["published_at"], utc=True, errors="coerce",
            )
            cutoff = pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1)
            # NaT comparisons return False, so unparseable rows get
            # dropped along with future-dated ones — acceptable: an
            # article we can't date-stamp can't be safely included in
            # a point-in-time view.
            mask = df["published_at"] < cutoff
            dropped = (~mask).sum()
            if dropped:
                logger.warning(
                    f"news snapshot: dropped {dropped} article(s) with "
                    f"published_at > as_of={as_of} or unparseable date "
                    f"(GDELT seendate skew)"
                )
            df = df[mask].reset_index(drop=True)

        self._save(
            path,
            query=query, lang=lang, as_of=as_of,
            lookback_days=lookback_days, df=df,
        )
        return df


# ─────────────────────────────────────────────────────────────
# Module-level singleton (matches the _shared_cache pattern)
# ─────────────────────────────────────────────────────────────
# Why a singleton: the news_impact agent's tools are leaf functions --
# they have no convenient place to receive a NewsSnapshot instance.
# A module-level get/set lets the predictor wire up the store before
# invoking the agent without threading it through every call.
_singleton: NewsSnapshot | None = None


def get_news_snapshot() -> NewsSnapshot | None:
    """Return the configured singleton, or None if none is set.

    Tools consult this; if it's None, they MUST fall back to live
    fetch (live mode behavior).
    """
    return _singleton


def set_news_snapshot(snapshot: NewsSnapshot | None) -> None:
    """Install (or clear) the singleton. Tests reset to None for hermetic isolation."""
    global _singleton
    _singleton = snapshot
