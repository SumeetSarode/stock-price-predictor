/**
 * dashboard.js — interactivity for the Nifty 50 dashboard table.
 *
 * Two concerns, both via event delegation on document.body so the
 * handlers work even after HTMX swaps fresh table HTML in.
 *
 *   1. Row click → navigate to /stock/<ticker>
 *      (ticker comes from data-ticker on the <tr>)
 *
 *   2. Sortable columns: click a <th data-sort="key"> to toggle
 *      asc/desc on that column. Sort keys read from data-sort-<key>
 *      attributes on each <tr>. Numeric vs string detected per-column.
 *
 * No external deps. ~80 lines of vanilla JS.
 */
(function () {
  "use strict";

  /* ── Row click → navigate ─────────────────────────────────────── */

  document.addEventListener("click", function (e) {
    // Ignore clicks inside the header (sort handles those).
    if (e.target.closest("thead")) return;

    const row = e.target.closest(".data-table__row");
    if (!row) return;
    if (row.classList.contains("data-table__row--skeleton")) return;

    const ticker = row.dataset.ticker;
    if (!ticker) return;
    window.location.href = `/stock/${encodeURIComponent(ticker)}`;
  });

  /* ── Sortable columns ─────────────────────────────────────────── */

  document.addEventListener("click", function (e) {
    const th = e.target.closest("th[data-sort]");
    if (!th) return;
    const table = th.closest("table[data-sortable]");
    if (!table) return;

    const key = th.dataset.sort;
    // Toggle: clicking the same col flips direction. Clicking a new
    // col uses the column's data-sort-default (or "asc" if unspecified).
    const currentDir = th.dataset.sortActive;
    let newDir;
    if (currentDir === "asc") newDir = "desc";
    else if (currentDir === "desc") newDir = "asc";
    else newDir = th.dataset.sortDefault || "asc";

    sortTable(table, key, newDir);
  });

  function sortTable(table, key, dir) {
    const tbody = table.querySelector("tbody");
    if (!tbody) return;
    const rows = Array.from(tbody.querySelectorAll("tr"));

    // Detect numeric vs string by sampling the first row.
    const sampleVal = rows[0]?.dataset[`sort${capitalize(key)}`] ?? "";
    const isNumeric = !isNaN(parseFloat(sampleVal)) && isFinite(sampleVal);

    rows.sort((a, b) => {
      const av = a.dataset[`sort${capitalize(key)}`] ?? "";
      const bv = b.dataset[`sort${capitalize(key)}`] ?? "";
      let cmp;
      if (isNumeric) {
        cmp = parseFloat(av) - parseFloat(bv);
      } else {
        cmp = av.localeCompare(bv);
      }
      return dir === "asc" ? cmp : -cmp;
    });

    // Re-attach in new order. Modern browsers handle this efficiently.
    const frag = document.createDocumentFragment();
    rows.forEach((r) => frag.appendChild(r));
    tbody.appendChild(frag);

    // Update header indicators: clear all, set the active one.
    table.querySelectorAll("th[data-sort-active]").forEach((th) => {
      delete th.dataset.sortActive;
    });
    table.querySelector(`th[data-sort="${key}"]`).dataset.sortActive = dir;
  }

  function capitalize(s) {
    // data-sort-foo_bar → dataset.sortFoo_bar  (dashes become camelCase
    // but underscores stay). Our keys avoid dashes for this reason —
    // 'change_pct' not 'change-pct'.
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
})();
