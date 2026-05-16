/* Shared navigation — injected on every page so it stays DRY.
   Each page calls `injectNav(currentPath)` after DOMContentLoaded.
   Falls back gracefully if JS is disabled (the static <nav> in HTML
   still works for screen readers and printers). */

const NAV_LINKS = [
  { href: "index.html",         label: "Home" },
  { href: "data_sources.html",  label: "Data we ingest" },
  { href: "indicators.html",    label: "Technical indicators" },
  { href: "crossovers.html",    label: "Crossovers (Golden, Death, MACD)" },
  { href: "candlesticks.html",  label: "Candlestick patterns (61)" },
  { href: "chart_patterns.html",label: "Chart patterns" },
  { href: "news.html",          label: "News analysis" },
  { href: "synthesis.html",     label: "Putting it together" },
  { href: "horizons.html",      label: "Horizons & targets" },
  { href: "guardrails.html",    label: "Quality controls" },
  { href: "example.html",       label: "Worked example" },
  { href: "glossary.html",      label: "Glossary" },
];

function injectNav(currentHref) {
  const host = document.getElementById("site-nav");
  if (!host) return;
  const items = NAV_LINKS.map(l => {
    const active = l.href === currentHref;
    const cls = active
      ? "block px-3 py-2 rounded text-sm font-semibold bg-blue-50 text-[#0053e2] border-l-4 border-[#0053e2]"
      : "block px-3 py-2 rounded text-sm text-gray-700 hover:bg-gray-100 border-l-4 border-transparent";
    return `<a href="${l.href}" class="${cls}">${l.label}</a>`;
  }).join("");
  host.innerHTML = `
    <div class="mb-4 pb-3 border-b border-gray-200">
      <a href="index.html" class="block">
        <div class="text-xs uppercase tracking-wider text-gray-500">Walkthrough</div>
        <div class="text-base font-bold text-[#0053e2]">How we predict</div>
      </a>
    </div>
    <nav aria-label="Primary">${items}</nav>
    <div class="mt-6 pt-4 border-t border-gray-200 text-xs text-gray-500">
      <p>Self-contained report. Open any file directly.</p>
      <p class="mt-2">Audience: domain expert (no programming required).</p>
    </div>
  `;
}
