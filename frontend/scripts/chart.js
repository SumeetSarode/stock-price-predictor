/**
 * chart.js — render the 90-day price chart on the stock detail page.
 *
 * Uses Chart.js (vendored at /static/vendor/chart-4.4.6.umd.min.js).
 * Loads only on pages that include the .price-chart container so the
 * 200KB library cost is paid only when needed.
 *
 * Auto-discovery: finds any element matching `.price-chart` on the
 * page, reads its data-ticker attribute, fetches /api/chart, and
 * renders. If the container also has data-entry-low / data-target /
 * data-stop attributes (from a cached prediction), those are drawn
 * as horizontal dashed reference lines.
 *
 * Container markup (in the Jinja template):
 *   <div class="price-chart" data-ticker="RELIANCE.NS"
 *        data-entry-low="1352" data-entry-high="1368"
 *        data-target="1410" data-stop="1325" data-direction="bullish">
 *     <canvas id="price-chart-canvas"></canvas>
 *   </div>
 */
(function () {
  "use strict";

  // Wait until DOM + Chart.js are both ready before booting.
  function boot() {
    const container = document.querySelector(".price-chart");
    if (!container) return;  // not on a chart page
    if (typeof Chart === "undefined") {
      console.warn("chart.js: Chart global not found; check vendor script load order");
      return;
    }

    const ticker = container.dataset.ticker;
    if (!ticker) {
      console.warn("chart.js: .price-chart missing data-ticker");
      return;
    }

    fetchAndRender(container, ticker);
  }

  async function fetchAndRender(container, ticker) {
    const canvas = container.querySelector("canvas");
    if (!canvas) return;

    let payload;
    try {
      const resp = await fetch(`/api/chart?ticker=${encodeURIComponent(ticker)}&days=90`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      payload = await resp.json();
    } catch (e) {
      console.warn("chart.js: fetch failed", e);
      renderEmpty(container, "Chart unavailable");
      return;
    }

    if (payload.is_empty || !payload.closes?.length) {
      renderEmpty(container, "No price history available");
      return;
    }

    renderChart(canvas, container, payload);
  }

  function renderEmpty(container, message) {
    container.innerHTML = `<div class="price-chart__empty">${message}</div>`;
  }

  function renderChart(canvas, container, payload) {
    const ds = container.dataset;
    // Build optional level annotations from data-* attributes.
    const annotations = [];
    if (ds.entryLow && ds.entryHigh) {
      annotations.push(level(parseFloat(ds.entryLow),  "Entry low",  "#6b7280", "dashed"));
      annotations.push(level(parseFloat(ds.entryHigh), "Entry high", "#6b7280", "dashed"));
    }
    if (ds.target) {
      const color = ds.direction === "bearish" ? "#dc2626" : "#16a34a";
      annotations.push(level(parseFloat(ds.target), "Target", color, "solid"));
    }
    if (ds.stop) {
      const color = ds.direction === "bearish" ? "#16a34a" : "#dc2626";
      annotations.push(level(parseFloat(ds.stop), "Stop", color, "solid"));
    }

    // Compute y-axis bounds that include both price history AND levels
    // so dashed lines aren't clipped off-canvas.
    const allValues = payload.closes.slice();
    annotations.forEach((a) => allValues.push(a.value));
    const yMin = Math.min(...allValues) * 0.98;
    const yMax = Math.max(...allValues) * 1.02;

    new Chart(canvas, {
      type: "line",
      data: {
        labels: payload.dates,
        datasets: [{
          label: "Close",
          data: payload.closes,
          borderColor: "#4f46e5",            // indigo-600 (primary)
          backgroundColor: "rgba(79,70,229,0.08)",
          borderWidth: 2,
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 4,
          fill: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          x: {
            ticks: {
              maxTicksLimit: 8,
              color: "#6b7280",
              font: { size: 11 },
            },
            grid: { display: false },
          },
          y: {
            min: yMin,
            max: yMax,
            ticks: {
              color: "#6b7280",
              font: { size: 11 },
              callback: (v) => "₹" + v.toFixed(0),
            },
            grid: { color: "rgba(0,0,0,0.04)" },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(17,24,39,0.95)",
            padding: 10,
            displayColors: false,
            callbacks: {
              label: (ctx) => "₹" + ctx.parsed.y.toFixed(2),
            },
          },
          // Custom plugin: draw horizontal level lines.
          // Registered inline so we don't need chartjs-plugin-annotation
          // (saves ~30KB).
        },
      },
      plugins: [{
        id: "level-lines",
        afterDraw: (chart) => {
          if (!annotations.length) return;
          const { ctx, chartArea, scales } = chart;
          const yScale = scales.y;
          ctx.save();
          annotations.forEach((a) => {
            const y = yScale.getPixelForValue(a.value);
            if (y < chartArea.top || y > chartArea.bottom) return;
            ctx.beginPath();
            if (a.style === "dashed") ctx.setLineDash([5, 4]);
            else ctx.setLineDash([]);
            ctx.strokeStyle = a.color;
            ctx.lineWidth = 1.5;
            ctx.moveTo(chartArea.left, y);
            ctx.lineTo(chartArea.right, y);
            ctx.stroke();

            // Label on the right edge.
            ctx.setLineDash([]);
            ctx.font = "600 10px ui-sans-serif, system-ui, sans-serif";
            ctx.fillStyle = a.color;
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            ctx.fillText(`${a.label} ₹${a.value.toFixed(0)}`,
                         chartArea.right - 4, y - 2);
          });
          ctx.restore();
        },
      }],
    });
  }

  function level(value, label, color, style) {
    return { value, label, color, style };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
