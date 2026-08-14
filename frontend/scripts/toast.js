/**
 * toast.js — bottom-right transient feedback messages.
 *
 * Listens for HTMX swaps and checks the new element for a
 * data-toast="message" attribute. If found, surfaces it as a toast
 * that auto-dismisses after ~3 seconds.
 *
 * Markup pattern (from the server):
 *   <button data-toast="Watchlist is full (10 max)">…</button>
 *
 * Why server-driven instead of client-side state: keeps the source
 * of truth in one place — the server decides when something is
 * worth toasting.
 */
(function () {
  "use strict";

  function showToast(message) {
    // Remove any existing toast first so they don't stack.
    document.querySelectorAll(".toast").forEach((t) => t.remove());

    const el = document.createElement("div");
    el.className = "toast";
    el.setAttribute("role", "status");
    el.setAttribute("aria-live", "polite");
    el.textContent = message;
    document.body.appendChild(el);

    // Self-remove after the CSS animation completes (~2.8s total).
    setTimeout(() => el.remove(), 3000);
  }

  document.body.addEventListener("htmx:afterSwap", function (e) {
    const target = e.detail.target;
    if (!target) return;
    const trigger = target.querySelector?.("[data-toast]") || target.closest?.("[data-toast]");
    if (trigger) {
      showToast(trigger.dataset.toast);
    }
  });

  /* ── Failure paths ──────────────────────────────────────────────
   *
   * WHY THIS EXISTS: afterSwap only fires on SUCCESS. htmx ignores
   * non-2xx responses by default (no swap, no message), while
   * hx-disabled-elt still re-enables the button when the request ends.
   *
   * So a failed "Run prediction" looked like: spinner spins for a
   * couple of minutes -> nothing changes -> button reappears -> the
   * user is told NOTHING. A silent failure is worse than a loud one:
   * it's indistinguishable from the app ignoring the click, so people
   * click again and queue up more doomed 60s requests.
   *
   * Three distinct events, because the user-actionable advice differs:
   *   responseError -> server answered with 4xx/5xx (it ran, it broke)
   *   sendError     -> never reached the server (server down / no net)
   *   timeout       -> exceeded htmx.config.timeout (see base.html)
   */

  function statusHint(status) {
    if (status === 429) return "Rate limited — every model is throttled. Wait a minute and retry.";
    if (status === 504 || status === 408) return "The prediction timed out. Try again.";
    if (status >= 500) return "The prediction failed on the server. Check the app logs for details.";
    if (status === 404) return "Not found — that ticker may not be supported.";
    if (status >= 400) return "That request was rejected (" + status + ").";
    return "Something went wrong (" + status + ").";
  }

  document.body.addEventListener("htmx:responseError", function (e) {
    const status = e.detail.xhr?.status ?? 0;
    // Prefer a server-supplied message; fall back to a status-based hint.
    // Guarded: an unhandled 500 can have a 0-byte body, and a stack-trace
    // page would be useless (and huge) in a toast.
    let msg = "";
    const body = (e.detail.xhr?.responseText || "").trim();
    if (body && body.length < 200 && !body.startsWith("<")) {
      msg = body;
    }
    showToast(msg || statusHint(status));
  });

  document.body.addEventListener("htmx:sendError", function () {
    showToast("Couldn't reach the server. Is the app still running?");
  });

  document.body.addEventListener("htmx:timeout", function () {
    showToast("The request timed out. The model may be slow or offline — try again.");
  });
})();
