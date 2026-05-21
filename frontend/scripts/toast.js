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
})();
