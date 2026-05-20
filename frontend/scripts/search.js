/**
 * search.js — keyboard nav + click delegation + outside-click-to-close
 * for the sticky-nav ticker search bar.
 *
 * Design notes:
 *   - HTMX owns the network round-trip. This file owns ONLY
 *     keyboard / mouse interactions with the rendered dropdown.
 *   - Uses event delegation on the root element so it works correctly
 *     even after HTMX swaps in new suggestion HTML.
 *   - No external deps. Just vanilla JS.
 *
 * Wired automatically on DOMContentLoaded.
 */
(function () {
  "use strict";

  function init() {
    const root = document.querySelector("[data-search-root]");
    if (!root) return;

    const input = root.querySelector(".search__input");
    const dropdown = root.querySelector(".search__dropdown");
    if (!input || !dropdown) return;

    let activeIndex = -1;

    /** Returns all currently-rendered suggestion buttons. */
    function getButtons() {
      return Array.from(dropdown.querySelectorAll(".search__btn"));
    }

    /** Highlight the button at `index`; null clears highlight. */
    function setActive(index) {
      const btns = getButtons();
      btns.forEach((b, i) => {
        b.classList.toggle("is-active", i === index);
        if (i === index) b.scrollIntoView({ block: "nearest" });
      });
      activeIndex = index;
    }

    /** Navigate to the ticker's detail page. */
    function pickTicker(ticker) {
      if (!ticker) return;
      window.location.href = `/stock/${encodeURIComponent(ticker)}`;
    }

    /** Close dropdown + clear input + reset state. */
    function closeDropdown() {
      dropdown.innerHTML = "";
      activeIndex = -1;
    }

    // Keyboard nav on the input.
    input.addEventListener("keydown", function (e) {
      const btns = getButtons();
      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (btns.length === 0) return;
        setActive((activeIndex + 1) % btns.length);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        if (btns.length === 0) return;
        setActive(activeIndex <= 0 ? btns.length - 1 : activeIndex - 1);
      } else if (e.key === "Enter") {
        // If user has highlighted a suggestion, pick it.
        // Else, if input looks like a ticker, jump to the predict page
        // pre-filled with it. For v1: ignore Enter when nothing selected.
        if (activeIndex >= 0 && btns[activeIndex]) {
          e.preventDefault();
          pickTicker(btns[activeIndex].dataset.ticker);
        }
      } else if (e.key === "Escape") {
        e.preventDefault();
        input.blur();
        closeDropdown();
      }
    });

    // Click on a suggestion → navigate.
    dropdown.addEventListener("click", function (e) {
      const btn = e.target.closest(".search__btn");
      if (btn) pickTicker(btn.dataset.ticker);
    });

    // Hover sets active index — keeps keyboard + mouse in sync.
    dropdown.addEventListener("mouseover", function (e) {
      const btn = e.target.closest(".search__btn");
      if (!btn) return;
      const btns = getButtons();
      const idx = btns.indexOf(btn);
      if (idx !== -1) setActive(idx);
    });

    // Reset active index whenever HTMX swaps in fresh content.
    dropdown.addEventListener("htmx:afterSwap", function () {
      activeIndex = -1;
    });

    // Outside click closes the dropdown.
    document.addEventListener("click", function (e) {
      if (!root.contains(e.target)) closeDropdown();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
