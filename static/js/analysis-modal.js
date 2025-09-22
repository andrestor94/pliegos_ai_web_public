// static/js/analysis-modal.js
(() => {
  // -------------------------------
  // Config
  // -------------------------------
  const LOAD_TIMEOUT_MS = 20000; // 20s para vistas remotas
  const LS_ACTIVE_TAB_KEY = "analysisModal:activeTab";

  // ================================
  // Utiles comunes
  // ================================
  function $(root, sel) {
    return (root || document).querySelector(sel);
  }
  function $all(root, sel) {
    return Array.from((root || document).querySelectorAll(sel));
  }
  function escapeHTML(s) {
    return (s ?? "")
      .toString()
      .replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function withTimeout(promise, ms, signal) {
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => {
        try { if (signal && signal.abort) signal.abort(); } catch (_) {}
        reject(new Error("timeout"));
      }, ms);
      promise.then(
        (v) => { clearTimeout(t); resolve(v); },
        (e) => { clearTimeout(t); reject(e); }
      );
    });
  }
  function makeAbortable() {
    const ctl = typeof AbortController !== "undefined" ? new AbortController() : null;
    return { ctl, signal: ctl ? ctl.signal : undefined };
  }

  // ================================
  // Montaje único en <body>
  // ================================
  function moveModalToBody(newRoot) {
    const found = newRoot?.id === "modalAnalysis" ? newRoot : newRoot?.querySelector?.("#modalAnalysis");
    if (!found) return;

    // Si ya hay una instancia previa, removerla para evitar listeners duplicados
    if (window.__analysisModal && window.__analysisModal !== found) {
      try { window.__analysisModal.remove(); } catch (_) {}
      window.__analysisModal = null;
    }

    if (found.parentElement !== document.body) {
      document.body.appendChild(found);
    }

    // A11y base
    found.setAttribute("role", "dialog");
    found.setAttribute("aria-modal", "true");
    if (!found.hasAttribute("tabindex")) found.setAttribute("tabindex", "-1");

    window.__analysisModal = found;
    wireModal(found); // cablear una sola vez
  }

  const bootstrapMO = new MutationObserver((list) => {
    for (const m of list) {
      for (const n of m.addedNodes) {
        if (n.nodeType === 1) moveModalToBody(n);
      }
    }
  });
  bootstrapMO.observe(document.documentElement, { childList: true, subtree: true });
  // Cubre el caso de que ya esté
  moveModalToBody(document);

  // ================================
  // Carga perezosa: vistas remotas
  // ================================
  async function loadStructured(modal) {
    if (modal.dataset.structuredLoaded === "1") return;
    const pane = $("#pane-structured", modal);
    const json = $("#analysis-json", modal)?.value || "{}";

    showSkeleton(pane);
    const { ctl, signal } = makeAbortable();
    modal.__structCtl?.abort?.(); // aborta anterior si existía
    modal.__structCtl = ctl;

    try {
      const fd = new FormData();
      fd.append("analysis_json", json);
      const r = await withTimeout(fetch("/render/structured", {
        method: "POST",
        body: fd,
        headers: { "X-Requested-With": "fetch" },
        signal
      }), LOAD_TIMEOUT_MS, ctl);

      if (!r.ok) throw new Error("HTTP " + r.status);
      pane.innerHTML = await r.text();
      modal.dataset.structuredLoaded = "1";
    } catch (e) {
      pane.innerHTML = errorBlock("No se pudo cargar la vista estructurada.", () => loadStructured(modal));
    }
  }

  async function loadDeep(modal) {
    if (modal.dataset.deepLoaded === "1") return;
    const pane = $("#pane-deep", modal);
    const json = $("#analysis-json", modal)?.value || "{}";

    showSkeleton(pane);
    const { ctl, signal } = makeAbortable();
    modal.__deepCtl?.abort?.();
    modal.__deepCtl = ctl;

    try {
      const fd = new FormData();
      fd.append("analysis_json", json);
      const r = await withTimeout(fetch("/render/deep", {
        method: "POST",
        body: fd,
        headers: { "X-Requested-With": "fetch" },
        signal
      }), LOAD_TIMEOUT_MS, ctl);

      if (!r.ok) throw new Error("HTTP " + r.status);
      pane.innerHTML = await r.text();
      modal.dataset.deepLoaded = "1";
    } catch (e) {
      pane.innerHTML = errorBlock("No se pudo cargar el análisis profundo.", () => loadDeep(modal));
    }
  }

  function showSkeleton(pane) {
    pane.innerHTML = `
      <div class="sa-skel" aria-hidden="true">
        <div class="sa-skel-line w-50"></div>
        <div class="sa-skel-line w-90"></div>
        <div class="sa-skel-line w-85"></div>
        <div class="sa-skel-line w-70"></div>
      </div>`;
  }

  function errorBlock(msg, onRetry) {
    const id = "err-" + Math.random().toString(36).slice(2);
    // Botón de reintento accesible
    setTimeout(() => {
      const btn = document.getElementById(id);
      btn?.addEventListener("click", (e) => {
        e.preventDefault();
        onRetry?.();
      });
    }, 0);

    return `
      <div class="alert alert-danger d-flex align-items-center justify-content-between">
        <div>${escapeHTML(msg)}</div>
        <button id="${id}" class="btn btn-sm btn-outline-light ms-2"><i class="bi bi-arrow-repeat"></i> Reintentar</button>
      </div>`;
  }

  // ================================
  // Cableado del modal (una vez)
  // ================================
  function wireModal(modal) {
    if (!modal || modal.dataset.wired === "1") return;
    modal.dataset.wired = "1";

    // Live region para avisos
    let live = $("#analysis-live", modal);
    if (!live) {
      live = document.createElement("div");
      live.id = "analysis-live";
      live.className = "visually-hidden";
      live.setAttribute("aria-live", "polite");
      modal.appendChild(live);
    }

    // Tabs — asegurar atributos ARIA
    const tabs = $all(modal, ".sa-tab");
    const panes = $all(modal, ".sa-pane");

    const tablist = $(".sa-tablist", modal) || tabs[0]?.parentElement;
    if (tablist) tablist.setAttribute("role", "tablist");

    tabs.forEach((btn, i) => {
      btn.setAttribute("role", "tab");
      btn.setAttribute("tabindex", i === 0 ? "0" : "-1");
      const tabName = btn.dataset.tab;
      const pane = $("#pane-" + tabName, modal);
      if (pane) {
        const paneId = "pane-" + tabName;
        pane.setAttribute("role", "tabpanel");
        pane.setAttribute("id", paneId);
        btn.setAttribute("aria-controls", paneId);
      }
    });

    panes.forEach((p, i) => {
      if (!p.id) p.id = "pane-" + (i + 1);
      p.setAttribute("tabindex", "0");
    });

    // Estado de scroll y foco
    let lastScrollY = 0;
    let lastActiveElement = null;

    function lockScroll() {
      lastScrollY = window.scrollY || document.documentElement.scrollTop || 0;
      document.body.style.overflow = "hidden";
    }
    function unlockScroll() {
      document.body.style.overflow = "";
      try { window.scrollTo({ top: lastScrollY }); } catch (_) {}
    }

    function focusFirstFocusable() {
      const focusables = $all(modal, 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')
        .filter((el) => !el.disabled && el.offsetParent !== null && getComputedStyle(el).visibility !== "hidden");
      (focusables[0] || modal).focus?.();
    }

    // Focus trap
    function onKeydownTrap(e) {
      if (e.key !== "Tab") return;
      const focusables = $all(modal, 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')
        .filter((el) => !el.disabled && el.offsetParent !== null && getComputedStyle(el).visibility !== "hidden");
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    }

    function onEsc(ev) { if (ev.key === "Escape") close(); }

    // API pública
    async function open(initialTab = null) {
      try { lastActiveElement = document.activeElement; } catch (_) { lastActiveElement = null; }
      modal.style.display = "flex";
      modal.setAttribute("aria-hidden", "false");
      document.addEventListener("keydown", onEsc, { capture: true });
      document.addEventListener("keydown", onKeydownTrap, { capture: true });
      lockScroll();

      // Restaurar tab activo (o el indicado)
      let tabToOpen = initialTab || localStorage.getItem(LS_ACTIVE_TAB_KEY) || "structured";
      if (!$("#pane-" + tabToOpen, modal)) tabToOpen = "structured";
      setActiveTab(tabToOpen, false);

      focusFirstFocusable();
      // Cargar la pestaña actual si es la estructurada
      if (tabToOpen === "structured") await loadStructured(modal);
      if (tabToOpen === "deep") await loadDeep(modal);

      live.textContent = "Análisis abierto.";
    }

    function close() {
      // Abortamos fetch en curso para liberar recursos
      try { modal.__structCtl?.abort?.(); modal.__deepCtl?.abort?.(); } catch (_) {}
      document.removeEventListener("keydown", onEsc, { capture: true });
      document.removeEventListener("keydown", onKeydownTrap, { capture: true });
      unlockScroll();
      modal.setAttribute("aria-hidden", "true");
      try { modal.remove(); } catch (_) { modal.style.display = "none"; }

      if (window.__analysisModal === modal) window.__analysisModal = null;
      if (window.AnalysisModalOpen) window.AnalysisModalOpen = null;
      if (window.AnalysisModalClose) window.AnalysisModalClose = null;

      // restaurar foco
      try { lastActiveElement?.focus?.(); } catch (_) {}
      live.textContent = "Análisis cerrado.";
    }

    // Cerrar por backdrop
    modal.addEventListener("click", (ev) => { if (ev.target === modal) close(); });
    // Cerrar por botones con data-close
    $all(modal, "[data-close]").forEach((btn) => btn.addEventListener("click", close));

    // Tabs + lazy deep + navegación con teclado (izq/der, Home/End)
    function setActiveTab(tab, focusBtn = true) {
      const targetBtn = tabs.find((b) => b.dataset.tab === tab) || tabs[0];
      const targetPane = $("#pane-" + (targetBtn?.dataset.tab || ""), modal);
      if (!targetBtn || !targetPane) return;

      tabs.forEach((b) => {
        const on = b === targetBtn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
        b.setAttribute("tabindex", on ? "0" : "-1");
        // visible focus
        if (on && focusBtn) b.focus();
      });
      panes.forEach((p) => p.classList.remove("active"));
      targetPane.classList.add("active");

      try { localStorage.setItem(LS_ACTIVE_TAB_KEY, targetBtn.dataset.tab); } catch (_) {}

      if (targetBtn.dataset.tab === "deep") loadDeep(modal);
      if (targetBtn.dataset.tab === "structured") loadStructured(modal);
    }

    tabs.forEach((btn, idx) => {
      btn.addEventListener("click", () => setActiveTab(btn.dataset.tab, true));
      btn.addEventListener("keydown", (e) => {
        const k = e.key;
        if (k === "ArrowRight" || k === "ArrowLeft" || k === "Home" || k === "End") {
          e.preventDefault();
          let nextIndex = idx;
          if (k === "ArrowRight") nextIndex = (idx + 1) % tabs.length;
          if (k === "ArrowLeft") nextIndex = (idx - 1 + tabs.length) % tabs.length;
          if (k === "Home") nextIndex = 0;
          if (k === "End") nextIndex = tabs.length - 1;
          setActiveTab(tabs[nextIndex].dataset.tab, true);
        }
        if (k === "Enter" || k === " ") {
          e.preventDefault();
          setActiveTab(btn.dataset.tab, true);
        }
      });
    });

    // Exportar PDF (si existen endpoints)
    const jsonInput = $("#analysis-json", modal);
    function exportPdf(kind) {
      if (!jsonInput) return;
      const form = document.createElement("form");
      form.method = "POST";
      form.action = kind === "structured" ? "/export/pdf/estructurado" : "/export/pdf/profundo";
      form.target = "_blank";
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = "analysis_json";
      input.value = jsonInput.value || "{}";
      form.appendChild(input);
      document.body.appendChild(form);
      form.submit();
      form.remove();
    }
    $all(modal, "[data-export]").forEach((btn) =>
      btn.addEventListener("click", () => exportPdf(btn.getAttribute("data-export")))
    );

    // API pública:
    // - AnalysisModalOpen(initialTab?: "structured" | "deep")
    // - AnalysisModalClose()
    // - AnalysisModalSetJSON(jsonString, opts?: { reset?: boolean, openTab?: "structured"|"deep" })
    async function setJSON(jsonString, opts) {
      const textarea = $("#analysis-json", modal);
      if (!textarea) return;
      textarea.value = jsonString || "{}";
      const reset = !!(opts && opts.reset);
      if (reset) {
        modal.dataset.structuredLoaded = "0";
        modal.dataset.deepLoaded = "0";
        $("#pane-structured", modal).innerHTML = "";
        $("#pane-deep", modal).innerHTML = "";
      }
      if (opts?.openTab) setActiveTab(opts.openTab, true);
    }

    window.AnalysisModalOpen = open;
    window.AnalysisModalClose = close;
    window.AnalysisModalSetJSON = setJSON;

    // Auto-carga de “Vista estructurada” al primer open desde fuera
    // (se hace en open() según la pestaña activa)
  }

  // ================================
  // Reintento tardío de cableado
  // ================================
  function tryWire() {
    const modal = document.getElementById("modalAnalysis");
    if (modal) moveModalToBody(modal);
  }
  document.addEventListener("DOMContentLoaded", tryWire);
  const lateMO = new MutationObserver(tryWire);
  lateMO.observe(document.body, { childList: true, subtree: true });
})();
