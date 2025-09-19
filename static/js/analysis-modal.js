// static/js/analysis-modal.js
(() => {
  // ====== Utilidad: asegurar que el modal viva en <body> y sea ÚNICO ======
  function moveModalToBody(newRoot) {
    const found =
      newRoot?.id === "modalAnalysis"
        ? newRoot
        : newRoot?.querySelector?.("#modalAnalysis");

    if (!found) return;

    // Si existe un modal anterior, eliminarlo (evita "anclaje" y listeners duplicados)
    if (window.__analysisModal && window.__analysisModal !== found) {
      try {
        window.__analysisModal.remove();
      } catch (_) {}
      window.__analysisModal = null;
    }

    if (found.parentElement !== document.body) {
      document.body.appendChild(found);
    }

    // Accesibilidad y foco
    found.setAttribute("role", "dialog");
    if (!found.hasAttribute("tabindex")) found.setAttribute("tabindex", "-1");

    window.__analysisModal = found;
    wireModal(found); // cablear una sola vez por instancia
  }

  // Observa DOM para cuando inyectás el modal por fetch (HTML)
  const bootstrapMO = new MutationObserver((list) => {
    for (const m of list) {
      for (const n of m.addedNodes) {
        if (n.nodeType === 1) moveModalToBody(n);
      }
    }
  });
  bootstrapMO.observe(document.documentElement, { childList: true, subtree: true });

  // Cubre el caso en el que ya estuviera en el DOM
  moveModalToBody(document);

  // ====== Cableado del modal ======
  function wireModal(modal) {
    if (!modal || modal.dataset.wired === "1") return;
    modal.dataset.wired = "1";

    // Guardar estado de scroll para restaurar al cerrar
    let lastScrollY = 0;

    function lockScroll() {
      lastScrollY = window.scrollY || document.documentElement.scrollTop || 0;
      document.body.style.overflow = "hidden";
    }
    function unlockScroll() {
      document.body.style.overflow = "";
      window.scrollTo({ top: lastScrollY });
    }

    // API pública para abrir/cerrar (usada por index.html tras inyectar el HTML)
    function open() {
      modal.style.display = "flex";
      lockScroll();
      // Foco al primer "tab" o elemento accionable
      (modal.querySelector(".sa-tab, [data-close], button, [href], input, select, textarea") || modal).focus?.();
      // Listener de Escape a nivel documento mientras esté abierto
      document.addEventListener("keydown", onEsc, { once: true });
    }
    function close() {
      try {
        modal.remove();
      } catch (_) {}
      unlockScroll();
      // Limpieza de referencias globales
      if (window.__analysisModal === modal) window.__analysisModal = null;
      if (window.AnalysisModalOpen) window.AnalysisModalOpen = null;
      if (window.AnalysisModalClose) window.AnalysisModalClose = null;
    }

    function onEsc(ev) {
      if (ev.key === "Escape") close();
    }

    // Cerrar por click en backdrop (si el HTML del modal usa el propio contenedor como backdrop)
    modal.addEventListener("click", (ev) => {
      if (ev.target === modal) close();
    });

    // Cerrar por botones con data-close
    modal.querySelectorAll("[data-close]").forEach((btn) => {
      btn.addEventListener("click", close);
    });

    // Exportar PDF (opcional si existen esos botones y el hidden con JSON)
    const jsonInput = modal.querySelector("#analysis-json");
    function exportPdf(kind) {
      const form = document.createElement("form");
      form.method = "POST";
      form.action = kind === "structured" ? "/export/pdf/estructurado" : "/export/pdf/profundo";
      form.target = "_blank";
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = "analysis_json";
      input.value = jsonInput?.value || "{}";
      form.appendChild(input);
      document.body.appendChild(form);
      form.submit();
      form.remove();
    }
    modal.querySelectorAll("[data-export]").forEach((btn) => {
      btn.addEventListener("click", () => exportPdf(btn.getAttribute("data-export")));
    });

    // Exponer funciones globales (usadas por tu index.html al terminar de insertar el modal)
    window.AnalysisModalOpen = open;
    window.AnalysisModalClose = close;
  }

  // Reintenta cablear si el modal aparece tardíamente
  function tryWire() {
    const modal = document.getElementById("modalAnalysis");
    if (modal) moveModalToBody(modal);
  }
  document.addEventListener("DOMContentLoaded", tryWire);
  const lateMO = new MutationObserver(tryWire);
  lateMO.observe(document.body, { childList: true, subtree: true });
})();
