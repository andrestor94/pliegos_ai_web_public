// static/js/analysis-modal.js
(function () {
  // --- Util: mover el modal al <body> si llega inyectado
  function moveToBodyIfNeeded(root) {
    const el = (root?.id === "modalAnalysis")
      ? root
      : root?.querySelector?.("#modalAnalysis");
    if (el && el.parentElement !== document.body) document.body.appendChild(el);
  }

  const mo = new MutationObserver((muts) => {
    for (const m of muts) for (const n of m.addedNodes)
      if (n.nodeType === 1) moveToBodyIfNeeded(n);
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });
  moveToBodyIfNeeded(document);

  // --- Cableado del nuevo modal con tabs
  function wireModal(modal) {
    if (!modal || modal.dataset.wired === "1") return;
    modal.dataset.wired = "1";

    // Cerrar por botón con data-close o por ✕ (ya viene en el HTML)
    modal.querySelectorAll("[data-close]").forEach(btn => {
      btn.addEventListener("click", () => {
        modal.remove();
        document.body.style.overflow = "";
      });
    });

    // Cerrar con ESC
    modal.addEventListener("keydown", (ev) => {
      if (ev.key === "Escape") {
        modal.remove();
        document.body.style.overflow = "";
      }
    });

    // Apertura programática
    function open() {
      modal.style.display = "flex";
      document.body.style.overflow = "hidden";
      // foco seguro para accesibilidad
      const firstTab = modal.querySelector('.sa-tab');
      firstTab?.focus?.();
    }
    window.AnalysisModalOpen = open;

    // --- Exportar a PDF (opcional)
    // Si tu plantilla incluye botones:
    //   <button data-export="structured">Exportar PDF</button>
    //   <button data-export="deep">Exportar PDF</button>
    // y un input hidden con el JSON:
    //   <input type="hidden" id="analysis-json" value="{}">
    const jsonInput = modal.querySelector("#analysis-json");
    function exportPdf(kind) {
      const form = document.createElement("form");
      form.method = "POST";
      form.action = (kind === "structured")
        ? "/export/pdf/estructurado"
        : "/export/pdf/profundo";
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

    modal.querySelectorAll("[data-export]").forEach(btn => {
      btn.addEventListener("click", () => {
        const kind = btn.getAttribute("data-export"); // 'structured' | 'deep'
        exportPdf(kind);
      });
    });
  }

  function tryWire() {
    const modal = document.getElementById("modalAnalysis");
    if (modal) wireModal(modal);
  }

  document.addEventListener("DOMContentLoaded", tryWire);
  const lateMO = new MutationObserver(tryWire);
  lateMO.observe(document.body, { childList: true, subtree: true });
})();
