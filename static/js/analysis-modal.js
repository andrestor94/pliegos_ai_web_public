// static/js/analysis-modal.js
(function () {
  // --- Util: mover #analysis-modal al <body> si aparece inyectado en runtime
  const moveToBody = (root) => {
    const el = (root?.id === "analysis-modal")
      ? root
      : root?.querySelector?.("#analysis-modal");
    if (el && el.parentElement !== document.body) document.body.appendChild(el);
  };

  const mo = new MutationObserver((muts) => {
    for (const m of muts) {
      for (const n of m.addedNodes) {
        if (n.nodeType === 1) moveToBody(n);
      }
    }
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });
  moveToBody(document); // por si ya estaba

  // --- Fallback local si no tenés definido window.setupAnalysisModal en la página
  function localSetupAnalysisModal(root) {
    if (!root || root.dataset.wired === "1") return;
    root.dataset.wired = "1";

    const $ = (sel, ctx = root) => ctx.querySelector(sel);

    // Vistas
    const stepChooser = $("#step-chooser");
    const stepResult  = $("#step-result");
    const viewEl      = $("#analysis-view");
    const titleEl     = $("#result-title");
    const btnConfirm  = $("#confirm-mode");
    const btnBack     = $("#go-back");
    const btnPdf      = $("#download-pdf");
    const jsonInput   = $("#analysis-json");

    // Cerrar modal (backdrop o botón con data-close)
    root.querySelectorAll("[data-close]").forEach((btn) => {
      btn.addEventListener("click", () => {
        root.style.display = "none";
        document.body.style.overflow = "";
      });
    });

    // Helpers de vista
    function show(which) {
      if (which === "chooser") {
        stepChooser && (stepChooser.style.display = "");
        stepResult && (stepResult.style.display = "none");
      } else {
        stepChooser && (stepChooser.style.display = "none");
        stepResult && (stepResult.style.display = "");
      }
    }

    function openModal() {
      root.style.display = "flex";
      document.body.style.overflow = "hidden";
      show("chooser");
      if (btnConfirm) btnConfirm.disabled = true;
      root.querySelectorAll(".sa-card").forEach((c) => c.classList.remove("active"));
    }

    // Exponer un abridor global por si lo necesitás desde otro script
    window.AnalysisModalOpen = openModal;

    // Manejo de selección de tarjetas
    let chosen = null;
    root.querySelectorAll(".sa-card").forEach((card) => {
      card.addEventListener("click", () => {
        root.querySelectorAll(".sa-card").forEach((c) => c.classList.remove("active"));
        card.classList.add("active");
        chosen = card.getAttribute("data-mode"); // 'structured' | 'deep'
        if (btnConfirm) btnConfirm.disabled = !chosen;
      });
    });

    // Render de la vista elegida
    async function loadView() {
      if (!viewEl) return;

      const endpoint = chosen === "structured" ? "/render/structured" : "/render/deep";
      const title    = chosen === "structured"
        ? "Resultado del Análisis Estructurado"
        : "Resultado del Análisis Profundo";

      const fd = new FormData();
      try {
        // jsonInput.value ya viene serializado (string), lo pasamos tal cual
        fd.append("analysis_json", jsonInput?.value || "{}");
      } catch {
        // fallback duro
        fd.append("analysis_json", "{}");
      }

      titleEl && (titleEl.textContent = title);
      viewEl.innerHTML = '<div class="text-muted">Cargando…</div>';

      try {
        const r = await fetch(endpoint, { method: "POST", body: fd });
        const html = await r.text();
        if (!r.ok) throw new Error("No se pudo renderizar.");
        viewEl.innerHTML = html;
      } catch (err) {
        viewEl.innerHTML = `<div class="text-danger">❌ ${err?.message || "Error al renderizar."}</div>`;
      }
    }

    // Confirmar y pasar a resultado
    btnConfirm && btnConfirm.addEventListener("click", async () => {
      if (!chosen) return;
      // configurar botón PDF
      if (btnPdf) {
        btnPdf.disabled = false;
        btnPdf.onclick = () => {
          const form = document.createElement("form");
          form.method = "POST";
          form.action = (chosen === "structured")
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
        };
      }

      show("result");
      await loadView();
    });

    // Volver a la elección
    btnBack && btnBack.addEventListener("click", () => {
      show("chooser");
      if (btnConfirm) btnConfirm.disabled = true;
      root.querySelectorAll(".sa-card").forEach((c) => c.classList.remove("active"));
      chosen = null;
      if (viewEl) viewEl.innerHTML = "";
      if (btnPdf) {
        btnPdf.disabled = true;
        btnPdf.onclick = null;
      }
    });

    // Si alguien inyecta el modal ya visible, respetamos display actual.
    // Si está oculto (display:none) lo abre quien corresponda (tu flujo en index.html).
  }

  // --- Inicialización idempotente: usa tu helper si existe; si no, el fallback
  function wireIfPresent() {
    const modal = document.getElementById("analysis-modal");
    if (!modal) return;

    // Si la página definió setupAnalysisModal, usarlo
    if (typeof window.setupAnalysisModal === "function") {
      if (modal.dataset.wired === "1") return;
      window.setupAnalysisModal(modal);
      modal.dataset.wired = "1";
      return;
    }

    // Si no, aplicar el fallback local
    localSetupAnalysisModal(modal);
  }

  document.addEventListener("DOMContentLoaded", wireIfPresent);

  // Por si el modal llega después (inyectado tras fetch)
  const lateMO = new MutationObserver(() => wireIfPresent());
  lateMO.observe(document.body, { childList: true, subtree: true });
})();
