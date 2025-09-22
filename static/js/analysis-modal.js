// static/js/analysis-modal.js
(() => {
  // ====== Utilidad: obtener el modal sin importar el id ======
  function findModal(root) {
    if (!root) return null;
    if (root.getElementById) {
      return (
        root.getElementById("modalAnalysis") ||
        root.getElementById("analysisModal") ||
        root.querySelector("#modalAnalysis, #analysisModal")
      );
    }
    return document.querySelector("#modalAnalysis, #analysisModal");
  }

  // ====== Asegurar que el modal viva en <body> y sea ÚNICO ======
  function moveModalToBody(newRoot) {
    const found =
      newRoot?.id === "modalAnalysis" || newRoot?.id === "analysisModal"
        ? newRoot
        : newRoot?.querySelector?.("#modalAnalysis, #analysisModal") || null;

    if (!found) return;

    // Si existe un modal anterior, eliminarlo (evita listeners duplicados)
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

  // Observa DOM por si inyectás el modal por fetch/htmx
  const bootstrapMO = new MutationObserver((list) => {
    for (const m of list) {
      for (const n of m.addedNodes) {
        if (n.nodeType === 1) moveModalToBody(n);
      }
    }
  });
  bootstrapMO.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });

  // Cubre el caso de que ya esté en el DOM
  moveModalToBody(document);

  // ====== Útiles para esperar el resultado ======
  function getAnalysisJSON(modal) {
    const v = modal.querySelector("#analysis-json")?.value?.trim() || "";
    // consideramos "no listo" si está vacío o es "{}"
    if (!v || v === "{}") return "";
    return v;
  }

  // Espera a que #analysis-json tenga datos (hasta timeoutMs)
  function waitForJSON(modal, timeoutMs = 30000) {
    return new Promise((resolve) => {
      const start = Date.now();
      const tick = () => {
        const json = getAnalysisJSON(modal);
        if (json) return resolve(true);
        if (Date.now() - start >= timeoutMs) return resolve(false);
        setTimeout(tick, 400);
      };
      tick();
    });
  }

  // ====== Carga de contenido ======
  async function loadStructured(modal) {
    if (modal.dataset.structuredLoaded === "1") return;
    const pane = modal.querySelector("#pane-structured");
    const json = getAnalysisJSON(modal);

    if (pane) {
      pane.innerHTML = `
        <div class="sa-skel">
          <div class="sa-skel-line w-50"></div>
          <div class="sa-skel-line w-90"></div>
          <div class="sa-skel-line w-85"></div>
          <div class="sa-skel-line w-70"></div>
        </div>`;
    }
    if (!json) {
      if (pane)
        pane.innerHTML = `<div class="alert alert-info">
          Esperando resultado del análisis...
        </div>`;
      return; // aún no hay datos: no pidas al servidor
    }

    try {
      const fd = new FormData();
      fd.append("analysis_json", json);
      const r = await fetch("/render/structured", {
        method: "POST",
        body: fd,
        headers: { "X-Requested-With": "fetch" },
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      if (pane) pane.innerHTML = await r.text();
      modal.dataset.structuredLoaded = "1";
    } catch (e) {
      if (pane)
        pane.innerHTML =
          "<div class='alert alert-danger'>No se pudo cargar la vista estructurada.</div>";
    }
  }

  async function loadDeep(modal) {
    if (modal.dataset.deepLoaded === "1") return;
    const pane = modal.querySelector("#pane-deep");
    const json = getAnalysisJSON(modal);

    if (pane) {
      pane.innerHTML = `
        <div class="sa-skel">
          <div class="sa-skel-line w-50"></div>
          <div class="sa-skel-line w-90"></div>
          <div class="sa-skel-line w-85"></div>
          <div class="sa-skel-line w-70"></div>
        </div>`;
    }
    if (!json) {
      if (pane)
        pane.innerHTML = `<div class="alert alert-info">
          Esperando resultado del análisis...
        </div>`;
      return;
    }

    try {
      const fd = new FormData();
      fd.append("analysis_json", json);
      const r = await fetch("/render/deep", {
        method: "POST",
        body: fd,
        headers: { "X-Requested-With": "fetch" },
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      if (pane) pane.innerHTML = await r.text();
      modal.dataset.deepLoaded = "1";
    } catch (e) {
      if (pane)
        pane.innerHTML =
          "<div class='alert alert-danger'>No se pudo cargar el análisis profundo.</div>";
    }
  }

  // ====== Cableado del modal ======
  function wireModal(modal) {
    if (!modal || modal.dataset.wired === "1") return;
    modal.dataset.wired = "1";

    // Evitar que algo externo “mate” la interactividad
    modal.style.pointerEvents = "auto";

    // Guardar estado de scroll para restaurar al cerrar
    let lastScrollY = 0;
    function lockScroll() {
      lastScrollY =
        window.scrollY || document.documentElement.scrollTop || 0;
      document.body.style.overflow = "hidden";
    }
    function unlockScroll() {
      document.body.style.overflow = "";
      window.scrollTo({ top: lastScrollY });
    }

    function onEsc(ev) {
      if (ev.key === "Escape") close();
    }

    // API pública
    function open() {
      modal.style.display = "flex";
      modal.setAttribute("aria-hidden", "false");
      lockScroll();
      (
        modal.querySelector(
          ".sa-tab, [data-close], button, [href], input, select, textarea"
        ) || modal
      ).focus?.();
      document.addEventListener("keydown", onEsc, { once: true });

      // 👉 cargar “Vista estructurada” cuando haya datos (o mostrar "esperando")
      (async () => {
        const hasJson =
          !!getAnalysisJSON(modal) || (await waitForJSON(modal, 30000));
        await loadStructured(modal); // si no hay JSON, mostrará "esperando"
        if (!hasJson) {
          // opcional: log para debug
          console.warn(
            "El análisis no llegó en 30s. Backend puede estar tardando o fallando."
          );
        }
      })();
    }
    function close() {
      try {
        modal.remove();
      } catch (_) {}
      unlockScroll();
      if (window.__analysisModal === modal) window.__analysisModal = null;
      if (window.AnalysisModalOpen) window.AnalysisModalOpen = null;
      if (window.AnalysisModalClose) window.AnalysisModalClose = null;
    }

    // Cerrar por click en backdrop
    modal.addEventListener("click", (ev) => {
      if (ev.target === modal) close();
    });

    // Cerrar por botones con data-close
    modal
      .querySelectorAll("[data-close]")
      .forEach((btn) => btn.addEventListener("click", close));

    // Tabs + lazy deep
    modal.querySelectorAll(".sa-tab").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const tab = btn.dataset.tab;
        modal.querySelectorAll(".sa-tab").forEach((b) => {
          const active = b.dataset.tab === tab;
          b.classList.toggle("active", active);
          b.setAttribute("aria-selected", active ? "true" : "false");
        });
        modal
          .querySelectorAll(".sa-pane")
          .forEach((p) => p.classList.remove("active"));
        const pane = modal.querySelector(`#pane-${tab}`);
        if (pane) pane.classList.add("active");
        if (tab === "deep") await loadDeep(modal);
      });
    });

    // Exportar PDF (si existen endpoints)
    const jsonInput = modal.querySelector("#analysis-json");
    function exportPdf(kind) {
      const form = document.createElement("form");
      form.method = "POST";
      form.action =
        kind === "structured"
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
    modal
      .querySelectorAll("[data-export]")
      .forEach((btn) =>
        btn.addEventListener("click", () =>
          exportPdf(btn.getAttribute("data-export"))
        )
      );

    // Exponer funciones globales
    window.AnalysisModalOpen = open;
    window.AnalysisModalClose = close;
  }

  // Reintento si aparece tarde
  function tryWire() {
    const modal =
      findModal(document) ||
      document.querySelector("#modalAnalysis, #analysisModal");
    if (modal) moveModalToBody(modal);
  }
  document.addEventListener("DOMContentLoaded", tryWire);
  const lateMO = new MutationObserver(tryWire);
  lateMO.observe(document.body, { childList: true, subtree: true });
})();
