// static/js/analysis-modal.js
(function () {
  // Mover cualquier #analysis-modal al <body> apenas aparezca
  const moveToBody = (root) => {
    const el = (root?.id === "analysis-modal") ? root : root?.querySelector?.("#analysis-modal");
    if (el && el.parentElement !== document.body) document.body.appendChild(el);
  };

  const mo = new MutationObserver((muts) => {
    for (const m of muts) for (const n of m.addedNodes) if (n.nodeType === 1) moveToBody(n);
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });
  // Por si ya está
  moveToBody(document);

  // Helpers para render y toggle de descargas
  async function renderTo(url, json) {
    const out = document.getElementById("analysis-result");
    if (!out) return;
    try {
      const fd = new FormData();
      fd.append("analysis_json", json || "");
      const r = await fetch(url, { method: "POST", body: fd, headers: { "X-Requested-With": "fetch" } });
      out.innerHTML = await r.text();
      out.scrollTop = 0;
    } catch {
      out.innerHTML = '<div class="text-danger">Error renderizando la vista.</div>';
    }
  }
  function currentJSON() {
    const ta = document.getElementById("analysis-json");
    return ta ? ta.value : "";
  }
  function show(which) {
    const fEstr = document.getElementById("dl-estr");
    const fDeep = document.getElementById("dl-deep");
    if (fEstr) fEstr.style.display = (which === "estr") ? "inline-block" : "none";
    if (fDeep) fDeep.style.display = (which === "deep") ? "inline-block" : "none";
  }

  // Delegación de eventos (funciona aunque el script del modal no se ejecute)
  document.addEventListener("click", (e) => {
    const estr = e.target.closest?.("#opt-estr");
    const deep = e.target.closest?.("#opt-deep");
    if (!estr && !deep) return;

    e.preventDefault();
    const json = currentJSON();
    if (estr) { show("estr"); renderTo("/render/structured", json); }
    if (deep) { show("deep"); renderTo("/render/deep", json); }
  });

  // Cerrar modal (por si el botón inline no corre)
  document.addEventListener("click", (e) => {
    const closeBtn = e.target.closest?.(".analysis-modal .close");
    if (!closeBtn) return;
    const modal = document.getElementById("analysis-modal");
    if (modal) modal.remove();
  });
})();
