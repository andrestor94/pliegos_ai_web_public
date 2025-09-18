# pdf_generator.py
from fpdf import FPDF
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# ====== Config ======
# Dónde guardar los PDFs. Por defecto usa ./generated_pdfs
PDF_DIR = os.getenv("PDF_DIR", "generated_pdfs")

# Nombre base por defecto del PDF (se le agrega timestamp)
DEFAULT_BASENAME = "informe_licitacion"


# ====== Utilidades ======
def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M%S")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _strip_html(html: str) -> str:
    """Quita tags HTML básicos para poder volcar texto en FPDF."""
    if not html:
        return ""
    # Reemplazos simples de bloques
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</(p|div|li|h\d)>", "\n", text, flags=re.I)
    # Quitar el resto de tags
    text = re.sub(r"<[^>]+>", "", text)
    # Normalizar espacios
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_text(s: Any, latin1_fallback: bool) -> str:
    """Convierte a str; si no hay fuente Unicode y falla, reemplaza caracteres fuera de latin-1."""
    s = "" if s is None else str(s)
    if not latin1_fallback:
        return s
    # Reemplazar caracteres fuera de latin-1 para evitar errores con fuentes core
    return s.encode("latin-1", "replace").decode("latin-1")


def _kv_to_lines(obj: Dict[str, Any]) -> List[str]:
    """Convierte dict clave-valor en líneas legibles."""
    lines = []
    for k, v in (obj or {}).items():
        if isinstance(v, (dict, list, tuple)):
            lines.append(f"{k}:")
            # sangrado simple
            for sub in _kv_to_lines(v) if isinstance(v, dict) else [f"- {x}" for x in v]:
                lines.append(f"  {sub}")
        else:
            lines.append(f"{k}: {v}")
    return lines


# ====== Clase PDF con header/footer ======
class ReportPDF(FPDF):
    def __init__(self, title: str, use_unicode: bool = True):
        super().__init__()
        self.title = title
        self._use_unicode = use_unicode
        self._latin1_fallback = False
        self._font_body = "Helvetica"
        self._font_bold = "Helvetica"
        self._setup_fonts()

        # Márgenes cómodos
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 18, 15)

    def _find_font(self) -> Optional[str]:
        """
        Intenta ubicar una fuente TTF Unicode (DejaVuSans / NotoSans).
        Devuelve ruta si la encuentra.
        """
        candidates = [
            # Comunes en Linux/Docker
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            # Mac
            "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf",
            # Windows
            "C:\\Windows\\Fonts\\DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\NotoSans-Regular.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",  # no es full unicode, pero ayuda
        ]
        for p in candidates:
            try:
                if os.path.isfile(p):
                    return p
            except Exception:
                pass
        return None

    def _setup_fonts(self):
        """
        - Si hay TTF Unicode disponible, la embebe y habilita Unicode real.
        - Si no, cae a Helvetica y usa fallback latin-1 para evitar errores.
        """
        if self._use_unicode:
            ttf = self._find_font()
            if ttf:
                try:
                    # Registrar fuente regular y "bold" si existe la variante
                    self.add_font("DejaVu", "", ttf, uni=True)
                    # Buscar variante bold a la par
                    bold_guess = ttf.replace("Regular", "Bold").replace(".ttf", "-Bold.ttf")
                    if os.path.isfile(bold_guess):
                        self.add_font("DejaVu", "B", bold_guess, uni=True)
                        self._font_bold = "DejaVu"
                    else:
                        # Si no hay bold, emular con style='B'
                        self._font_bold = "DejaVu"
                    self._font_body = "DejaVu"
                    self._latin1_fallback = False
                    return
                except Exception:
                    pass
        # Fallback a fuentes base (no unicode completo)
        self._font_body = "Helvetica"
        self._font_bold = "Helvetica"
        self._latin1_fallback = True

    # Encabezado/Pie
    def header(self):
        self.set_y(12)
        self.set_font(self._font_bold, "B", 12)
        tit = _safe_text(self.title, self._latin1_fallback)
        self.cell(0, 8, tit, ln=1)
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._font_body, "", 9)
        self.set_text_color(120, 120, 120)
        page_str = f"Página {self.page_no()}/{{nb}}"
        ts = datetime.now().strftime("%d/%m/%Y %H:%M")
        txt = _safe_text(f"{page_str}  •  Generado: {ts}", self._latin1_fallback)
        self.cell(0, 10, txt, align="R")

    # Helpers de texto
    def h2(self, text: str):
        self.set_font(self._font_bold, "B", 12)
        self.ln(1)
        self.multi_cell(0, 6, _safe_text(text, self._latin1_fallback))
        self.ln(1)

    def h3(self, text: str):
        self.set_font(self._font_bold, "B", 11)
        self.ln(0.5)
        self.multi_cell(0, 5.5, _safe_text(text, self._latin1_fallback))

    def p(self, text: str):
        self.set_font(self._font_body, "", 10)
        self.multi_cell(0, 5.5, _safe_text(text, self._latin1_fallback))
        self.ln(0.5)

    def bullets(self, items: List[str]):
        self.set_font(self._font_body, "", 10)
        for it in items or []:
            line = f"• {it}"
            self.multi_cell(0, 5.5, _safe_text(line, self._latin1_fallback))
        self.ln(0.5)

    def kv_block(self, pairs: Dict[str, Any]):
        self.set_font(self._font_body, "", 10)
        for k, v in (pairs or {}).items():
            key = _safe_text(str(k), self._latin1_fallback)
            if isinstance(v, (dict, list, tuple)):
                self.set_font(self._font_bold, "B", 10)
                self.multi_cell(0, 5.5, key + ":")
                self.set_font(self._font_body, "", 10)
                if isinstance(v, dict):
                    for line in _kv_to_lines(v):
                        self.multi_cell(0, 5.5, _safe_text("  " + line, self._latin1_fallback))
                else:
                    for x in v:
                        self.multi_cell(0, 5.5, _safe_text(f"  - {x}", self._latin1_fallback))
            else:
                val = _safe_text("" if v is None else str(v), self._latin1_fallback)
                # clave en "bold" + valor normal en misma línea (simple)
                self.set_font(self._font_bold, "B", 10)
                self.cell(0, 5.5, key + ":", ln=1)
                self.set_font(self._font_body, "", 10)
                self.multi_cell(0, 5.5, val)
        self.ln(0.5)


# ====== Renderers ======
def _render_structured(pdf: ReportPDF, s: Dict[str, Any]):
    """
    Espera un dict con claves como:
    - basic_info (dict)
    - timeline (list[str] o list[dict])
    - min_requirements (dict)
    - special_clauses (list[str] o list[dict])
    - contract_amount_duration (dict)
    - minutes_awards (dict o list)
    """
    if not s:
        pdf.h2("Análisis estructurado")
        pdf.p("No hay datos estructurados para mostrar.")
        return

    pdf.h2("Análisis estructurado")

    basic = s.get("basic_info") or {}
    if basic:
        pdf.h3("Datos básicos")
        pdf.kv_block(basic)

    tl = s.get("timeline") or []
    if tl:
        pdf.h3("Cronograma / Timeline")
        # Admitir lista de strings o dicts {'hito':'...','fecha':'...'}
        if tl and isinstance(tl[0], dict):
            items = []
            for it in tl:
                fecha = it.get("fecha") or it.get("date") or ""
                hito = it.get("hito") or it.get("event") or it.get("milestone") or ""
                items.append(f"{fecha} — {hito}".strip(" —"))
            pdf.bullets(items)
        else:
            pdf.bullets([str(x) for x in tl])

    reqs = s.get("min_requirements") or {}
    if reqs:
        pdf.h3("Requisitos mínimos")
        pdf.kv_block(reqs)

    clauses = s.get("special_clauses") or []
    if clauses:
        pdf.h3("Cláusulas especiales")
        if clauses and isinstance(clauses[0], dict):
            pdf.kv_block({"Cláusulas": clauses})
        else:
            pdf.bullets([str(x) for x in clauses])

    amount = s.get("contract_amount_duration") or {}
    if amount:
        pdf.h3("Monto y duración")
        pdf.kv_block(amount)

    minutes = s.get("minutes_awards") or {}
    if minutes:
        pdf.h3("Actas / Adjudicaciones")
        if isinstance(minutes, dict):
            pdf.kv_block(minutes)
        else:
            pdf.bullets([str(x) for x in minutes])


def _render_deep(pdf: ReportPDF, deep_sections: List[Dict[str, Any]]):
    """
    Espera una lista de secciones con:
    - title
    - content_html (o 'content')
    - section_key (opcional)
    """
    pdf.h2("Análisis profundo")

    if not deep_sections:
        pdf.p("No hay secciones de análisis profundo para mostrar.")
        return

    for sec in deep_sections:
        title = sec.get("title") or sec.get("section_key") or "Sección"
        html = sec.get("content_html")
        raw = sec.get("content")
        body = _strip_html(html) if html else (raw or "")
        pdf.h3(str(title))
        if body:
            # Dividir por párrafos para legibilidad
            paragraphs = [p.strip() for p in body.split("\n") if p.strip()]
            for p in paragraphs:
                pdf.p(p)
        else:
            pdf.p("(sin contenido)")


# ====== API principal ======
def generate_pdf(
    data: Dict[str, Any],
    filename: Optional[str] = None,
    title: str = "Informe de Licitación",
    include_structured: bool = True,
    include_deep: bool = True,
) -> str:
    """
    Genera un PDF a partir de:
      - data['análisis'] (texto plano legado)
      - y/o data['structured'], data['deep_analysis'] (nuevo modelo)

    Params:
      - filename: nombre de salida (sin ruta). Si no se pasa, usa DEFAULT_BASENAME + timestamp.
      - title: título de encabezado del PDF.
      - include_structured / include_deep: qué secciones incluir (si existen).

    Return:
      Ruta (absoluta) del PDF generado.
    """
    _ensure_dir(PDF_DIR)

    if not filename:
        filename = f"{DEFAULT_BASENAME}_{_now_stamp()}.pdf"

    # Asegurar .pdf
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    out_path = os.path.join(PDF_DIR, filename)

    pdf = ReportPDF(title=title, use_unicode=True)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Metadatos
    pdf.set_title(_safe_text(title, pdf._latin1_fallback))
    pdf.set_author("Sistema de Análisis")
    pdf.set_creator("Suizo — Análisis de Pliegos")

    # 1) Si viene análisis plano (legacy)
    legacy_text = data.get("análisis") or data.get("analisis") or data.get("resumen")
    if legacy_text:
        pdf.h2("Resumen")
        # Dividimos por líneas/ párrafos
        for line in str(legacy_text).splitlines():
            line = line.strip()
            if not line:
                pdf.ln(0.5)
                continue
            pdf.p(line)

    # 2) Modelo nuevo (estructurado / profundo)
    s = data.get("structured")
    deep = data.get("deep_analysis")

    if include_structured and s:
        _render_structured(pdf, s)

    if include_deep and deep:
        _render_deep(pdf, deep)

    # Si no hubo nada estructurado/ profundo y tampoco legacy, algo mínimo
    if not legacy_text and not s and not deep:
        pdf.p("No se encontró información para renderizar.")

    # Guardar
    pdf.output(out_path)

    # Devolver ruta absoluta (útil para servir desde FastAPI con StaticFiles)
    return os.path.abspath(out_path)
