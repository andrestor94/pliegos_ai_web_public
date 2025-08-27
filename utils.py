# -*- coding: utf-8 -*-
# utils.py — Parte 1/5
import io
import os
import re
import base64
import mimetypes
import time
import json  # necesario para tool-calling en el chat
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from tempfile import NamedTemporaryFile

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from zoneinfo import ZoneInfo  # fallback local AR

# === NUEVO: prompts centralizados ===
# Solo importamos lo que realmente usamos; con fallbacks si no existe en prompts.py
try:
    # prompts.py debe estar en la MISMA carpeta que utils.py y main.py
    from prompts import SINONIMOS_CANONICOS, prompt_andres, CRAFT_PROMPT_NOTAS
except Exception as e:
    print(f"[WARN] prompts.py faltante o sin símbolos esperados: {e}")
    SINONIMOS_CANONICOS = ""  # fallback: guía vacía
    def prompt_andres(varios_anexos: bool) -> str:
        # fallback mínimo para no romper si prompts.py no define esta función
        return (
            "Elabora un informe técnico-jurídico estructurado con citas literales. "
            "No inventes. Cita como '(p. N)'. Si hay múltiples anexos, usa '(Anexo X, p. N)'."
        )
    CRAFT_PROMPT_NOTAS = (
        "Extrae bullets técnicos y concisos con citas literales; cero invenciones."
    )

# ========================= Opcionales (DOCX) =========================
try:
    import docx  # python-docx
except Exception:
    docx = None

load_dotenv()

# ========================= OpenAI client =========================
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "90"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT)

# ========================= Modelos / Heurísticas =========================
# Sugerencia: si usas variantes específicas, setea las envs:
# OPENAI_MODEL_ANALISIS=gpt-5.1   OPENAI_MODEL_VISION=gpt-5.1
MODEL_ANALISIS  = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-5")
VISION_MODEL    = os.getenv("OPENAI_MODEL_VISION", "gpt-5")
MODEL_NOTAS     = os.getenv("OPENAI_MODEL_NOTAS", MODEL_ANALISIS)
MODEL_SINTESIS  = os.getenv("OPENAI_MODEL_SINTESIS", MODEL_ANALISIS)
FAST_FORCE_MODEL = os.getenv("FAST_FORCE_MODEL", "").strip()  # opcional para fast

MAX_SINGLE_PASS_CHARS = int(os.getenv("MAX_SINGLE_PASS_CHARS", "120000"))
MAX_SINGLE_PASS_CHARS_MULTI = int(os.getenv("MAX_SINGLE_PASS_CHARS_MULTI", str(MAX_SINGLE_PASS_CHARS)))

CHUNK_SIZE_BASE = int(os.getenv("CHUNK_SIZE", "24000"))
TARGET_PARTS = int(os.getenv("TARGET_PARTS", "2"))
MAX_COMPLETION_TOKENS_SALIDA = int(os.getenv("MAX_COMPLETION_TOKENS_SALIDA", "3500"))
TEMPERATURE_ANALISIS = os.getenv("TEMPERATURE_ANALISIS", "").strip()
ANALISIS_MODO = os.getenv("ANALISIS_MODO", "").lower().strip()  # "fast" opcional

# Granularidad / anti-copia ligera (sin perder cobertura)
RENGLON_DESC_MAX_WORDS = int(os.getenv("RENGLON_DESC_MAX_WORDS", "24"))
ART_SNIPPET_MAX_WORDS  = int(os.getenv("ART_SNIPPET_MAX_WORDS", "18"))

# Concurrencia
ANALISIS_CONCURRENCY = int(os.getenv("ANALISIS_CONCURRENCY", "3"))
NOTAS_MAX_TOKENS = int(os.getenv("NOTAS_MAX_TOKENS", "1400"))

# OCR
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "8"))
VISION_DPI = int(os.getenv("VISION_DPI", "150"))
OCR_TEXT_MIN_CHARS = int(os.getenv("OCR_TEXT_MIN_CHARS", "120"))
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))

# Control de paginado en texto nativo
PAGINAR_TEXTO_NATIVO = int(os.getenv("PAGINAR_TEXTO_NATIVO", "1"))

# Calidad/recall
MULTI_FORCE_TWO_STAGE_MIN_CHARS = int(os.getenv("MULTI_FORCE_TWO_STAGE_MIN_CHARS", "45000"))
ENABLE_REGEX_HINTS = int(os.getenv("ENABLE_REGEX_HINTS", "1"))
HINTS_MAX_CHARS = int(os.getenv("HINTS_MAX_CHARS", "12000"))
HINTS_PER_FIELD = int(os.getenv("HINTS_PER_FIELD", "8"))
ENABLE_SECOND_PASS_COMPLETION = int(os.getenv("ENABLE_SECOND_PASS_COMPLETION", "1"))

# Limitar enumeraciones y desactivar expansiones automáticas
EXPAND_SECTIONS_213_216 = int(os.getenv("EXPAND_SECTIONS_213_216", "0"))
MAX_RENGLONES_OUT       = int(os.getenv("MAX_RENGLONES_OUT", "12"))
MAX_ARTICULOS_OUT       = int(os.getenv("MAX_ARTICULOS_OUT", "12"))

# Forzar reemplazo determinístico de 2.13 y 2.16 (cobertura total)
FORCE_DETERMINISTIC_213_216 = int(os.getenv("FORCE_DETERMINISTIC_213_216", "0"))

# ========================= Timers PERF =========================
def _t(): return time.perf_counter()
def _log_tiempo(etiqueta, t0):
    try:
        dt = time.perf_counter() - t0
        print(f"[PERF] {etiqueta}: {dt:0.2f}s")
    except Exception:
        pass

# ==================== OCR / Raster ====================
def _rasterizar_pagina(page, dpi=VISION_DPI) -> bytes:
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def _ocr_openai_imagen_b64(b64_png: str) -> str:
    prompt = (
        "Extraé el TEXTO literal de esta imagen escaneada de un pliego. "
        "Conservá títulos, tablas como líneas con separadores, listas y números. No resumas ni interpretes."
    )
    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_png}"}}
                ]
            }],
            # Chat Completions usa 'max_tokens' (no 'max_completion_tokens')
            max_tokens=2400,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[OCR-ERROR] {e}"

# ---- OCR selectivo en paralelo (muestreo uniforme en el doc) ----
from concurrent.futures import ThreadPoolExecutor, as_completed

def _ocr_pagina_png_bytes(png_bytes: bytes, idx: int) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    txt = _ocr_openai_imagen_b64(b64)
    return f"[PÁGINA {idx+1}]\n{txt}" if txt else f"[PÁGINA {idx+1}] (sin texto OCR)"

def _ocr_selectivo_por_pagina(doc: fitz.Document, max_pages: int) -> str:
    """
    Muestrea páginas a lo largo de todo el documento para no perder planillas al final.
    """
    n = len(doc)
    if n == 0:
        return ""
    to_process = min(n, max_pages)

    if to_process >= n:
        page_idxs = list(range(n))
    else:
        page_idxs = sorted({int(round(i * (n - 1) / max(1, to_process - 1))) for i in range(to_process)})

    resultados_map: Dict[int, str] = {}

    def _proc_page(i: int) -> Tuple[int, str]:
        p = doc.load_page(i)
        txt_nat = (p.get_text() or "").strip()
        if len(txt_nat) >= OCR_TEXT_MIN_CHARS:
            return i, f"[PÁGINA {i+1}]\n{txt_nat}"
        png_bytes = _rasterizar_pagina(p)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        txt = _ocr_openai_imagen_b64(b64)
        return i, (f"[PÁGINA {i+1}]\n{txt}" if txt else f"[PÁGINA {i+1}] (sin texto OCR)")

    with ThreadPoolExecutor(max_workers=OCR_CONCURRENCY) as ex:
        futs = [ex.submit(_proc_page, i) for i in page_idxs]
        for fut in as_completed(futs):
            try:
                i, s = fut.result()
                resultados_map[i] = s
            except Exception:
                pass

    orden = sorted(resultados_map.keys())
    res = [resultados_map[i] for i in orden]
    if n > to_process:
        res.append(f"\n[AVISO] OCR muestreó {to_process}/{n} páginas distribuidas.")
    return "\n\n".join([r for r in res if r]).strip()
# utils.py — Parte 2/5

# ==================== Extracción por tipo de archivo ====================
def _leer_todo(file) -> bytes:
    try:
        file.file.seek(0)
        raw = file.file.read()
    except Exception:
        try:
            raw = file.read()
        except Exception:
            raw = b""
    return raw or b""

def _ext_de_archivo(file) -> str:
    nombre = getattr(file, "filename", "") or ""
    _, ext = os.path.splitext(nombre)
    return (ext or "").lower().strip()

def _mime_guess(file) -> str:
    nombre = getattr(file, "filename", "") or ""
    m, _ = mimetypes.guess_type(nombre)
    return m or ""

def _texto_nativo_etiquetado(doc: fitz.Document) -> str:
    partes = []
    for i, p in enumerate(doc, 1):
        t = (p.get_text() or "").strip()
        if t:
            partes.append(f"[PÁGINA {i}]\n{t}")
        else:
            partes.append(f"[PÁGINA {i}] (sin texto)")
    return "\n\n".join(partes).strip()

def extraer_texto_de_pdf(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_pdf_sin_bytes", t0); return ""
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            suma = 0
            for p in doc:
                suma += len((p.get_text() or "").strip())
            if suma < 500:
                ocr_t0 = _t()
                ocr_text = _ocr_selectivo_por_pagina(doc, VISION_MAX_PAGES)
                _log_tiempo("ocr_selectivo", ocr_t0)
                _log_tiempo("extraccion_pdf_total", t0)
                return ocr_text
            out = _texto_nativo_etiquetado(doc) if PAGINAR_TEXTO_NATIVO else "\n".join([p.get_text() or "" for p in doc])
            _log_tiempo("extraccion_pdf_total", t0)
            return out.strip()
    except Exception:
        try:
            out = raw.decode("utf-8", errors="ignore")
            _log_tiempo("extraccion_pdf_decode", t0)
            return out
        except Exception:
            _log_tiempo("extraccion_pdf_error", t0)
            return ""

def extraer_texto_de_docx(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_docx_sin_bytes", t0); return ""
    if docx is None:
        try:
            out = raw.decode("utf-8", errors="ignore")
            _log_tiempo("extraccion_docx_decode", t0)
            return out
        except Exception:
            _log_tiempo("extraccion_docx_error", t0); return ""
    try:
        document = docx.Document(io.BytesIO(raw))
        partes: List[str] = []
        for p in document.paragraphs:
            txt = (p.text or "").strip()
            if txt:
                partes.append(txt)
        for tbl in document.tables:
            for row in tbl.rows:
                celdas = [(cell.text or "").strip() for cell in row.cells]
                partes.append(" | ".join(celdas))
        out = "\n".join(partes).strip()
        _log_tiempo("extraccion_docx_total", t0)
        return out
    except Exception:
        try:
            out = raw.decode("utf-8", errors="ignore")
            _log_tiempo("extraccion_docx_decode_fallback", t0)
            return out
        except Exception:
            _log_tiempo("extraccion_docx_error", t0); return ""

def extraer_texto_de_imagen(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_imagen_sin_bytes", t0); return ""
    try:
        img_doc = fitz.open(stream=raw, filetype=_ext_de_archivo(file).lstrip(".") or None)
        page = img_doc.load_page(0)
        png = page.get_pixmap(alpha=False).tobytes("png")
        b64 = base64.b64encode(png).decode("utf-8")
    except Exception:
        b64 = base64.b64encode(raw).decode("utf-8")
    out = _ocr_openai_imagen_b64(b64)
    _log_tiempo("extraccion_imagen_ocr", t0)
    return out

def extraer_texto_universal(file) -> str:
    t0 = _t()
    ext = _ext_de_archivo(file)
    mime = _mime_guess(file)

    if ext == ".pdf" or (mime == "application/pdf"):
        out = extraer_texto_de_pdf(file); _log_tiempo("extraer_texto_universal_pdf", t0); return out
    if ext == ".docx" or (mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]):
        out = extraer_texto_de_docx(file); _log_tiempo("extraer_texto_universal_docx", t0); return out
    if ext in [".png", ".jpg", ".jpeg", ".webp"] or (mime.startswith("image/") if mime else False):
        out = extraer_texto_de_imagen(file); _log_tiempo("extraer_texto_universal_imagen", t0); return out

    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraer_texto_universal_sin_bytes", t0)
        return ""
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    if ext == ".rtf":
        text = re.sub(r"{\\rtf1.*?\\viewkind4\\uc1", "", text, flags=re.S)
        text = re.sub(r"\\[a-z]+-?\d* ?", "", text)
        text = text.replace("{", "").replace("}", "")
    out = (text or "").strip()
    _log_tiempo("extraer_texto_universal_texto_plano", t0)
    return out

# ==================== Pre-limpieza ====================
def _limpieza_basica_preanalisis(s: str) -> str:
    s = re.sub(r"\n?P[aá]gina\s+\d+\s+de\s+\d+\s*\n", "\n", s, flags=re.I)
    s = re.sub(r"\n[-_]{3,}\n", "\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()
# utils.py — Parte 3/5

# ==================== Filtrado de meta-frases y utilidades ====================
_META_PATTERNS = [
    re.compile(r"(?i)\bparte\s+\d+\s+de\s+\d+"),
    re.compile(r"(?i)informe\s+basado\s+en\s+la\s+parte"),
    re.compile(r"(?i)revise\s+las\s+partes\s+restantes"),
    re.compile(r"(?i)informaci[oó]n\s+puede\s+estar\s+incompleta"),
    re.compile(r"(?i)^\s*informe\s+completo\s*$"),
    re.compile(r"(?i)^\s*informe\s+original\s*$"),
]

def _limpiar_meta(texto: str) -> str:
    lineas = []
    for ln in (texto or "").splitlines():
        if any(p.search(ln) for p in _META_PATTERNS):
            continue
        lineas.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()

def _particionar(texto: str, max_chars: int) -> list[str]:
    return [texto[i:i + max_chars] for i in range(0, len(texto or ""), max_chars)]

# =============== Indices de anexos y paginas ===============
_ANEXO_RE = re.compile(r"(?im)^===\s*ANEXO\s+(\d+)")
def _contar_anexos(s: str) -> int:
    return len(_ANEXO_RE.findall(s or ""))

_PAG_TAG_RE = re.compile(r"\[PÁGINA\s+(\d+)\]")

def _index_paginas(s: str) -> List[Tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in _PAG_TAG_RE.finditer(s or "")]

def _pagina_de_indice(indices: List[Tuple[int, int]], pos: int) -> int:
    last = 1
    for i, p in indices:
        if i <= pos:
            last = p
        else:
            break
    return last

def _index_anexos(s: str) -> List[Tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in _ANEXO_RE.finditer(s or "")]

def _anexo_en_pos(indices: List[Tuple[int, int]], pos: int) -> Optional[int]:
    last = None
    for i, a in indices:
        if i <= pos:
            last = a
        else:
            break
    return last

# =============== Normalizacion de citas segun modo (multi vs unico) ===============
_CITA_ANEXO_RE = re.compile(r"\(Anexo\s+([IVXLCDM\d]+)(?:,\s*p\.\s*(\d+))?\)", re.I)

def _normalize_citas_salida(texto: str, varios_anexos: bool) -> str:
    if varios_anexos:
        return texto or ""
    # Si es documento unico, convertir "(Anexo X, p. N)" en "(p. N)" o fuente generica
    def repl(m):
        pag = m.group(2)
        if pag:
            return f"(p. {pag})"
        return "(Fuente: documento provisto)"
    return _CITA_ANEXO_RE.sub(repl, texto or "")

# ==================== Normalizacion para PDF (sin markdown) ====================
_HDR_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```.*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_ITALIC_RE = re.compile(r"(\*\*|\*|__|_)(.*?)\1")

def _title_case(s: str) -> str:
    return " ".join(w.capitalize() if w else w for w in re.split(r"(\s+)", s or ""))

def preparar_texto_para_pdf(markdown_text: str) -> str:
    out_lines: List[str] = []
    for raw_ln in (markdown_text or "").splitlines():
        ln = raw_ln.rstrip()

        # Normaliza encabezado de Ficha si el modelo lo numeró como "0)"
        if re.match(r"^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento\b", ln, flags=re.I):
            ln = "Ficha estandarizada del procedimiento (campos estandarizados)"

        if _CODE_FENCE_RE.match(ln):
            continue
        # filtra titulos indeseados
        if re.match(r"(?i)^\s*informe\s+completo\s*$", ln):
            continue
        if re.match(r"(?i)^\s*informe\s+original\s*$", ln):
            continue

        m = _HDR_RE.match(ln)
        if m:
            titulo = _title_case(m.group(2).strip(": ").strip())
            out_lines.append(titulo)
            out_lines.append("")  # espacio tras titulo
            continue

        if _TABLE_SEP_RE.match(ln):
            continue

        if _BULLET_RE.match(ln):
            ln = _BULLET_RE.sub("• ", ln)

        ln = _LINK_RE.sub(lambda mm: f"{mm.group(1)} ({mm.group(2)})", ln)
        ln = _BOLD_ITALIC_RE.sub(lambda mm: mm.group(2), ln)
        out_lines.append(ln)

        if ln.strip().endswith(":"):
            out_lines.append("")  # espacio extra tras linea-titulo

    texto = "\n".join(out_lines)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto

# ==================== Hints regex (recall) ====================
def _buscar_candidatos(texto: str, pats: List[str], idx_pag: List[Tuple[int, int]], limit: int) -> List[str]:
    hits = []
    for pat in pats:
        for m in re.finditer(pat, texto or "", flags=re.I):
            pos = m.start()
            p = _pagina_de_indice(idx_pag, pos)
            start = max(0, pos - 160)
            end = min(len(texto), pos + 240)
            snippet = (texto[start:end]).replace("\n", " ").strip()
            hits.append(f"- p. {p}: {snippet}")
            if len(hits) >= limit:
                return hits
    return hits[:limit]

def _build_regex_hints(texto: str, limit_per_field: int = None, max_chars: int = None) -> str:
    if not texto:
        return ""
    if limit_per_field is None:
        limit_per_field = HINTS_PER_FIELD
    if max_chars is None:
        max_chars = HINTS_MAX_CHARS
    idx_pag = _index_paginas(texto)
    secciones = []
    for key, meta in DETECTABLE_FIELDS.items():
        hits = _buscar_candidatos(texto, meta["pats"], idx_pag, limit_per_field)
        if hits:
            secciones.append(f"[{meta['label']}]\n" + "\n".join(hits))
        if sum(len(s) for s in secciones) > max_chars:
            break
    return "\n\n".join(secciones[:])

# Campos detectables (ampliados y adaptados a AR)
DETECTABLE_FIELDS: Dict[str, Dict] = {
    "mant_oferta": {"label": "Mantenimiento de oferta", "pats": [r"mantenim[ií]ento de la oferta", r"validez de la oferta"]},
    "gar_mant":    {"label": "Garantia de mantenimiento", "pats": [r"garant[ií]a.*manten", r"\b5 ?%"]},
    "gar_cumpl":   {"label": "Garantia de cumplimiento", "pats": [r"garant[ií]a.*cumpl", r"\b10 ?%"]},
    "plazo_ent":   {"label": "Plazo de entrega", "pats": [r"plazo de entrega", r"\b\d{1,3}\s*d[ií]as"]},
    "tipo_cambio": {"label": "Tipo de cambio", "pats": [r"Banco\s+Naci[oó]n", r"tipo de cambio", r"BNA"]},
    "comision":    {"label": "Comision de (Pre)?Adjudicacion", "pats": [r"Comisi[oó]n.*(pre)?adjudicaci[oó]n"]},
    "muestras":    {"label": "Muestras", "pats": [r"\bmuestras?\b"]},
    "planilla":    {"label": "Planilla de cotizacion y renglones", "pats": [r"planilla.*cotizaci[oó]n", r"renglones?"]},
    "modalidad":   {"label": "Procedimiento/Modalidad", "pats": [r"licitaci[oó]n\s+(p[úu]blica|privada)", r"contrataci[oó]n\s+directa", r"compra\s+menor", r"subasta", r"modalidad"]},
    "plazo_contr": {"label": "Duracion del contrato", "pats": [r"duraci[oó]n del contrato", r"plazo contractual", r"por el t[eé]rmino\s+de\s+\d+", r"\b\d{1,4}\s*d[ií]as"]},
    "prorroga":    {"label": "Prorroga/Ampliacion", "pats": [r"pr[oó]rroga", r"ampliaci[oó]n", r"hasta\s+el\s+100%"]},
    "presupuesto": {"label": "Monto / Presupuesto", "pats": [r"presupuesto (estimado|oficial|referencial)", r"monto\s+estimado", r"cr[eé]dito\s+disponible", r"\$\s?\d{1,3}(\.\d{3})*(,\d{2})?"]},
    "expediente":  {"label": "Expediente / N° proceso", "pats": [r"\bEX-\d{4}-[A-Z0-9-]+", r"\bN[°º]\s*de\s*(proceso|procedimiento|expediente)"]},
    "fechas":      {"label": "Fechas y horas", "pats": [r"\b\d{2}/\d{2}/\d{4}\b", r"\b\d{1,2}:\d{2}\s*(hs|h)"]},
    "contacto":    {"label": "Contactos y portales", "pats": [r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", r"https?://[^\s)]+|www\.[^\s)]+"]},
    "costo_pliego":{"label": "Costo/valor del pliego", "pats": [r"(costo|valor)\s+del\s+pliego", r"adquisici[oó]n\s+del\s+pliego", r"\$\s?\d{1,3}(\.\d{3})*(,\d{2})?"]},
    "subsanacion": {"label": "Subsanacion", "pats": [r"subsanaci[oó]n"]},
    "perf_modif":  {"label": "Perfeccionamiento/Modificaciones", "pats": [r"perfeccionamiento", r"modificaci[oó]n"]},
    "preferencias":{"label": "Preferencias", "pats": [r"preferencias"]},
    "criterios":   {"label": "Criterios de evaluacion", "pats": [r"criterios?\s+de\s+evaluaci[oó]n"]},
    "renglones":   {"label": "Renglones y especificaciones", "pats": [r"Rengl[oó]n\s*\d+", r"Especificaciones?\s+t[ée]cnicas?"]},
    "articulos":   {"label": "Articulos citados", "pats": [r"\bArt(?:[íi]culo|\.)\s*\d+[A-Za-z]?\b"]},
    "estado":      {"label": "Estado del tramite", "pats": [r"\bestado\b", r"\bvigente\b", r"\b(adjudicado|desierto|fracasado|cerrado)\b"]},
    "consultas":   {"label": "Inicio y final de consultas", "pats": [r"\bconsultas\b", r"aclaraciones", r"preguntas"]},
    "apertura":    {"label": "Acto de apertura", "pats": [r"acto\s+de\s+apertura", r"\bapertura\b"]},
    "tipo_cotiz":  {"label": "Tipo de cotizacion", "pats": [r"forma\s+de\s+cotizaci[oó]n", r"tipo\s+de\s+cotizaci[oó]n", r"cotizaci[oó]n\s+por"]},
    "tipo_adj":    {"label": "Tipo de adjudicacion", "pats": [r"adjudicaci[oó]n\s+por\s+(rengl[oó]n|lote|total)"]},
    "moneda":      {"label": "Moneda", "pats": [r"\bmoneda\b", r"\bARS\b", r"\bUSD\b"]},
    "obj_gasto":   {"label": "Objeto del gasto", "pats": [r"objeto\s+del\s+gasto", r"partida\s+presupuestaria", r"clasificador"]},
    "ofertas_perm":{"label": "Ofertas permitidas", "pats": [r"m[aá]s\s+de\s+una\s+oferta", r"ofertas?\s+alternativas", r"una\s+sola\s+oferta"]},
}
# utils.py — Parte 4/5

# ==================== Utilidades de conteo y evidencia ====================
def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text or "", flags=re.I))

_ART_HEAD_RE = re.compile(r"(?im)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.*)$")
_ART_BLOCK_RE = re.compile(
    r"(?ims)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.+?)(?=^\s*art(?:[íi]culo|\.?)\s*\d+[a-zº°]?|\Z)"
)

def _extraer_articulos_con_snippets(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (rotulo_articulo, snippet_200c, pagina_aprox, anexo_num).
    """
    texto = texto or ""
    idx = _index_paginas(texto)
    idx_ax = _index_anexos(texto)
    res: List[Tuple[str, str, int, Optional[int]]] = []

    for m in _ART_BLOCK_RE.finditer(texto):
        start = m.start()
        p = _pagina_de_indice(idx, start)
        ax = _anexo_en_pos(idx_ax, start)
        rotulo = (m.group(1) or "").strip()
        contenido = (m.group(2) or "").strip()
        snippet = contenido[:200].replace("\n", " ").strip()
        res.append((rotulo, snippet, p, ax))

    if not res:
        for m in _ART_HEAD_RE.finditer(texto):
            start = m.start()
            p = _pagina_de_indice(idx, start)
            ax = _anexo_en_pos(idx_ax, start)
            rotulo = (m.group(1) or "").strip()
            snippet = ((m.group(2) or "").strip())[:200].replace("\n", " ")
            res.append((rotulo, snippet, p, ax))

    return res

# --- Renglones robustos (exigir literalmente "Renglón" o variantes) ---
_ROW_START_RE = re.compile(r"(?im)^(?:reng(?:l[oó]n)?\.?\s*)(\d{1,4})\b")
_CODE_RE = re.compile(r"\b[A-Z]{1,3}\d{5,8}\b")  # ej: D0330113, GB079001, E5001253
_QTY_RE = re.compile(r"\b\d{1,6}\b")

def _extraer_renglones_y_especificaciones(texto: str) -> List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]]:
    """
    Devuelve lista: (num_renglon, cantidad, codigo, descripcion_full, pagina_aprox, anexo_num)
    - Reconoce filas numeradas que empiezan con "Renglón".
    - Agrega lineas subsiguientes hasta el siguiente comienzo de fila.
    """
    texto = texto or ""
    idx = _index_paginas(texto)
    idx_ax = _index_anexos(texto)
    res: List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]] = []

    lines = texto.splitlines()
    pos = 0
    starts: List[Tuple[int, int]] = []  # (line_index, abs_pos)
    for i, ln in enumerate(lines):
        m = _ROW_START_RE.match(ln)
        if m:
            starts.append((i, pos))
        pos += len(ln) + 1

    if not starts:
        return res

    # Sentinel
    starts.append((len(lines), len(texto)))

    for k in range(len(starts) - 1):
        i_line, abs_pos = starts[k]
        j_line, _abs_pos_next = starts[k + 1]
        block_lines = lines[i_line:j_line]
        block_text = " ".join([re.sub(r"\s+", " ", x).strip() for x in block_lines if x.strip()])

        # numero de renglon
        mnum = _ROW_START_RE.match(lines[i_line])
        try:
            num_r = int(mnum.group(1)) if mnum else None
        except Exception:
            num_r = None

        # cantidad (primer entero de la linea tras el numero)
        qty = None
        if mnum:
            tail = lines[i_line][mnum.end():]
            mqty = _QTY_RE.search(tail)
            if mqty:
                try:
                    qty = int(mqty.group(0))
                except Exception:
                    qty = None

        # codigo (en todo el bloque)
        mcode = _CODE_RE.search(block_text)
        code = mcode.group(0) if mcode else None

        # descripcion y especificaciones
        desc = block_text
        if code:
            desc = re.sub(re.escape(code), "", desc)
        if qty is not None:
            desc = re.sub(rf"\b{qty}\b", "", desc)
        if num_r is not None:
            desc = re.sub(rf"^\s*{num_r}\b", "", desc)
        desc = re.sub(r"\s+", " ", desc).strip()

        p = _pagina_de_indice(idx, abs_pos)
        ax = _anexo_en_pos(idx_ax, abs_pos)

        if num_r is not None:
            res.append((num_r, qty, code, desc, p, ax))

    res.sort(key=lambda t: t[0])
    return res

def _construir_evidencia_ampliacion(texto: str) -> Tuple[str, int, int]:
    """
    Arma evidencias literales con paginas/anexos para renglones y articulos.
    Devuelve: (bloque_evidencia, cant_renglones, cant_articulos).
    """
    texto = texto or ""
    renglones = _extraer_renglones_y_especificaciones(texto)
    articulos = _extraer_articulos_con_snippets(texto)

    ev_parts: List[str] = []
    if renglones:
        ev = []
        for (num, qty, code, desc, p, ax) in renglones:
            cit = f"(Anexo {ax}, p. {p})" if ax else f"(p. {p})"
            det = []
            if qty is not None:
                det.append(f"cant {qty}")
            if code:
                det.append(f"cod {code}")
            det_txt = " — ".join(det) if det else ""
            linea = f"- Renglon {num}{(' — ' + det_txt) if det_txt else ''}: {desc} {cit}"
            ev.append(linea)
        ev_parts.append("### EVIDENCIA Renglones / Planilla (literal)\n" + "\n".join(ev))
    if articulos:
        ev = []
        for (rot, sn, p, ax) in articulos:
            cit = f"(Anexo {ax}, p. {p})" if ax else f"(p. {p})"
            ev.append(f"- {rot} — {sn} {cit}")
        ev_parts.append("### EVIDENCIA Articulos (literal)\n" + "\n".join(ev))

    return ("\n\n".join(ev_parts) if ev_parts else ""), len(renglones), len(articulos)

def _conteo_en_informe(informe: str) -> Tuple[int, int]:
    informe = informe or ""
    return _count(r"(?im)\brengl[oó]n\s*\d+", informe), _count(r"(?im)\bart(?:[íi]culo|\.?)\s*\d+", informe)

def _max_out_for_text(texto: str) -> int:
    texto = texto or ""
    base_chars = len(texto)
    r_count = _count(r"(?im)^\s*(?:reng(?:l[oó]n)?\.?\s*)?\d{1,4}\b", texto)
    a_count = _count(r"(?im)^\s*art(?:[íi]culo|\.?)\s*\d+", texto)
    base = MAX_COMPLETION_TOKENS_SALIDA
    if r_count >= 20 or a_count >= 20:
        base = max(base, 6500)
    elif r_count >= 8 or a_count >= 8:
        base = max(base, 5000)
    if ANALISIS_MODO == "fast":
        if base_chars < 15000:
            base = max(base, 2800)
        elif base_chars < 40000:
            base = max(base, 3500)
    return int(base)

# ====== Utilidad de compresion no literal ======
def _truncate_words(s: str, max_words: int) -> str:
    try:
        words = re.findall(r"\S+", s or "")
        if len(words) <= max_words:
            return (s or "").strip()
        return " ".join(words[:max_words]).rstrip(",.;:") + "..."
    except Exception:
        return (s or "").strip()

# Palabras-clave para filtrar articulos realmente utiles
_ART_KEYS = re.compile(
    r"(objeto|tipolog|modalidad|mantenim|pr[oó]rroga|oferta|apertura|evaluaci[oó]n|empate|mejora|adjudicaci[oó]n|"
    r"garant[ií]a|entrega|plazo|pago|factura|sancion|penalidad|rescis[ií]n|perfeccionamiento|subsanaci[oó]n)",
    re.I
)

# ====== Generadores deterministicos de 2.13 y 2.16 ======
def _build_section_213(texto: str, varios_anexos: bool) -> str:
    rows = _extraer_renglones_y_especificaciones(texto or "")
    if not rows:
        return ""
    rows = rows[:max(1, MAX_RENGLONES_OUT)]  # tope
    lines = ["2.13 Planilla de cotizacion y renglones:"]
    for (num, qty, code, desc, p, ax) in rows:
        desc_corta = _truncate_words(desc, RENGLON_DESC_MAX_WORDS)
        partes = [f"Renglon {num}"]
        if qty is not None:
            partes.append(f"Cantidad: {qty}")
        if code:
            partes.append(f"Codigo: {code}")
        partes.append(f"Descripcion/Especificaciones: {desc_corta}")
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        lines.append(" - " + " — ".join(partes) + f" {cita}")
    return "\n".join(lines)

def _build_section_216(texto: str, varios_anexos: bool) -> str:
    arts = _extraer_articulos_con_snippets(texto or "")
    if not arts:
        return ""
    # filtrar por relevancia practica
    arts = [(rot, sn, p, ax) for (rot, sn, p, ax) in arts if _ART_KEYS.search(sn or "") or _ART_KEYS.search(rot or "")]
    if not arts:
        return ""
    arts = arts[:max(1, MAX_ARTICULOS_OUT)]  # tope
    lines = ["2.16 Catalogo de articulos citados:"]
    for (rot, sn, p, ax) in arts:
        rot_norm = re.sub(r"(?i)art(?:[íi]culo|\.)\s*", "Art. ", rot or "").strip()
        sn = _truncate_words(sn or "", ART_SNIPPET_MAX_WORDS)
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        lines.append(f" - {rot_norm} — {sn} {cita}")
    return "\n".join(lines)

# ====== Contactos (emails/URLs) con pagina/anexo ======
CONTACT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CONTACT_URL_RE   = re.compile(r"(https?://[^\s)]+|www\.[^\s)]+)")

def _extraer_contactos_con_paginas(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (tipo, valor, p, anexo) con tipo in {"email","url"}.
    """
    texto = texto or ""
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)
    res: List[Tuple[str, str, int, Optional[int]]] = []

    for m in CONTACT_EMAIL_RE.finditer(texto):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos)
        ax = _anexo_en_pos(idx_ax, pos)
        res.append(("email", m.group(0), p, ax))

    for m in CONTACT_URL_RE.finditer(texto):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos)
        ax = _anexo_en_pos(idx_ax, pos)
        v = (m.group(0) or "").rstrip(").,;")
        res.append(("url", v, p, ax))

    # dedupe preservando orden
    seen = set()
    dedup: List[Tuple[str, str, int, Optional[int]]] = []
    for t, v, p, ax in res:
        key = (t, v.lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append((t, v, p, ax))
    return dedup

def _build_section_23(texto: str, varios_anexos: bool) -> str:
    items = _extraer_contactos_con_paginas(texto or "")
    if not items:
        return ""
    out = ["2.3 Contactos y portales:"]
    for (t, v, p, ax) in items:
        etiqueta = "Email" if t == "email" else "URL"
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {etiqueta}: {v} {cita}")
    return "\n".join(out)

# ====== Normativa aplicable ======
NORM_TIPOS = [
    ("Ley",        r"\bLey(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    ("Decreto",    r"\bDecreto(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    ("Resolucion", r"\bResoluci[oó]n(?:\s*(?:Ministerial|Conjunta))?\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
    ("Disposicion",r"\bDisposici[oó]n\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
]
NORM_PATTS = [(tipo, re.compile(patt, re.I)) for (tipo, patt) in NORM_TIPOS]

def _extraer_normativa(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (tipo, numero, p, anexo).
    """
    texto = texto or ""
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)
    res: List[Tuple[str, str, int, Optional[int]]] = []

    for (tipo, cre) in NORM_PATTS:
        for m in cre.finditer(texto):
            pos = m.start()
            p = _pagina_de_indice(idx_pag, pos)
            ax = _anexo_en_pos(idx_ax, pos)
            numero = (m.group(1) or "").strip()
            res.append((tipo, numero, p, ax))

    # dedupe preservando orden
    seen = set()
    dedup: List[Tuple[str, str, int, Optional[int]]] = []
    for t, n, p, ax in res:
        key = (t.lower(), n)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((t, n, p, ax))
    return dedup

def _build_section_215(texto: str, varios_anexos: bool) -> str:
    normas = _extraer_normativa(texto or "")
    if not normas:
        return ""
    out = ["2.15 Normativa aplicable:"]
    for (t, n, p, ax) in normas:
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {t} {n} {cita}")
    return "\n".join(out)

# ====== Reemplazo de secciones en el informe ======
def _find_section_bounds(text: str, header_regex: str) -> Tuple[int, int]:
    """
    Devuelve (start, end) del bloque que inicia en header_regex hasta el proximo encabezado 2.X o fin.
    """
    text = text or ""
    m = re.search(header_regex, text, flags=re.I)
    if not m:
        return (-1, -1)
    start = m.start()
    nxt = re.search(r"(?im)^\s*2\.(1[0-9]|[1-9])\s", text[m.end():])
    if not nxt:
        return (start, len(text))
    return (start, m.end() + nxt.start())

def _replace_section(text: str, header_regex: str, replacement: str) -> str:
    s, e = _find_section_bounds(text or "", header_regex)
    if s == -1:
        return (text or "").rstrip() + "\n\n" + (replacement or "").strip() + "\n"
    return (text or "")[:s] + (replacement or "").strip() + "\n" + (text or "")[e:]

# ==================== Ampliacion / sustitucion de 2.13 y 2.16 ====================
def _ampliar_secciones_especificas(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Por defecto (EXPAND_SECTIONS_213_216=0) NO toca 2.13 ni 2.16.
    Siempre normaliza 2.3 Contactos y 2.15 Normativa a partir de extraccion deterministica.
    """
    out = informe or ""

    # Actualizar deterministico 2.3 y 2.15
    sec23 = _build_section_23(texto_fuente or "", varios_anexos)
    if sec23:
        out = _replace_section(out, r"(?im)^\s*2\.3\s+Contactos", sec23)

    sec215 = _build_section_215(texto_fuente or "", varios_anexos)
    if sec215:
        out = _replace_section(out, r"(?im)^\s*2\.15\s+Normativa", sec215)

    # Si no se solicita expansion de renglones/articulos, retornar
    if not EXPAND_SECTIONS_213_216:
        return out

    # Construir 2.13 y 2.16 con topes y reemplazar
    sec213 = _build_section_213(texto_fuente or "", varios_anexos)
    if sec213:
        alt213 = sec213.replace("2.13 Planilla de cotizacion y renglones:", "9) Renglones y planilla de cotizacion:")
        out = _replace_section(out, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla", alt213)
        out = _replace_section(out, r"(?im)^\s*2\.13\s+Planilla", sec213)

    sec216 = _build_section_216(texto_fuente or "", varios_anexos)
    if sec216:
        out = _replace_section(out, r"(?im)^\s*2\.16\s+Cat[aá]logo\s+de\s+art", sec216)
        out = re.sub(r"(?im)^\s*(ANEXO|Anexo)\s*[-–—]?\s*Cat[aá]logo\s+de\s+art[^\n]*\n?", "", out)

    out = re.sub(r"(?im)^\s*informe\s+original\s*$", "", out)
    return out

# === Post-proceso de Ficha (reparaciones determinísticas) ===
def _reparar_ficha(informe: str, texto_fuente: str) -> str:
    """
    Corrige campos de la Ficha que a veces quedan como placeholders:
    - 'Total de renglones: N' -> cuenta real de renglones detectados
    - 'Monto: $...'          -> 'NO ESPECIFICADO' (si el modelo dejó puntos/suspensivos)
    """
    try:
        total_renglones = len(_extraer_renglones_y_especificaciones(texto_fuente or ""))
    except Exception:
        total_renglones = 0

    if total_renglones:
        # Reemplaza cualquier línea de 'Número de renglón' por el total real
        informe = re.sub(
            r"(?im)^(\s*•\s*(?:N[uú]mero\s+de\s+rengl[oó]n|Numero\s+de\s+renglon)\s*:\s*)[^\n]*$",
            lambda m: f"{m.group(1)}Total de renglones: {total_renglones}; ver Seccion 9 para el detalle completo",
            informe or ""
        )
        # Si en otro lado quedó 'Total de renglones: N', reemplaza la N
        informe = re.sub(
            r"(?im)\bTotal de renglones:\s*N\b",
            f"Total de renglones: {total_renglones}",
            informe or ""
        )

    # Normaliza placeholder de monto tipo '$...' a 'NO ESPECIFICADO' (preservando cita si existe)
    informe = re.sub(
        r"(?im)^(\s*•\s*Monto:\s*)(?:\$+\s*\.{0,3}|[$…]+)\s*(\(.*?\))?\s*$",
        lambda m: f"{m.group(1)}NO ESPECIFICADO{(' ' + m.group(2) if m.group(2) else '')}",
        informe or ""
    )

    return (informe or "")
# utils.py — Parte 5/5

# ==================== Llamada a OpenAI robusta ====================
def _max_tokens_salida_adaptivo(longitud_chars: int) -> int:
    base = MAX_COMPLETION_TOKENS_SALIDA
    if ANALISIS_MODO != "fast":
        return base
    if longitud_chars < 15000:
        return min(base, 2200)
    if longitud_chars < 40000:
        return min(base, 2800)
    return base

def _pick_model(stage_default: str) -> str:
    if ANALISIS_MODO == "fast" and FAST_FORCE_MODEL:
        return FAST_FORCE_MODEL
    if stage_default == "notas":
        return MODEL_NOTAS
    if stage_default == "sintesis":
        return MODEL_SINTESIS
    return MODEL_ANALISIS

def _llamada_openai(messages, model=None, temperature_str=TEMPERATURE_ANALISIS,
                    max_completion_tokens=None, retries=2, fallback_model="gpt-5-mini"):
    mdl = model or _pick_model("analisis")

    def _build_kwargs(m):
        kw = dict(
            model=m,
            messages=messages,
            max_completion_tokens=max_completion_tokens or MAX_COMPLETION_TOKENS_SALIDA
        )
        if ANALISIS_MODO == "fast":
            kw["temperature"] = 0
        elif temperature_str != "":
            try:
                kw["temperature"] = float(temperature_str)
            except Exception:
                pass
        return kw

    models_to_try = [mdl]
    if fallback_model and fallback_model != mdl:
        models_to_try.append(fallback_model)

    last_error = None
    for m in models_to_try:
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(**_build_kwargs(m))
                if not getattr(resp, "choices", None):
                    raise RuntimeError("El modelo no devolvió 'choices'.")
                content = (resp.choices[0].message.content or "").strip()
                if not content:
                    raise RuntimeError("La respuesta del modelo llegó vacía.")
                return resp
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(1.2 * (attempt + 1))
                else:
                    break
    raise RuntimeError(str(last_error) if last_error else "Fallo en _llamada_openai")

# ==================== Concurrencia para NOTAS ====================
def _compute_chunk_size(total_chars: int) -> int:
    if TARGET_PARTS <= 0:
        return CHUNK_SIZE_BASE
    ideal = (total_chars + TARGET_PARTS - 1) // TARGET_PARTS
    return max(CHUNK_SIZE_BASE, ideal)

def _generar_notas_concurrente(partes: List[str]) -> List[str]:
    resultados: List[Optional[str]] = [None] * len(partes)
    t0 = _t()

    def worker(idx: int, parte: str):
        msg = [
            {"role": "system",
             "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user",
             "content": f"{CRAFT_PROMPT_NOTAS}\n\n## Guía de sinónimos/normalización\n{SINONIMOS_CANONICOS}\n\n=== FRAGMENTO {idx+1}/{len(partes)} ===\n{parte}"}
        ]
        r = _llamada_openai(msg, max_completion_tokens=NOTAS_MAX_TOKENS, model=_pick_model("notas"))
        return idx, (r.choices[0].message.content or "").strip()

    with ThreadPoolExecutor(max_workers=max(1, ANALISIS_CONCURRENCY)) as ex:
        futs = [ex.submit(worker, i, p) for i, p in enumerate(partes)]
        for fut in as_completed(futs):
            try:
                i, content = fut.result()
                resultados[i] = content
            except Exception as e:
                resultados[i] = f"[ERROR] No se pudieron generar notas de la parte {i+1}: {e}"

    _log_tiempo(f"notas_intermedias_{len(partes)}_partes_concurrente", t0)
    return [r or "" for r in resultados]

# ==================== Segundo pase (focalizado) ====================
_NOESP_RE = re.compile(r"(?i)\bNO ESPECIFICADO\b")

def _segundo_pase_si_falta(original_report: str, texto_fuente: str, varios_anexos: bool) -> str:
    if not ENABLE_SECOND_PASS_COMPLETION:
        return original_report
    if not _NOESP_RE.search(original_report or ""):
        return original_report

    evidencia: List[str] = []
    for clave, meta in DETECTABLE_FIELDS.items():
        label = meta["label"]
        if re.search(rf"{re.escape(label)}.*NO ESPECIFICADO", original_report or "", flags=re.I) or \
           re.search(rf"{re.escape(label)}\s*:\s*NO ESPECIFICADO", original_report or "", flags=re.I):
            hits = _buscar_candidatos(texto_fuente or "", meta["pats"], _index_paginas(texto_fuente or ""), 10)
            if hits:
                evidencia.append(f"### {label}\n" + "\n".join(hits))

    if not evidencia:
        return original_report

    prompt_corr = f"""
(Revision focalizada) Completa UNICAMENTE los campos marcados como "NO ESPECIFICADO" en el informe,
usando SOLO la evidencia literal que te paso abajo. Mantiene exactamente la estructura y secciones del
informe original, sin agregar nuevas secciones. Donde la evidencia sea ambigua, deja "NO ESPECIFICADO".
Respeta las reglas de citas del informe original (usa (Anexo X, p. N) o (p. N) segun corresponda).
NO imprimas rotulos como 'Informe Original'.

=== CONTENIDO A CORREGIR ===
{original_report}

=== EVIDENCIA LITERAL (snippets con paginas) ===
{'\n\n'.join(evidencia)}
"""
    try:
        resp = _llamada_openai(
            [{"role": "system", "content": "Redactor técnico-jurídico. Cero invenciones."},
             {"role": "user", "content": prompt_corr}],
            model=_pick_model("sintesis"),
            max_completion_tokens=MAX_COMPLETION_TOKENS_SALIDA
        )
        corregido = (resp.choices[0].message.content or "").strip()
        corregido = _normalize_citas_salida(_limpiar_meta(corregido), varios_anexos)
        corregido = re.sub(r"(?im)^\s*informe\s+original\s*$", "", corregido)
        return corregido
    except Exception:
        return original_report

# ==================== Normalizaciones finales de salida ====================
def _normalizar_encabezados_salida(informe: str) -> str:
    """
    - Elimina el prefijo '0) ' en la Ficha estandarizada.
    - Limpia duplicados o espacios errantes.
    """
    s = informe or ""
    s = re.sub(
        r"(?im)^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento\s*\(campos\s+estandarizados\)\s*$",
        "Ficha estandarizada del procedimiento (campos estandarizados)", s
    )
    s = re.sub(
        r"(?im)^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento\s*$",
        "Ficha estandarizada del procedimiento", s
    )
    return s.strip()

def _corregir_seccion_9_si_vacia(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Si la sección 9) quedó vacía, genérica o sin renglones, la reemplaza
    por una versión determinística construida desde el texto fuente.
    """
    s = informe or ""
    start, end = _find_section_bounds(s, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla")
    needs_fix = False

    if start == -1:
        needs_fix = True
    else:
        bloque = s[start:end]
        if (_count(r"(?im)\bRengl[oó]n\s+\d+", bloque) == 0) or \
           (_NOESP_RE.search(bloque) is not None) or \
           (len(bloque.strip()) < 80):
            needs_fix = True

    if not needs_fix:
        return s

    sec213 = _build_section_213(texto_fuente or "", varios_anexos)
    if not sec213:
        return s

    sec9 = sec213.replace("2.13 Planilla de cotizacion y renglones:", "9) Renglones y planilla de cotizacion:")
    if start == -1:
        return (s.rstrip() + "\n\n" + sec9.strip() + "\n")
    else:
        return _replace_section(s, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla", sec9)

# ==================== Analizador principal ====================
def analizar_con_openai(texto: str) -> str:
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    texto_len = len(texto)
    n_anexos = _contar_anexos(texto)
    varios_anexos = n_anexos >= 2
    prompt_maestro_local = prompt_andres(varios_anexos)

    hints = _build_regex_hints(texto) if ENABLE_REGEX_HINTS else ""
    hints_block = f"\n\n=== HALLAZGOS AUTOMATICOS (snippets literales para verificacion) ===\n{hints}\n" if hints else ""

    force_two_stage = (varios_anexos and texto_len >= MULTI_FORCE_TWO_STAGE_MIN_CHARS)

    if (not varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS) or \
       (varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS_MULTI and not force_two_stage):
        t0 = _t()
        max_out = _max_out_for_text(texto)
        messages = [
            {"role": "system",
             "content": "Actúa como equipo experto en derecho administrativo argentino y compras públicas. Redactor técnico-jurídico. Cero invenciones."},
            {"role": "user",
             "content": f"{prompt_maestro_local}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\nDevuelve SOLO el informe final, sin preámbulos."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = (resp.choices[0].message.content or "").strip()
            bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
            bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
            bruto = _ampliar_secciones_especificas(bruto, texto, varios_anexos)
            bruto = _reparar_ficha(bruto, texto)
            bruto = _corregir_seccion_9_si_vacia(bruto, texto, varios_anexos)
            bruto = _normalizar_encabezados_salida(bruto)
            out = preparar_texto_para_pdf(bruto)
            _log_tiempo("analizar_single_pass" + ("_multi" if varios_anexos else ""), t0)
            return out
        except Exception as e:
            return f"Error al generar el análisis: {e}"

    # Dos etapas (chunking + concurrencia)
    chunk_size = _compute_chunk_size(texto_len)
    partes = _particionar(texto, chunk_size)

    if len(partes) == 1:
        t0 = _t()
        max_out = _max_out_for_text(texto)
        messages = [
            {"role": "system",
             "content": "Actúa como equipo experto en derecho administrativo argentino y compras públicas. Redactor técnico-jurídico. Cero invenciones."},
            {"role": "user",
             "content": f"{prompt_maestro_local}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\nDevuelve SOLO el informe final, sin preámbulos."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = (resp.choices[0].message.content or "").strip()
            bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
            bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
            bruto = _ampliar_secciones_especificas(bruto, texto, varios_anexos)
            bruto = _reparar_ficha(bruto, texto)
            bruto = _corregir_seccion_9_si_vacia(bruto, texto, varios_anexos)
            bruto = _normalizar_encabezados_salida(bruto)
            out = preparar_texto_para_pdf(bruto)
            _log_tiempo("analizar_single_pass_len1", t0)
            return out
        except Exception as e:
            return f"Error al generar el análisis: {e}"

    # A) Notas intermedias (concurrente)
    notas_list = _generar_notas_concurrente(partes)
    notas_integradas = "\n".join(notas_list)

    # B) Síntesis final
    t0_sint = _t()
    max_out = _max_out_for_text(texto)
    messages_final = [
        {"role": "system",
         "content": "Actúa como equipo experto en derecho administrativo argentino y compras públicas. Redactor técnico-jurídico. Cero invenciones."},
        {"role": "user",
         "content": f"""{prompt_andres(varios_anexos)}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

{("=== HALLAZGOS AUTOMATICOS (snippets literales) ===\n" + _build_regex_hints(texto)) if ENABLE_REGEX_HINTS else ""}

Integra TODO en un solo informe; deduplica; cita una vez por dato. Prohibido meta-comentarios.
Devuelve SOLO el informe final en texto.
"""}
    ]
    try:
        resp_final = _llamada_openai(messages_final, max_completion_tokens=max_out, model=_pick_model("sintesis"))
        bruto = (resp_final.choices[0].message.content or "").strip()
        bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
        bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
        bruto = _ampliar_secciones_especificas(bruto, texto, varios_anexos)
        bruto = _reparar_ficha(bruto, texto)
        bruto = _corregir_seccion_9_si_vacia(bruto, texto, varios_anexos)
        bruto = _normalizar_encabezados_salida(bruto)
        out = preparar_texto_para_pdf(bruto)
        _log_tiempo("sintesis_final", t0_sint)
        return out
    except Exception as e:
        return f"Error en la síntesis final: {e}\n\nNotas intermedias:\n{_limpiar_meta(notas_integradas)}"

# ==================== Multi-anexo ====================
def analizar_anexos(files: list) -> str:
    """
    Combina anexos y ejecuta análisis.
    - 1 archivo: NO marca '=== ANEXO ... ===' para habilitar single-pass y citas (p. N).
    - >=2: marca ANEXOS para trazabilidad.
    """
    if not files:
        return "No se recibieron anexos para analizar."

    t0 = _t()
    bloques: List[str] = []
    multi = len(files) >= 2

    for idx, f in enumerate(files, 1):
        try:
            texto = extraer_texto_universal(f)
        except Exception:
            try:
                f.file.seek(0)
                texto = f.file.read().decode("utf-8", errors="ignore")
            except Exception:
                texto = ""

        nombre = getattr(f, "filename", f"anexo_{idx}") or f"anexo_{idx}"
        if multi:
            bloques.append(f"=== ANEXO {idx:02d}: {nombre} ===\n{texto}\n")
        else:
            bloques.append(texto)

    contenido_unico = "\n".join(bloques).strip()
    if len(contenido_unico) < 100:
        _log_tiempo("anexos_armado_vacio", t0)
        return ("No se pudo extraer texto útil de los anexos. "
                "Verifica si los documentos están escaneados y eleva VISION_MAX_PAGES/DPI, "
                "o subí archivos en texto nativo.")

    contenido_unico = _limpieza_basica_preanalisis(contenido_unico)
    _log_tiempo("anexos_armado_y_limpieza", t0)

    return analizar_con_openai(contenido_unico)

# ==================== Chat (mejorado con tools/RAG ligero) ====================
MAX_CHAT_CONTEXT_CHARS = int(os.getenv("MAX_CHAT_CONTEXT_CHARS", "38000"))
CHAT_MAX_TOKENS        = int(os.getenv("CHAT_MAX_TOKENS", "1200"))
CHAT_RETRIES           = int(os.getenv("CHAT_RETRIES", "2"))
CHAT_FALLBACK_MODEL    = os.getenv("OPENAI_MODEL_CHAT_FALLBACK", "gpt-5-mini")

def _compactar_contexto_para_chat(contexto: str) -> str:
    s = (contexto or "").strip()
    if len(s) <= MAX_CHAT_CONTEXT_CHARS:
        return s
    head = s[: MAX_CHAT_CONTEXT_CHARS // 3]
    tail = s[- MAX_CHAT_CONTEXT_CHARS // 3 :]
    medio = s[len(s)//2 - MAX_CHAT_CONTEXT_CHARS//6 : len(s)//2 + MAX_CHAT_CONTEXT_CHARS//6]
    return head + "\n\n[...] (contenido intermedio omitido por longitud) [...]\n\n" + medio + \
           "\n\n[...] (contenido intermedio omitido por longitud) [...]\n\n" + tail

def _buscar_en_historial_impl(contexto: str, query: str, k: int = 8, window: int = 280) -> dict:
    texto = contexto or ""
    q = (query or "").strip()
    if not texto or not q:
        return {"hits": []}

    low = texto.lower()
    terms = [t for t in re.findall(r"[a-z0-9áéíóúñ/.-]{3,}", q.lower()) if t not in {"que","con","por","del","para","los","las"}]
    if not terms:
        terms = [q.lower()]

    idx_pag = _index_paginas(texto)
    seen = set()
    hits = []

    for t in terms:
        for m in re.finditer(re.escape(t), low):
            pos = m.start()
            if any(abs(pos - h) < window//2 for h in seen):
                continue
            seen.add(pos)
            start = max(0, pos - window)
            end   = min(len(texto), pos + window)
            snippet = texto[start:end].replace("\n", " ").strip()
            p = _pagina_de_indice(idx_pag, pos) if idx_pag else None
            hits.append({"term": t, "page": p, "snippet": ("..." + snippet + "...")})
            if len(hits) >= k:
                break
        if len(hits) >= k:
            break

    return {"hits": hits}

def responder_chat_openai(mensaje: str, contexto: str = "", usuario: str = "Usuario") -> str:
    """
    Chat con búsqueda ligera en el historial mediante tool-calling.
    """
    contexto_compacto = _compactar_contexto_para_chat(contexto or "(No hay historial disponible.)")

    tools = [{
        "type": "function",
        "function": {
            "name": "buscar_en_historial",
            "description": "Busca evidencia textual en el historial y en informes ya analizados. Devuelve snippets con página.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string", "description": "Consulta o palabras clave a buscar."},
                    "k":      {"type": "integer", "description": "Cantidad máxima de snippets a devolver.", "default": 8}
                },
                "required": ["query"]
            }
        }
    }]

    system_msg = (
        "Eres el asistente del sistema 'Suizo Argentina – Licitaciones IA'. "
        "Respondes con precisión, sin inventar. "
        "Si la pregunta se refiere a pliegos/informes/archivos analizados, "
        "PRIMERO usa la herramienta buscar_en_historial con términos concretos para traer evidencia. "
        "Cita los datos con (p. N) cuando sea posible. "
        "Si no hay evidencia en el material, indícalo explícitamente. "
        "Para preguntas generales, responde breve y claro."
    )

    user_prompt = f"""
Usuario: {usuario}

=== CONTEXTO DISPONIBLE (recortado) ===
{contexto_compacto}

=== PREGUNTA ===
{mensaje}

Instrucciones de salida:
- Si usaste evidencia del historial, menciónala con '(p. N)' cuando tengas la página.
- Si no encontraste nada en el historial, di: "No lo veo en los archivos/historial que tengo" y luego da orientación.
- Nada de meta-texto tipo "parte X/Y". No inventes campos ni datos.
"""

    def _chat_call(model_name: str, msgs: list):
        return client.chat.completions.create(
            model=model_name,
            messages=msgs,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=CHAT_MAX_TOKENS,
            temperature=0.2
        )

    model_primary = os.getenv("OPENAI_MODEL_CHAT", _pick_model("analisis"))
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt}
    ]

    last_error = None
    for model_try in [model_primary, CHAT_FALLBACK_MODEL]:
        if not model_try:
            continue
        for attempt in range(CHAT_RETRIES + 1):
            try:
                resp = _chat_call(model_try, messages)
                choice = resp.choices[0]
                if getattr(choice.message, "tool_calls", None):
                    for tc in choice.message.tool_calls:
                        if tc.function.name == "buscar_en_historial":
                            try:
                                args = json.loads(tc.function.arguments or "{}")
                            except Exception:
                                args = {"query": (mensaje or "")}
                            result = _buscar_en_historial_impl(contexto_compacto, args.get("query", ""), int(args.get("k", 8)))
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": "buscar_en_historial",
                                "content": json.dumps(result, ensure_ascii=False)
                            })
                    resp2 = _chat_call(model_try, messages)
                    out = (resp2.choices[0].message.content or "").strip()
                    if out:
                        return out
                    raise RuntimeError("La respuesta llegó vacía tras tool-calling.")
                else:
                    out = (choice.message.content or "").strip()
                    if out:
                        return out
                    raise RuntimeError("La respuesta llegó vacía.")
            except Exception as e:
                last_error = e
                time.sleep(1.2 * (attempt + 1))

    return (
        "No pude generar respuesta en este momento. "
        f"Detalle técnico: {last_error}"
        if last_error else
        "No pude generar respuesta en este momento."
    )

# ==================== PDF ====================
def _render_pdf_bytes(resumen: str, fecha_display: Optional[str] = None) -> bytes:
    """
    Renderiza el PDF. Si 'fecha_display' viene informada (ej: '31/05/2025 14:03'),
    la usa tal cual. Si no, usa hora local de AR.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    plantilla_path = os.path.join("static", "fondo-pdf.png")
    if os.path.exists(plantilla_path):
        plantilla = ImageReader(plantilla_path)
        c.drawImage(plantilla, 0, 0, width=A4[0], height=A4[1])

    azul = HexColor("#044369")
    c.setFillColor(azul)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(A4[0] / 2, A4[1] - 30 * mm, "Resumen Analítico de Licitación")
    c.setFont("Helvetica", 10)
    c.drawCentredString(A4[0] / 2, A4[1] - 36 * mm, "Inteligencia Comercial")
    c.setFillColor("black")
    c.setFont("Helvetica", 10)

    if not fecha_display:
        try:
            fecha_display = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).strftime("%d/%m/%Y %H:%M")
        except Exception:
            fecha_display = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.drawCentredString(A4[0] / 2, A4[1] - 42 * mm, f"{fecha_display}")

    resumen = (resumen or "").replace("**", "")
    resumen = re.sub(r"(?im)^\s*informe\s+completo\s*$", "", resumen)
    resumen = re.sub(r"(?im)^\s*informe\s+original\s*$", "", resumen)
    resumen = preparar_texto_para_pdf(resumen)

    c.setFont("Helvetica", 11)
    margen_izquierdo = 20 * mm
    margen_superior = A4[1] - 54 * mm
    ancho_texto = 170 * mm
    alto_linea = 14
    y = margen_superior

    for parrafo in resumen.split("\n"):
        if not parrafo.strip():
            y -= alto_linea
            continue
        if parrafo.strip().endswith(":") or parrafo.isupper() or re.match(r"^\d+(\.\d+)*\s", parrafo):
            c.setFont("Helvetica-Bold", 12); c.setFillColor(azul)
        else:
            c.setFont("Helvetica", 11); c.setFillColor("black")
        for linea in dividir_texto(parrafo.strip(), c, ancho_texto):
            if y <= 20 * mm:
                c.showPage()
                if os.path.exists(plantilla_path):
                    c.drawImage(plantilla, 0, 0, width=A4[0], height=A4[1])
                c.setFont("Helvetica", 11); c.setFillColor("black")
                y = margen_superior
            c.drawString(margen_izquierdo, y, linea)
            y -= alto_linea
        if parrafo.strip().endswith(":") or parrafo.isupper() or re.match(r"^\d+(\.\d+)*\s", parrafo):
            y -= alto_linea // 2

    c.save()
    return buffer.getvalue()

def generar_pdf_con_plantilla(resumen: str, nombre_archivo: str, fecha_display: Optional[str] = None):
    """
    Genera el PDF en generated_pdfs/{nombre_archivo}
    """
    output_dir = os.path.join("generated_pdfs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, nombre_archivo)

    data = _render_pdf_bytes(resumen, fecha_display=fecha_display)

    with NamedTemporaryFile(dir=output_dir, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        os.replace(tmp_path, output_path)
    except Exception:
        with open(output_path, "wb") as f:
            f.write(data)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return output_path

def dividir_texto(texto, canvas_obj, max_width):
    palabras = (texto or "").split(" ")
    lineas, linea_actual = [], ""
    for palabra in palabras:
        prueba = (linea_actual + " " + palabra) if linea_actual else palabra
        if canvas_obj.stringWidth(prueba, canvas_obj._fontname, canvas_obj._fontsize) <= max_width:
            linea_actual = prueba
        else:
            if linea_actual:
                lineas.append(linea_actual)
            linea_actual = palabra
    if linea_actual:
        lineas.append(linea_actual)
    return lineas
