# -*- coding: utf-8 -*-
# utils.py — COMPLETO (v2 compacto)
"""
Utilitarios de extracción (PDF/DOCX/Imagen + OCR), normalización y
pipeline de análisis con OpenAI, optimizado para salida COMPACTA.

Env vars más relevantes:
- OPENAI_API_KEY
- OPENAI_MODEL_ANALISIS, OPENAI_MODEL_VISION, OPENAI_MODEL_NOTAS, OPENAI_MODEL_SINTESIS
- ANALISIS_MODO=fast| (vacío)
- COMPACT_MODE=1  (default: 1 -> siempre formato tipo “12:55”)
- DISABLE_TEMPERATURE_PARAM=1  (para modelos gpt-5 que rechazan temperature)
"""

from __future__ import annotations

import io
import os
import re
import base64
import mimetypes
import time
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from zoneinfo import ZoneInfo  # fallback local AR

# ============ prompts centralizados (compat) ============
try:
    import prompts as _prom
except Exception:
    _prom = None

SINONIMOS_CANONICOS = getattr(_prom, "SINONIMOS_CANONICOS", "")

def _reglas_citas(varios_anexos: bool) -> str:
    if _prom and hasattr(_prom, "reglas_citas"):
        try:
            return _prom.reglas_citas(varios_anexos)
        except Exception:
            pass
    if varios_anexos:
        return ("Reglas de Citas:\n"
                "- Documento MULTI-ANEXO: al final de cada línea con dato, usar (Anexo X, p. N).\n"
                "- Si no hay paginación: (Fuente: documento provisto).")
    return ("Reglas de Citas:\n"
            "- Documento ÚNICO: al final de cada línea con dato, usar (p. N) a partir de la etiqueta [PÁGINA N].\n"
            "- Si no hay paginación: (Fuente: documento provisto).")

# ============ PROMPT COMPACTO (layout 1–18 + conclusiones) ============
def prompt_compacto(varios_anexos: bool) -> str:
    return f"""
# (No imprimir este bloque interno)

Objetivo
- Redactar un INFORME TÉCNICO-JURÍDICO compacto, con citas literales (sin inventar).
- Prohibido imprimir "Ficha estandarizada del procedimiento" o checklists ajenos a la estructura.
- Cita al final de cada línea con dato crítico según estas reglas:
{_reglas_citas(varios_anexos)}

Estructura obligatoria EXACTA (encabezados tal cual, sin agregar/renombrar/insertar nuevas secciones):
INFORME TÉCNICO-JURÍDICO SOBRE EL PLIEGO DE BASES Y CONDICIONES – <título del llamado>
1) Identificación del procedimiento y objeto
2) Marco normativo aplicable y régimen general
3) Tipología y modalidad
4) Mantenimiento de la oferta y prórroga automática
5) Presentación de ofertas y contenido mínimo
6) Documentación obligatoria y apercibimiento
7) Apertura, subsanación y causales no subsanables
8) Evaluación, tipo de cambio y Comisión de Preadjudicación
9) Impugnaciones y garantías asociadas
10) Adjudicación
11) Perfeccionamiento del contrato
12) Garantías: mantenimiento de oferta y cumplimiento de contrato
13) Entrega, lugar y plazo
14) Especificaciones técnicas del renglón (si aplica)
15) Facturación, documentación y forma de pago
16) Cotizaciones en moneda extranjera y tipo de cambio
17) Penalidades contractuales y sanciones registrales
18) Observaciones operativas relevantes para oferentes
Conclusiones y recomendaciones

Reglas de redacción
- Máxima concisión. Una o dos viñetas por subtema, con comillas si son literales.
- Si un dato no existe, escribir “NO ESPECIFICADO” (sin inferencias).
- No copiar/pegar bloques de “hallazgos” o “evidencias” que se pasen como contexto.
- No crear secciones fuera de las listadas. No listar “Checklist”, “Fechas y plazos críticos” ni “Ficha…”.
- Mantener terminología del pliego; usar 2 decimales en precios si el pliego lo exige.
- Deduplicar. Prohibido meta-texto (p.ej., “parte X/Y”).
"""

# ============ PROMPT ANDRES (legacy) ============
def prompt_andres(varios_anexos: bool) -> str:
    # usa PROMPT_PARAMETRIZADO si existe; si no, un fallback breve
    if _prom and hasattr(_prom, "PROMPT_PARAMETRIZADO") and hasattr(_prom, "NO_RENGLONES_RULE"):
        try:
            return _prom.PROMPT_PARAMETRIZADO.format(
                REGLAS_CITAS=_reglas_citas(varios_anexos),
                NO_RENGLONES_RULE=getattr(_prom, "NO_RENGLONES_RULE", "")
            )
        except Exception:
            return (_prom.PROMPT_PARAMETRIZADO + "\n\n" + _reglas_citas(varios_anexos)).strip()
    return ("Elabora un informe con trazabilidad y citas (p. N). "
            "Formato claro, sin 'Ficha estandarizada', sin invenciones.")

CRAFT_PROMPT_NOTAS = getattr(
    _prom,
    "CRAFT_PROMPT_NOTAS",
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

# ========================= Compat modelos: no mandar temperature si no corresponde =========================
_DISABLE_TEMP_FLAG = os.getenv("DISABLE_TEMPERATURE_PARAM", "").strip()

def _supports_temperature_param(model_name: str) -> bool:
    try:
        if _DISABLE_TEMP_FLAG == "1":
            return False
        return not bool(re.match(r"(?i)^gpt-5", (model_name or "").strip()))
    except Exception:
        return True

def _normalize_chat_kwargs(**kw):
    if "max_completion_tokens" in kw and "max_tokens" in kw:
        kw.pop("max_tokens", None)
    if kw.get("temperature", None) is None:
        kw.pop("temperature", None)
    return kw

def _chat_create_safe(**kw):
    # limpia None
    if kw.get("temperature", None) is None:
        kw.pop("temperature", None)
    # intentos con max_completion_tokens y luego max_tokens
    tok = kw.pop("max_tokens", kw.pop("max_completion_tokens", None))
    base = dict(kw)
    attempts = []
    if tok is not None:
        attempts.append({**base, "max_completion_tokens": int(tok)})
        attempts.append({**base, "max_tokens": int(tok)})
    else:
        attempts.append(base)

    last_err = None
    for payload in attempts:
        mdl = payload.get("model", "")
        if "temperature" in payload and not _supports_temperature_param(mdl):
            payload.pop("temperature", None)
        try:
            return client.chat.completions.create(**payload)
        except Exception as e:
            last_err = e
            continue

    # intento final sin temperature
    payload = dict(attempts[0])
    payload.pop("temperature", None)
    return client.chat.completions.create(**payload)

# ========================= Modelos / Heurísticas =========================
MODEL_ANALISIS  = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-5")
VISION_MODEL    = os.getenv("OPENAI_MODEL_VISION", "gpt-5")
MODEL_NOTAS     = os.getenv("OPENAI_MODEL_NOTAS", MODEL_ANALISIS)
MODEL_SINTESIS  = os.getenv("OPENAI_MODEL_SINTESIS", MODEL_ANALISIS)
FAST_FORCE_MODEL = os.getenv("FAST_FORCE_MODEL", "").strip()

COMPACT_MODE = int(os.getenv("COMPACT_MODE", "1"))

MAX_SINGLE_PASS_CHARS = int(os.getenv("MAX_SINGLE_PASS_CHARS", "120000"))
MAX_SINGLE_PASS_CHARS_MULTI = int(os.getenv("MAX_SINGLE_PASS_CHARS_MULTI", str(MAX_SINGLE_PASS_CHARS)))
CHUNK_SIZE_BASE = int(os.getenv("CHUNK_SIZE", "24000"))
TARGET_PARTS = int(os.getenv("TARGET_PARTS", "2"))

# Cap de salida general (se ajusta abajo si COMPACT_MODE)
MAX_COMPLETION_TOKENS_SALIDA = int(os.getenv("MAX_COMPLETION_TOKENS_SALIDA", "3500"))
TEMPERATURE_ANALISIS = os.getenv("TEMPERATURE_ANALISIS", "").strip()
ANALISIS_MODO = os.getenv("ANALISIS_MODO", "").lower().strip()  # "fast" opcional

RENGLON_DESC_MAX_WORDS = int(os.getenv("RENGLON_DESC_MAX_WORDS", "24"))
ART_SNIPPET_MAX_WORDS  = int(os.getenv("ART_SNIPPET_MAX_WORDS", "18"))

ANALISIS_CONCURRENCY = int(os.getenv("ANALISIS_CONCURRENCY", "3"))
NOTAS_MAX_TOKENS = int(os.getenv("NOTAS_MAX_TOKENS", "1400"))

VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "8"))
VISION_DPI = int(os.getenv("VISION_DPI", "150"))
OCR_TEXT_MIN_CHARS = int(os.getenv("OCR_TEXT_MIN_CHARS", "120"))
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))

PAGINAR_TEXTO_NATIVO = int(os.getenv("PAGINAR_TEXTO_NATIVO", "1"))

MULTI_FORCE_TWO_STAGE_MIN_CHARS = int(os.getenv("MULTI_FORCE_TWO_STAGE_MIN_CHARS", "45000"))
ENABLE_REGEX_HINTS = int(os.getenv("ENABLE_REGEX_HINTS", "0"))  # OFF por defecto para compactar
HINTS_MAX_CHARS = int(os.getenv("HINTS_MAX_CHARS", "12000"))
HINTS_PER_FIELD = int(os.getenv("HINTS_PER_FIELD", "8"))
ENABLE_SECOND_PASS_COMPLETION = int(os.getenv("ENABLE_SECOND_PASS_COMPLETION", "1"))

# Expansiones opcionales SOLO para modo "andres"
EXPAND_SECTIONS_213_216 = int(os.getenv("EXPAND_SECTIONS_213_216", "0"))
MAX_RENGLONES_OUT = int(os.getenv("MAX_RENGLONES_OUT", "12"))
MAX_ARTICULOS_OUT = int(os.getenv("MAX_ARTICULOS_OUT", "12"))
FORCE_DETERMINISTIC_213_216 = int(os.getenv("FORCE_DETERMINISTIC_213_216", "0"))

# ========================= Timers PERF =========================
def _t() -> float: return time.perf_counter()
def _log_tiempo(etiqueta: str, t0: float) -> None:
    try:
        dt = time.perf_counter() - t0
        print(f"[PERF] {etiqueta}: {dt:0.2f}s")
    except Exception:
        pass

# ==================== OCR / Raster ====================
def _rasterizar_pagina(page: fitz.Page, dpi: int = VISION_DPI) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def _ocr_openai_imagen_b64(b64_img: str, mime: str = "image/png") -> str:
    prompt = (
        "Extraé el TEXTO literal de esta imagen escaneada de un pliego. "
        "Conservá títulos, tablas como líneas con separadores, listas y números. No resumas ni interpretes."
    )
    try:
        resp = _chat_create_safe(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}}
                ]
            }],
            max_tokens=900,
            temperature=None,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[OCR-ERROR] {e}"

def _ocr_selectivo_por_pagina(doc: fitz.Document, max_pages: int) -> str:
    n = len(doc)
    if n == 0: return ""
    to_process = min(n, max_pages)
    if to_process >= n:
        page_idxs = list(range(n))
    else:
        page_idxs = sorted({
            int(round(i * (n - 1) / max(1, to_process - 1)))
            for i in range(to_process)
        })
    resultados_map: Dict[int, str] = {}

    def _proc_page(i: int) -> Tuple[int, str]:
        p = doc.load_page(i)
        txt_nat = (p.get_text() or "").strip()
        if len(txt_nat) >= OCR_TEXT_MIN_CHARS:
            return i, f"[PÁGINA {i+1}]\n{txt_nat}"
        png_bytes = _rasterizar_pagina(p)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        txt = _ocr_openai_imagen_b64(b64, mime="image/png")
        return i, (f"[PÁGINA {i+1}]\n{txt}" if txt else f"[PÁGINA {i+1}] (sin texto OCR)")

    t0 = _t()
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
    _log_tiempo("ocr_selectivo", t0)
    return "\n\n".join([r for r in res if r]).strip()

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
    partes: List[str] = []
    for i, p in enumerate(doc, 1):
        t = (p.get_text() or "").strip()
        if t: partes.append(f"[PÁGINA {i}]\n{t}")
        else: partes.append(f"[PÁGINA {i}] (sin texto)")
    return "\n\n".join(partes).strip()

def extraer_texto_de_pdf(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_pdf_sin_bytes", t0); return ""
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            suma = sum(len((p.get_text() or "").strip()) for p in doc)
            if suma < 500:
                ocr_t0 = _t()
                ocr_text = _ocr_selectivo_por_pagina(doc, VISION_MAX_PAGES)
                _log_tiempo("ocr_selectivo", ocr_t0)
                _log_tiempo("extraccion_pdf_total", t0)
                return ocr_text
            out = _texto_nativo_etiquetado(doc) if PAGINAR_TEXTO_NATIVO else "\n".join([(p.get_text() or "") for p in doc])
            _log_tiempo("extraccion_pdf_total", t0)
            return (out or "").strip()
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
            if txt: partes.append(txt)
        for tbl in document.tables:
            for row in tbl.rows:
                celdas = [(cell.text or "").strip() for cell in row.cells]
                fila = " | ".join([c for c in celdas if c is not None])
                if fila.strip(): partes.append(fila)
        out = "\n".join(partes).strip()
        _log_tiempo("extraccion_docx_total", t0)
        return out
    except Exception:
        try:
            out = raw.decode("utf-8", errors="ignore")
            _log_tiempo("extraccion_docx_decode_fallback", t0)
            return out
        except Exception:
            _log_tiempo("extraccion_docx_error", t0)
            return ""

def extraer_texto_de_imagen(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_imagen_sin_bytes", t0); return ""
    mime_guess = _mime_guess(file) or ""
    ext = _ext_de_archivo(file)
    try:
        img_doc = fitz.open(stream=raw, filetype=ext.lstrip(".") or None)
        page = img_doc.load_page(0)
        png = page.get_pixmap(alpha=False).tobytes("png")
        b64 = base64.b64encode(png).decode("utf-8")
        mime = "image/png"
    except Exception:
        b64 = base64.b64encode(raw).decode("utf-8")
        mime = mime_guess if mime_guess.startswith("image/") else ("image/png" if ext == ".png" else "image/jpeg")
    out = _ocr_openai_imagen_b64(b64, mime=mime)
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
        _log_tiempo("extraer_texto_universal_sin_bytes", t0); return ""
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
    return (s or "").strip()

# ==================== Filtrado de meta-frases ====================
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

# ==================== Índices de anexos y páginas ====================
_ANEXO_RE = re.compile(r"(?im)^===\s*ANEXO\s+(\d+)")
_PAG_TAG_RE = re.compile(r"\[PÁGINA\s+(\d+)\]")

def _contar_anexos(s: str) -> int:
    return len(_ANEXO_RE.findall(s or ""))

def _index_paginas(s: str) -> List[Tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in _PAG_TAG_RE.finditer(s or "")]

def _pagina_de_indice(indices: List[Tuple[int, int]], pos: int) -> int:
    last = 1
    for i, p in indices:
        if i <= pos: last = p
        else: break
    return last

def _index_anexos(s: str) -> List[Tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in _ANEXO_RE.finditer(s or "")]

def _anexo_en_pos(indices: List[Tuple[int, int]], pos: int) -> Optional[int]:
    last = None
    for i, a in indices:
        if i <= pos: last = a
        else: break
    return last

# ==================== Hints regex (recall) ====================
def _buscar_candidatos(texto: str, pats: List[str], idx_pag: List[Tuple[int, int]], limit: int) -> List[str]:
    hits: List[str] = []
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

DETECTABLE_FIELDS: Dict[str, Dict] = {
    "mant_oferta": {"label": "Mantenimiento de oferta", "pats": [r"mantenim[ií]ento de la oferta", r"validez de la oferta"]},
    "gar_mant":    {"label": "Garantía de mantenimiento", "pats": [r"garant[ií]a.*manten", r"\b5 ?%"]},
    "gar_cumpl":   {"label": "Garantía de cumplimiento", "pats": [r"garant[ií]a.*cumpl", r"\b10 ?%"]},
    "plazo_ent":   {"label": "Plazo de entrega", "pats": [r"plazo de entrega", r"\b\d{1,3}\s*d[ií]as"]},
    "tipo_cambio": {"label": "Tipo de cambio", "pats": [r"Banco\s+Naci[oó]n", r"tipo de cambio", r"\bBNA\b"]},
    "planilla":    {"label": "Planilla de cotización y renglones", "pats": [r"planilla.*cotizaci[oó]n", r"renglones?"]},
    "modalidad":   {"label": "Procedimiento/Modalidad", "pats": [r"licitaci[oó]n\s+(p[úu]blica|privada)", r"contrataci[oó]n\s+directa", r"modalidad"]},
    "plazo_contr": {"label": "Duración del contrato", "pats": [r"duraci[oó]n del contrato", r"plazo contractual"]},
    "presupuesto": {"label": "Monto / Presupuesto", "pats": [r"presupuesto (estimado|oficial|referencial)", r"monto\s+estimado"]},
}

def _build_regex_hints(texto: str, limit_per_field: Optional[int] = None, max_chars: Optional[int] = None) -> str:
    if not texto: return ""
    if limit_per_field is None: limit_per_field = HINTS_PER_FIELD
    if max_chars is None: max_chars = HINTS_MAX_CHARS
    idx_pag = _index_paginas(texto)
    secciones: List[str] = []
    for key, meta in DETECTABLE_FIELDS.items():
        hits = _buscar_candidatos(texto, meta["pats"], idx_pag, limit_per_field)
        if hits:
            secciones.append(f"[{meta['label']}]\n" + "\n".join(hits))
        if sum(len(s) for s in secciones) > max_chars:
            break
    return "\n\n".join(secciones[:])

# ==================== Utilidades varias para salida ====================
def _truncate_words(s: str, max_words: int) -> str:
    try:
        words = re.findall(r"\S+", s or "")
        if len(words) <= max_words:
            return (s or "").strip()
        return " ".join(words[:max_words]).rstrip(",.;:") + "..."
    except Exception:
        return (s or "").strip()

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
        if _CODE_FENCE_RE.match(ln): continue
        if re.match(r"(?i)^\s*informe\s+completo\s*$", ln): continue
        if re.match(r"(?i)^\s*informe\s+original\s*$", ln): continue

        m = _HDR_RE.match(ln)
        if m:
            titulo = _title_case(m.group(2).strip(": ").strip())
            out_lines.append(titulo); out_lines.append(""); continue
        if _TABLE_SEP_RE.match(ln): continue
        if _BULLET_RE.match(ln): ln = _BULLET_RE.sub("• ", ln)

        ln = _LINK_RE.sub(lambda mm: f"{mm.group(1)} ({mm.group(2)})", ln)
        ln = _BOLD_ITALIC_RE.sub(lambda mm: mm.group(2), ln)

        out_lines.append(ln)
        if ln.strip().endswith(":"):
            out_lines.append("")

    texto = "\n".join(out_lines)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto

# ==================== Token caps y llamada OpenAI ====================
def _max_out_for_text(longitud_chars: int) -> int:
    # Modo compacto: cap más bajo para evitar verborrea
    if COMPACT_MODE:
        if ANALISIS_MODO == "fast":
            return 1800 if longitud_chars < 40000 else 2200
        return 2200 if longitud_chars < 40000 else 2600
    # Legacy (andres)
    base = MAX_COMPLETION_TOKENS_SALIDA
    if longitud_chars < 15000 and ANALISIS_MODO == "fast": base = min(base, 2800)
    elif longitud_chars < 40000 and ANALISIS_MODO == "fast": base = min(base, 3500)
    return int(base)

def _pick_model(stage_default: str) -> str:
    if ANALISIS_MODO == "fast" and FAST_FORCE_MODEL:
        return FAST_FORCE_MODEL
    if stage_default == "notas":
        return MODEL_NOTAS
    if stage_default == "sintesis":
        return MODEL_SINTESIS
    return MODEL_ANALISIS

def _llamada_openai(
    messages,
    model=None,
    temperature_str=TEMPERATURE_ANALISIS,
    max_completion_tokens=None,
    retries=2,
    fallback_model="gpt-5-mini",
):
    mdl = model or _pick_model("analisis")
    temp_wanted = None
    if ANALISIS_MODO == "fast":
        temp_wanted = 0.0
    elif temperature_str != "":
        try: temp_wanted = float(temperature_str)
        except Exception: temp_wanted = None

    def _build_kwargs(m, with_temperature=True, max_tok=None):
        kw = dict(
            model=m,
            messages=messages,
            max_completion_tokens=int(max_tok or max_completion_tokens or MAX_COMPLETION_TOKENS_SALIDA),
        )
        if with_temperature and (temp_wanted is not None):
            if _supports_temperature_param(m):
                kw["temperature"] = temp_wanted
        return kw

    models_to_try = [mdl] + ([fallback_model] if fallback_model and fallback_model != mdl else [])
    last_error = None
    for m in models_to_try:
        for attempt in range(retries + 1):
            try:
                kw = _build_kwargs(m, with_temperature=True)
                resp = _chat_create_safe(**kw)
                content = (resp.choices[0].message.content or "").strip()
                if content: return resp
                max_tok_used = int(kw.get("max_completion_tokens", MAX_COMPLETION_TOKENS_SALIDA))
                kw2 = _build_kwargs(m, with_temperature=False, max_tok=min(1024, max_tok_used))
                resp2 = _chat_create_safe(**kw2)
                content2 = (resp2.choices[0].message.content or "").strip()
                if content2: return resp2
                raise RuntimeError("La respuesta del modelo llegó vacía.")
            except Exception as e:
                last_error = e
                if attempt < retries: time.sleep(1.2 * (attempt + 1))
                else: break
    raise RuntimeError(str(last_error) if last_error else "Fallo en _llamada_openai")

# ==================== Chat (sin cambios sustanciales) ====================
MAX_CHAT_CONTEXT_CHARS = int(os.getenv("MAX_CHAT_CONTEXT_CHARS", "38000"))
CHAT_MAX_TOKENS        = int(os.getenv("CHAT_MAX_TOKENS", "1200"))
CHAT_RETRIES           = int(os.getenv("CHAT_RETRIES", "2"))
CHAT_FALLBACK_MODEL    = os.getenv("OPENAI_MODEL_CHAT_FALLBACK", "gpt-5-mini")

def _compactar_contexto_para_chat(contexto: str) -> str:
    s = (contexto or "").strip()
    if len(s) <= MAX_CHAT_CONTEXT_CHARS: return s
    head = s[: MAX_CHAT_CONTEXT_CHARS // 3]
    tail = s[- MAX_CHAT_CONTEXT_CHARS // 3 :]
    medio = s[len(s)//2 - MAX_CHAT_CONTEXT_CHARS//6 : len(s)//2 + MAX_CHAT_CONTEXT_CHARS//6]
    return head + "\n\n[...] (contenido intermedio omitido) [...]\n\n" + medio + "\n\n[...] (contenido intermedio omitido) [...]\n\n" + tail

def _buscar_en_historial_impl(contexto: str, query: str, k: int = 8, window: int = 280) -> dict:
    texto = contexto or ""; q = (query or "").strip()
    if not texto or not q: return {"hits": []}
    low = texto.lower()
    terms = [t for t in re.findall(r"[a-z0-9áéíóúñ/.-]{3,}", q.lower()) if t not in {"que","con","por","del","para","los","las"}] or [q.lower()]
    idx_pag = _index_paginas(texto); seen=set(); hits=[]
    for t in terms:
        for m in re.finditer(re.escape(t), low):
            pos = m.start()
            if any(abs(pos - h) < window//2 for h in seen): continue
            seen.add(pos)
            start = max(0, pos - window); end = min(len(texto), pos + window)
            snippet = texto[start:end].replace("\n"," ").strip()
            p = _pagina_de_indice(idx_pag, pos) if idx_pag else None
            hits.append({"term": t, "page": p, "snippet": ("..." + snippet + "...")})
            if len(hits) >= k: break
        if len(hits) >= k: break
    return {"hits": hits}

def responder_chat_openai(mensaje: str, contexto: str = "", usuario: str = "Usuario") -> str:
    contexto_compacto = _compactar_contexto_para_chat(contexto or "(No hay historial disponible.)")
    tools = [{
        "type": "function",
        "function": {
            "name": "buscar_en_historial",
            "description": "Busca evidencia textual en el historial e informes.",
            "parameters": {"type": "object","properties": {"query":{"type":"string"},"k":{"type":"integer","default":8}},"required":["query"]}
        }
    }]
    system_msg = (
        "Eres el asistente del sistema 'Suizo Argentina – Licitaciones IA'. "
        "Respondes con precisión, sin inventar. Cita (p. N) cuando sea posible."
    )
    user_prompt = f"""
Usuario: {usuario}

=== CONTEXTO (recortado) ===
{contexto_compacto}

=== PREGUNTA ===
{mensaje}

Salida:
- Cita '(p. N)' si usaste evidencia.
- Si no hay evidencia en el material, indícalo y orienta brevemente.
- Sin meta-texto.
"""
    def _chat_call(model_name: str, msgs: list):
        return _chat_create_safe(
            model=model_name, messages=msgs, tools=tools, tool_choice="auto",
            max_tokens=CHAT_MAX_TOKENS, temperature=None,
        )
    model_primary = os.getenv("OPENAI_MODEL_CHAT", _pick_model("analisis"))
    messages = [{"role":"system","content":system_msg},{"role":"user","content":user_prompt}]
    last_error=None
    for model_try in [model_primary, CHAT_FALLBACK_MODEL]:
        if not model_try: continue
        for attempt in range(CHAT_RETRIES + 1):
            try:
                resp = _chat_call(model_try, messages)
                choice = resp.choices[0]
                if getattr(choice.message, "tool_calls", None):
                    for tc in choice.message.tool_calls:
                        if tc.function.name == "buscar_en_historial":
                            try: args = json.loads(tc.function.arguments or "{}")
                            except Exception: args = {"query": (mensaje or "")}
                            result = _buscar_en_historial_impl(contexto_compacto, args.get("query",""), int(args.get("k",8)))
                            messages.append({"role":"tool","tool_call_id":tc.id,"name":"buscar_en_historial","content":json.dumps(result, ensure_ascii=False)})
                    resp2 = _chat_call(model_try, messages)
                    out = (resp2.choices[0].message.content or "").strip()
                    if out: return out
                    raise RuntimeError("Respuesta vacía tras tool-calling.")
                else:
                    out = (choice.message.content or "").strip()
                    if out: return out
                    raise RuntimeError("Respuesta vacía.")
            except Exception as e:
                last_error=e; time.sleep(1.2*(attempt+1))
    return "No pude generar respuesta en este momento." + (f" Detalle: {last_error}" if last_error else "")

# ==================== Post-procesos específicos ====================
def _strip_ficha(informe: str) -> str:
    # Elimina bloque de “Ficha estandarizada…” si apareciera
    s = informe or ""
    m = re.search(r"(?is)^\s*Ficha\s+estandarizada\s+del\s+procedimiento.*?(?=^\s*\d+\))", s)
    if m:
        s = s[:m.start()] + s[m.end():]
    return s

def _normalizar_encabezados_salida(informe: str) -> str:
    s = informe or ""
    s = re.sub(
        r"(?im)^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento.*$",
        "", s
    )
    return s.strip()

# ==================== Analizador principal ====================
def analizar_con_openai(texto: str) -> str:
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    texto = _limpieza_basica_preanalisis(texto)
    texto_len = len(texto)
    n_anexos = _contar_anexos(texto)
    varios_anexos = n_anexos >= 2

    # Elige prompt por modo
    if COMPACT_MODE:
        prompt_local = prompt_compacto(varios_anexos)
    else:
        prompt_local = prompt_andres(varios_anexos)

    hints_block = ""
    if ENABLE_REGEX_HINTS and not COMPACT_MODE:
        hints = _build_regex_hints(texto)
        if hints:
            hints_block = f"\n\n=== HALLAZGOS AUTOMÁTICOS (snippets literales para verificación) ===\n{hints}\n"

    force_two_stage = (varios_anexos and texto_len >= MULTI_FORCE_TWO_STAGE_MIN_CHARS)

    # Single pass (preferido)
    if (not varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS) or \
       (varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS_MULTI and not force_two_stage):
        t0 = _t()
        max_out = _max_out_for_text(texto_len)
        messages = [
            {"role":"system","content":"Equipo experto en compras públicas argentinas. Redactor técnico-jurídico. Cero invenciones."},
            {"role":"user","content": f"{prompt_local}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\nDevuelve SOLO el informe final (sin preámbulos)."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = (resp.choices[0].message.content or "").strip()
            bruto = _limpiar_meta(bruto)
            if COMPACT_MODE:
                bruto = _strip_ficha(bruto)
            bruto = _normalizar_encabezados_salida(bruto)

            # Segundo pase: rellena solo NO ESPECIFICADO, sin alterar estructura
            if ENABLE_SECOND_PASS_COMPLETION:
                if re.search(r"(?i)\bNO ESPECIFICADO\b", bruto or ""):
                    corr_prompt = f"""Corrige ÚNICAMENTE los campos marcados como "NO ESPECIFICADO" usando citas literales del texto fuente.
No agregues secciones ni cambies el orden ni los encabezados. Mantén el mismo formato compacto 1–18 + 'Conclusiones y recomendaciones'.
Respeta las reglas de citas del informe original.

=== INFORME ===
{bruto}

=== TEXTO FUENTE (para buscar literal) ===
{texto}
"""
                    resp2 = _llamada_openai(
                        [{"role":"system","content":"Redactor técnico-jurídico. Cero invenciones."},
                         {"role":"user","content": corr_prompt}],
                        model=_pick_model("sintesis"),
                        max_completion_tokens=_max_out_for_text(texto_len)
                    )
                    bruto2 = (resp2.choices[0].message.content or "").strip()
                    if bruto2:
                        bruto = _limpiar_meta(_strip_ficha(bruto2))

            out = preparar_texto_para_pdf(bruto)
            _log_tiempo("analizar_single_pass_compacto" if COMPACT_MODE else "analizar_single_pass", t0)
            return out
        except Exception as e:
            return f"Error al generar el análisis: {e}"

    # Dos etapas (chunking + concurrencia) — poco frecuente en compacto
    chunk_size = max(CHUNK_SIZE_BASE, (texto_len + TARGET_PARTS - 1) // TARGET_PARTS)
    partes = [texto[i:i + chunk_size] for i in range(0, texto_len, chunk_size)]

    # A) Notas intermedias
    def _worker_notas(idx: int, parte: str):
        msg = [
            {"role":"system","content":"Analista jurídico. Extrae bullets concisos con citas; cero invenciones."},
            {"role":"user","content": f"{CRAFT_PROMPT_NOTAS}\n\n=== FRAGMENTO {idx+1}/{len(partes)} ===\n{parte}"}
        ]
        r = _llamada_openai(msg, max_completion_tokens=min(NOTAS_MAX_TOKENS, 900), model=_pick_model("notas"))
        return idx, (r.choices[0].message.content or "").strip()

    notas_list: List[Optional[str]] = [None]*len(partes)
    t0_notas = _t()
    with ThreadPoolExecutor(max_workers=max(1, ANALISIS_CONCURRENCY)) as ex:
        future_to_idx = {ex.submit(_worker_notas, i, p): i for i,p in enumerate(partes)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try: _, content = fut.result(); notas_list[i] = content
            except Exception as e: notas_list[i] = f"[ERROR notas {i+1}]: {e}"
    _log_tiempo(f"notas_intermedias_{len(partes)}p", t0_notas)
    notas_integradas = "\n".join([r or "" for r in notas_list])

    # B) Síntesis final
    t0_sint = _t()
    max_out = _max_out_for_text(texto_len)
    messages_final = [
        {"role":"system","content":"Equipo experto en compras públicas argentinas. Redactor técnico-jurídico. Cero invenciones."},
        {"role":"user","content": f"""{prompt_local}

=== NOTAS INTERMEDIAS (integradas) ===
{notas_integradas}

Integra TODO en un solo informe COMPACTO (1–18 + Conclusiones). No agregues otras secciones.
Cita una sola vez por dato. Devuelve SOLO el informe final en texto.
"""}
    ]
    try:
        resp_final = _llamada_openai(messages_final, max_completion_tokens=max_out, model=_pick_model("sintesis"))
        bruto = (resp_final.choices[0].message.content or "").strip()
        bruto = _limpiar_meta(bruto)
        if COMPACT_MODE:
            bruto = _strip_ficha(bruto)
        bruto = _normalizar_encabezados_salida(bruto)
        out = preparar_texto_para_pdf(bruto)
        _log_tiempo("sintesis_final_compacto" if COMPACT_MODE else "sintesis_final", t0_sint)
        return out
    except Exception as e:
        return f"Error en la síntesis final: {e}\n\nNotas intermedias:\n{_limpiar_meta(notas_integradas)}"

# ==================== PDF ====================
def _render_pdf_bytes(resumen: str, fecha_display: Optional[str] = None) -> bytes:
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
            y -= alto_linea; continue
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
    output_dir = os.path.join("generated_pdfs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, nombre_archivo)
    data = _render_pdf_bytes(resumen, fecha_display=fecha_display)
    with NamedTemporaryFile(dir=output_dir, delete=False) as tmp:
        tmp.write(data); tmp_path = tmp.name
    try:
        os.replace(tmp_path, output_path)
    except Exception:
        with open(output_path, "wb") as f:
            f.write(data)
        try: os.remove(tmp_path)
        except Exception: pass
    return output_path

def dividir_texto(texto, canvas_obj, max_width):
    palabras = (texto or "").split(" ")
    lineas, linea_actual = [], ""
    for palabra in palabras:
        prueba = (linea_actual + " " + palabra) if linea_actual else palabra
        if canvas_obj.stringWidth(prueba, canvas_obj._fontname, canvas_obj._fontsize) <= max_width:
            linea_actual = prueba
        else:
            if linea_actual: lineas.append(linea_actual)
            linea_actual = palabra
    if linea_actual: lineas.append(linea_actual)
    return lineas
