# utils.py
import io
import os
import re
import base64
import mimetypes
import time
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
from zoneinfo import ZoneInfo  # <<< fallback horario AR

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
MODEL_ANALISIS  = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-4o-mini")
VISION_MODEL    = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")
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

# Granularidad / anti-copia ligera
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

# --- Limitar enumeraciones y desactivar expansiones automáticas (nuevo) ---
EXPAND_SECTIONS_213_216 = int(os.getenv("EXPAND_SECTIONS_213_216", "0"))   # 0 = NO expandir ni sustituir 2.13/2.16
MAX_RENGLONES_OUT       = int(os.getenv("MAX_RENGLONES_OUT", "12"))
MAX_ARTICULOS_OUT       = int(os.getenv("MAX_ARTICULOS_OUT", "12"))

# Forzar reemplazo determinístico de 2.13 y 2.16 -> default 0
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
            max_completion_tokens=2400
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[OCR-ERROR] {e}"

# ---- OCR selectivo en paralelo (MUESTREO UNIFORME EN TODO EL DOC) ----
from concurrent.futures import ThreadPoolExecutor, as_completed

def _ocr_pagina_png_bytes(png_bytes: bytes, idx: int) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    txt = _ocr_openai_imagen_b64(b64)
    return f"[PÁGINA {idx+1}]\n{txt}" if txt else f"[PÁGINA {idx+1}] (sin texto OCR)"

def _ocr_selectivo_por_pagina(doc: fitz.Document, max_pages: int) -> str:
    """
    NUEVO: muestreo distribuido de páginas a lo largo de todo el documento.
    Así, aunque la planilla esté al final, entra en el muestreo (p.ej. 8 páginas: inicio/medio/fin).
    """
    n = len(doc)
    if n == 0:
        return ""
    to_process = min(n, max_pages)

    # Índices muestreados uniformemente en [0, n-1]
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
            if txt: partes.append(txt)
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
# ==================== Prompts y limpieza ====================
SINONIMOS_CANONICOS = r"""
[Guía de mapeo semántico – Argentina (nacional, provincial, municipal)]
- "Número de proceso" ≈ "Expediente", "N° de procedimiento", "N° de trámite", "EX-...", "IF-...".
- "Nombre de proceso" ≈ "Denominación del procedimiento", "Título del llamado".
- "Objeto de la contratación" ≈ "Objeto", "Adquisición/Contratación de", "Finalidad".
- "Procedimiento de selección" ≈ "Tipo de procedimiento", "Modalidad", "Clase del llamado" (Licitación Pública/Privada, Contratación Directa, Compra Menor, Subasta, etc.).
- "Tipo de cotización" ≈ "Forma de cotización", "Modo de cotizar", "Planilla de precios", "Ítem por ítem", "Global/Total", "Por renglón/lote".
- "Tipo de adjudicación" ≈ "Criterio de adjudicación", "Adjudicación por renglón/lote/total".
- "Cantidad de ofertas permitidas" ≈ "Número de propuestas por oferente", "Ofertas alternativas/adicionales".
- "Estado" ≈ "Situación del trámite" (vigente, abierto, cerrado, desierto, fracasado, adjudicado), si el documento lo consigna.
- "Plazo de mantenimiento de la oferta" ≈ "Validez de la oferta".
- "Número de renglón" ≈ "Renglón", "Ítem (número)".
- "Objeto del gasto" ≈ "Partida presupuestaria", "Clasificador/Objeto del gasto", "Estructura programática".
- "Código del ítem" ≈ "Código interno", "Código catálogo", "SKU".
- "Descripción" ≈ "Descripción del ítem", "Especificaciones técnicas".
- "Cantidad" ≈ "Cantidad solicitada/Requerida".
- "Inicio y final de consultas" ≈ "Plazo de consultas/aclaraciones", "Recepción de consultas", "Preguntas y respuestas".
- "Fecha y hora del acto de apertura" ≈ "Apertura", "Acto de apertura de ofertas".
- "Monto" ≈ "Presupuesto oficial/referencial", "Monto estimado", "Crédito disponible".
- "Moneda" ≈ "Moneda de cotización" (ARS, USD, etc.), "Tipo de cambio".
- "Duración del contrato" ≈ "Plazo contractual", "Vigencia", "Por el término de".
- "Presentación de ofertas" ≈ "Acto de presentación", "Límite de recepción".
- "Garantía de mantenimiento" ≈ "Garantía de oferta".
- "Garantía de cumplimiento" ≈ "Garantía contractual".
- "Planilla de cotización" ≈ "Formulario de oferta", "Cuadro comparativo", "Planilla de precios".
- "Tipo de cambio BNA" ≈ "Banco Nación vendedor del día anterior".

Usá esta guía: si un campo aparece con sinónimos/variantes, NO lo marques como "NO ESPECIFICADO".
No menciones nombres de portales/sistemas salvo que estén explícitamente en los documentos analizados.
"""

# ======= PROMPT MAESTRO ESTILO ANDRÉS (ajustado anti-eco) =======
_BASE_PROMPT_ANDRES = r"""
# (Instrucciones internas: NO imprimir este encabezado ni estas reglas en la salida)

Objetivo
- Generar un **informe de análisis de licitación en Argentina** (ámbitos nacional, provincial o municipal), exhaustivo y **sin invenciones**.
- La salida debe comenzar con el encabezado literal: **0) Ficha estandarizada del procedimiento (campos estandarizados)**,
  seguido EXCLUSIVAMENTE por las 19 líneas con rótulo exacto y valor en el formato **"Rótulo: valor (cita)"**.
  **No copies ninguna frase de estas instrucciones dentro de la Ficha.**
- Para **"Número de renglón"**: NO escribas texto libre. Escribe exactamente **"Total de renglones: N; ver Sección 9 para el detalle completo"** (con N real).
- Para **"Código del ítem"**, **"Descripción"** y **"Cantidad"**: si existen a nivel renglón/planilla, escribe **"Ver Sección 9 (detalle por renglón)"**.
- Si algo NO figura en los archivos, escribir **"NO ESPECIFICADO"** y **no inventar ni inferir**.
- Cada línea con dato crítico debe terminar con **cita de fuente** según “Reglas de Citas”.
- **Además de la Ficha**, incluir las secciones 1–12 (debajo) para no perder nada del informe ampliado.

{REGLAS_CITAS}

Estilo
- Encabezados y listas claras; sin meta-texto (“parte X de Y”, “revise el resto”, etc.).
- Deduplicar, fusionar y no repetir información.
- Mantener terminología del pliego. Usar 2 decimales si el pliego lo exige para precios.
- No mencionar nombres de portales/sistemas salvo que figuren explícitamente en los documentos.

Estructura de salida EXACTA (usar estos títulos tal cual)
0) Ficha estandarizada del procedimiento (campos estandarizados)
1) Resumen ejecutivo (≤200 palabras)
2) Datos clave del llamado
3) Alcance contractual y vigencias
4) Entregas y logística
5) Presentación y contenido de la oferta
6) Evaluación, empate y mejora de oferta
7) Garantías
8) Muestras, envases, etiquetado y caducidad (si aplica)
9) Renglones y planilla de cotización
10) Checklist operativo
11) Fechas y plazos críticos
12) Observaciones finales

Cobertura obligatoria por sección (según aplique)
- 2) Datos clave: Organismo, Expediente/N° proceso, Tipo/Modalidad/Etapa, Objeto, Rubro, Lugar/área; contactos/portales (mails/URLs) si figuran.
- 3) Alcance/vigencias: mantenimiento de oferta y prórroga; perfeccionamiento; ampliaciones/topes; duración/termino del contrato.
- 4) Entregas: lugar/horarios; forma (única/parcelada); plazos; flete/descarga.
- 5) Presentación: sobre/caja, duplicado, firma, rotulado; documentación fiscal/registral; costo/valor del pliego si existe.
- 6) Evaluación: cuadro comparativo; tipo de cambio; criterios cuali/cuantitativos; empates; mejora de precio.
- 7) Garantías: umbrales por UC si aplica; % mantenimiento y % cumplimiento con plazos/condiciones; contragarantías.
- 8) Muestras/envases/etiquetado/caducidad: ANMAT/BPM; cadena de frío; rotulados; vigencia mínima.
- 9) Renglones/planilla: incluir TODOS los renglones (si existe planilla). Por renglón: Cantidad, Código (si hay), Descripción y especificaciones técnicas relevantes en 1 línea.
- 10) Checklist: acciones para el oferente.
- 11) Fechas críticas: presentación, apertura, mantenimiento, entregas, consultas, etc.
- 12) Observaciones finales: alertas y condicionantes.

Guía de sinónimos:
{SINONIMOS}
"""

def _prompt_andres(varios_anexos: bool) -> str:
    if varios_anexos:
        reglas = (
            "Reglas de Citas:\n"
            "- Documento MULTI-ANEXO: al final de cada línea con dato, usar (Anexo X, p. N).\n"
            "- Deducir N tomando la etiqueta [PÁGINA N] más cercana dentro del texto del ANEXO correspondiente.\n"
            "- Si no hay paginación: (Fuente: documento provisto)."
        )
    else:
        reglas = (
            "Reglas de Citas:\n"
            "- Documento ÚNICO: al final de cada línea con dato, usar (p. N) a partir de la etiqueta [PÁGINA N] más cercana.\n"
            "- Si no hay paginación: (Fuente: documento provisto)."
        )
    return _BASE_PROMPT_ANDRES.format(
        REGLAS_CITAS=reglas,
        SINONIMOS=SINONIMOS_CANONICOS
    )

# ---------------------- Índices y citas ----------------------
_ANEXO_RE = re.compile(r"(?im)^===\s*ANEXO\s+(\d+)")
_PAG_TAG_RE = re.compile(r"\[PÁGINA\s+(\d+)\]")

def _index_paginas(s: str) -> List[Tuple[int,int]]:
    return [(m.start(), int(m.group(1))) for m in _PAG_TAG_RE.finditer(s or "")]

def _index_anexos(s: str) -> List[Tuple[int,int]]:
    return [(m.start(), int(m.group(1))) for m in _ANEXO_RE.finditer(s or "")]

def _pagina_de_indice(indices: List[Tuple[int,int]], pos: int) -> int:
    last = 1
    for i, p in indices:
        if i <= pos:
            last = p
        else:
            break
    return last

def _anexo_en_pos(indices: List[Tuple[int,int]], pos: int) -> Optional[int]:
    last = None
    for i, a in indices:
        if i <= pos:
            last = a
        else:
            break
    return last

def _cita_por_pos(texto: str, pos: Optional[int], varios_anexos: bool) -> str:
    if pos is None or pos < 0:
        return "(Fuente: documento provisto)"
    ip = _index_paginas(texto)
    ia = _index_anexos(texto)
    p = _pagina_de_indice(ip, pos) if ip else None
    ax = _anexo_en_pos(ia, pos) if ia else None
    if varios_anexos and ax:
        return f"(Anexo {ax}, p. {p or '?'} )"
    return f"(p. {p})" if p else "(Fuente: documento provisto)"

# ---------------------- Extractores simples por regex ----------------------
def _pick_group(m: re.Match) -> str:
    # devuelve el grupo no vacío más largo; si no hay, el group(0)
    if not m:
        return ""
    groups = [g for g in m.groups(default="") if g]
    if not groups:
        return m.group(0)
    return max(groups, key=len)

def _search_first(texto: str, patrones: List[re.Pattern]) -> Tuple[str, Optional[int]]:
    for cre in patrones:
        m = cre.search(texto)
        if m:
            val = _pick_group(m)
            val = re.sub(r"\s+", " ", (val or "")).strip()
            return val, m.start()
    return "", None

# Patrones por campo
_CRE_EXPEDIENTE = [
    re.compile(r"\b(EX-\d{4}-[A-Z0-9-]+)"),
    re.compile(r"(?im)^\s*Expediente\s*(?:N[°º.]?\s*)?[:\-]\s*([A-Z0-9./\-\s]+)$"),
]
_CRE_NOMBRE_PROC = [
    re.compile(r"(?im)^\s*(?:Denominaci[oó]n del procedimiento|Nombre del proceso|T[íi]tulo del llamado)\s*[:\-]\s*(.+)$"),
]
_CRE_OBJETO = [
    re.compile(r"(?ims)^\s*Objeto(?:\s+de la contrataci[oó]n)?\s*[:\-]\s*(.+?)(?=\n\s*\w|\Z)"),
]
_CRE_MODALIDAD = [
    re.compile(r"(?i)\bLicitaci[oó]n\s+(P[úu]blica|Privada)\b"),
    re.compile(r"(?i)\bContrataci[oó]n\s+Directa\b"),
    re.compile(r"(?i)\bCompra\s+Menor\b"),
    re.compile(r"(?i)\bSubasta\b"),
    re.compile(r"(?i)\bModalidad\s*[:\-]\s*([^\n]+)"),
]
_CRE_TIPO_COTIZ = [
    re.compile(r"(?i)(?:Forma|Tipo)\s+de\s+cotizaci[oó]n\s*[:\-]\s*([^\n]+)"),
    re.compile(r"(?i)\bcotizaci[oó]n\s+por\s+(rengl[oó]n|[ií]tem|lote|global|total)"),
]
_CRE_TIPO_ADJ = [
    re.compile(r"(?i)adjudicaci[oó]n\s+por\s+(rengl[oó]n|lote|total)"),
]
_CRE_OFERTAS = [
    re.compile(r"(?i)\bofertas?\s+alternativas\b"),
    re.compile(r"(?i)\buna\s+sola\s+oferta\b"),
]
_CRE_ESTADO = [
    re.compile(r"(?i)\b(vigente|abierto|cerrado|adjudicado|desierto|fracasado)\b"),
]
_CRE_MANT_OFERTA = [
    re.compile(r"(?i)(?:mantenim[ií]ento|validez)\s+de\s+la\s+oferta\s*[:\-]?\s*([\d]{1,3}\s*d[ií]as(?:\s*h[aá]biles)?)"),
]
_CRE_CONSULTAS = [
    re.compile(r"(?i)consultas?.{0,40}?(\d{2}/\d{2}/\d{4}).{0,40}?(?:al|hasta)\s*(\d{2}/\d{2}/\d{4})"),
]
_CRE_APERTURA = [
    re.compile(r"(?i)(?:acto\s+de\s+)?apertura.{0,60}?(\d{2}/\d{2}/\d{4}).{0,20}?(\d{1,2}:\d{2})"),
]
_CRE_PRESUP = [
    re.compile(r"(?i)(?:presupuesto\s+(?:oficial|referencial|estimado)|monto\s+estimado|cr[eé]dito\s+disponible)\s*[:\-]?\s*(?:\$|ARS|USD)?\s*([\d\.\,]+)"),
]
_CRE_MONEDA = [
    re.compile(r"(?i)\b(ARS|USD|PESOS?|D[ÓO]LARES?)\b"),
]
_CRE_DURACION = [
    re.compile(r"(?i)(?:duraci[oó]n|vigencia|por\s+el\s+t[eé]rmino\s+de)\s*[:\-]?\s*(\d{1,4}\s*(?:d[ií]as|meses|a[nñ]os))"),
]
_CRE_OBJ_GASTO = [
    re.compile(r"(?im)^\s*(?:Objeto\s+del\s+gasto|Partida\s+presupuestaria|Clasificador.*)\s*[:\-]\s*(.+)$"),
]

def _extract_expediente(texto):     return _search_first(texto, _CRE_EXPEDIENTE)
def _extract_nombre(texto):         return _search_first(texto, _CRE_NOMBRE_PROC)
def _extract_objeto(texto):         return _search_first(texto, _CRE_OBJETO)
def _extract_modalidad(texto):      return _search_first(texto, _CRE_MODALIDAD)
def _extract_tipo_cotiz(texto):     return _search_first(texto, _CRE_TIPO_COTIZ)
def _extract_tipo_adj(texto):       return _search_first(texto, _CRE_TIPO_ADJ)
def _extract_ofertas(texto):        return _search_first(texto, _CRE_OFERTAS)
def _extract_estado(texto):         return _search_first(texto, _CRE_ESTADO)
def _extract_mant_oferta(texto):    return _search_first(texto, _CRE_MANT_OFERTA)
def _extract_consultas(texto):      return _search_first(texto, _CRE_CONSULTAS)
def _extract_apertura(texto):       return _search_first(texto, _CRE_APERTURA)
def _extract_presupuesto(texto):    return _search_first(texto, _CRE_PRESUP)
def _extract_moneda(texto):         return _search_first(texto, _CRE_MONEDA)
def _extract_duracion(texto):       return _search_first(texto, _CRE_DURACION)
def _extract_objeto_gasto(texto):   return _search_first(texto, _CRE_OBJ_GASTO)

# ---------------------- Ficha estandarizada determinística ----------------------
def _valor_o_noesp(valor: str) -> str:
    v = (valor or "").strip()
    return v if v else "NO ESPECIFICADO"

def _build_ficha_deterministica(texto: str, varios_anexos: bool) -> str:
    # Campos base
    v_exp, p_exp = _extract_expediente(texto)
    v_nom, p_nom = _extract_nombre(texto)
    v_obj, p_obj = _extract_objeto(texto)
    v_mod, p_mod = _extract_modalidad(texto)
    v_tc,  p_tc  = _extract_tipo_cotiz(texto)
    v_ta,  p_ta  = _extract_tipo_adj(texto)
    v_of,  p_of  = _extract_ofertas(texto)
    v_est, p_est = _extract_estado(texto)
    v_mant,p_mant= _extract_mant_oferta(texto)
    v_cons,p_cons= _extract_consultas(texto)
    v_ap,  p_ap  = _extract_apertura(texto)
    v_mont,p_mont= _extract_presupuesto(texto)
    v_mon, p_mon = _extract_moneda(texto)
    v_dur, p_dur = _extract_duracion(texto)
    v_objg,p_objg= _extract_objeto_gasto(texto)

    # Conteo renglones (usa extractor avanzado que se define más abajo en el archivo)
    try:
        rows = _extraer_renglones_y_especificaciones(texto)
    except Exception:
        rows = []
    total_renglones = len(rows)
    # página/anexo del primer renglón (para citar "ver Sección 9")
    p_first = rows[0][4] if rows else None
    ax_first = rows[0][5] if rows else None
    if p_first is not None:
        # si tenemos página/anexo embebidos en rows, construir cita explícita
        cita_renglones = f"(Anexo {ax_first}, p. {p_first})" if (varios_anexos and ax_first) else (f"(p. {p_first})" if p_first else "(Fuente: documento provisto)")
    else:
        cita_renglones = "(Fuente: documento provisto)"

    # Armar ficha
    out = []
    out.append("0) Ficha estandarizada del procedimiento (campos estandarizados):")
    out.append(f"- N° de proceso: {_valor_o_noesp(v_exp)} {_cita_por_pos(texto, p_exp, varios_anexos)}")
    out.append(f"- Nombre de proceso: {_valor_o_noesp(v_nom)} {_cita_por_pos(texto, p_nom, varios_anexos)}")
    out.append(f"- Objeto de la contratación: {_valor_o_noesp(v_obj)} {_cita_por_pos(texto, p_obj, varios_anexos)}")
    out.append(f"- Procedimiento de selección: {_valor_o_noesp(v_mod)} {_cita_por_pos(texto, p_mod, varios_anexos)}")
    out.append(f"- Tipo de cotización: {_valor_o_noesp(v_tc)} {_cita_por_pos(texto, p_tc, varios_anexos)}")
    out.append(f"- Tipo de adjudicación: {_valor_o_noesp(v_ta)} {_cita_por_pos(texto, p_ta, varios_anexos)}")
    out.append(f"- Cantidad de ofertas permitidas: {_valor_o_noesp(v_of)} {_cita_por_pos(texto, p_of, varios_anexos)}")
    out.append(f"- Estado: {_valor_o_noesp(v_est)} {_cita_por_pos(texto, p_est, varios_anexos)}")
    out.append(f"- Plazo de mantenimiento de la oferta: {_valor_o_noesp(v_mant)} {_cita_por_pos(texto, p_mant, varios_anexos)}")

    if total_renglones > 0:
        out.append(f"- Número de renglón: Total de renglones: {total_renglones}; ver Sección 9 para el detalle completo {cita_renglones}")
    else:
        out.append(f"- Número de renglón: NO ESPECIFICADO (Fuente: documento provisto)")

    out.append(f"- Objeto del gasto: {_valor_o_noesp(v_objg)} {_cita_por_pos(texto, p_objg, varios_anexos)}")

    # Estos 3 se informan a nivel renglón → remitir a Sección 9
    rem9 = f"Ver Sección 9 (detalle por renglón) {cita_renglones}"
    out.append(f"- Código del ítem: {rem9}")
    out.append(f"- Descripción: {rem9}")
    out.append(f"- Cantidad: {rem9}")

    out.append(f"- Inicio y final de consultas: {_valor_o_noesp(' al '.join([x for x in [v_cons] if x]))} {_cita_por_pos(texto, p_cons, varios_anexos)}")
    out.append(f"- Fecha y hora del acto de apertura: {_valor_o_noesp(' '.join([x for x in [v_ap] if x]))} {_cita_por_pos(texto, p_ap, varios_anexos)}")
    out.append(f"- Monto: {_valor_o_noesp(v_mont)} {_cita_por_pos(texto, p_mont, varios_anexos)}")
    out.append(f"- Moneda: {_valor_o_noesp(v_mon)} {_cita_por_pos(texto, p_mon, varios_anexos)}")
    out.append(f"- Duración del contrato: {_valor_o_noesp(v_dur)} {_cita_por_pos(texto, p_dur, varios_anexos)}")

    return "\n".join(out).strip()

def _reemplazar_ficha_en_informe(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Reemplaza lo que el modelo haya puesto en "0) Ficha..." por una FICHA determinística,
    corrigiendo así:
     - que no copie frases de las instrucciones;
     - que 'Número de renglón' muestre 'Total de renglones: N; ...';
     - que Código/Descripción/Cantidad remitan a Sección 9 si aplica.
    """
    ficha = _build_ficha_deterministica(texto_fuente, varios_anexos)
    # buscar el inicio de 0) Ficha...
    m0 = re.search(r"(?im)^\s*0\)\s*Ficha\s+estandarizada[^\n]*\n?", informe)
    if not m0:
        # si no está, la anteponemos
        return ficha + "\n\n" + informe
    start = m0.start()
    # el fin de la ficha es el comienzo de "1) " (o el fin del texto)
    m1 = re.search(r"(?im)^\s*1\)\s", informe[m0.end():])
    end = m0.end() + (m1.start() if m1 else 0)
    if end <= m0.end():
        # no encontró 1) — insertamos al inicio del documento
        return informe[:start] + ficha + "\n\n" + informe[m0.end():]
    return informe[:start] + ficha + "\n" + informe[end:]
# ==================== Renglones y Artículos (extractores robustos) ====================
# Artículos: detecta encabezados y bloques hasta el siguiente artículo
_ART_HEAD_RE = re.compile(r"(?im)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.*)$")
_ART_BLOCK_RE = re.compile(
    r"(?ims)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.+?)(?=^\s*art(?:[íi]culo|\.?)\s*\d+[a-zº°]?|\Z)"
)

def _extraer_articulos_con_snippets(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (rotulo_articulo, snippet_200c, pagina_aprox, anexo_num).
    Usa etiquetas [PÁGINA N] y '=== ANEXO' para ubicar citas.
    """
    idx = _index_paginas(texto)
    idx_ax = _index_anexos(texto)
    res: List[Tuple[str, str, int, Optional[int]]] = []

    # Bloques completos (preferido)
    for m in _ART_BLOCK_RE.finditer(texto or ""):
        start = m.start()
        p = _pagina_de_indice(idx, start)
        ax = _anexo_en_pos(idx_ax, start)
        rotulo = (m.group(1) or "").strip()
        contenido = (m.group(2) or "").strip()
        snippet = contenido[:200].replace("\n", " ").strip()
        res.append((rotulo, snippet, p, ax))

    # Si no hubo bloques, al menos tomar encabezados sueltos
    if not res:
        for m in _ART_HEAD_RE.finditer(texto or ""):
            start = m.start()
            p = _pagina_de_indice(idx, start)
            ax = _anexo_en_pos(idx_ax, start)
            rotulo = (m.group(1) or "").strip()
            snippet = ((m.group(2) or "").strip()[:200]).replace("\n", " ")
            res.append((rotulo, snippet, p, ax))
    return res


# Renglones: exige que la fila comience con "Renglón" / "Reng." y agrupa texto hasta el siguiente
_ROW_START_RE = re.compile(r"(?im)^(?:reng(?:l[oó]n)?\.?\s*)(\d{1,4})\b")
_CODE_RE      = re.compile(r"\b[A-Z]{1,3}\d{5,8}\b")  # p.ej. D0330113, GB079001, E5001253
_QTY_RE       = re.compile(r"\b\d{1,6}\b")

def _extraer_renglones_y_especificaciones(texto: str) -> List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]]:
    """
    Devuelve lista de tuplas:
      (num_renglon, cantidad, codigo, descripcion_full, pagina_aprox, anexo_num)

    - Reconoce filas numeradas que EMPIEZAN con 'Renglón'/'Reng.' (robusto a acentos y punto).
    - Acumula líneas hasta el próximo comienzo de fila.
    - Extrae cantidad (primer entero tras el número), código catálogo si aparece (CODE_RE),
      y deja en 'descripcion_full' la descripción con especificaciones.
    """
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)

    lines = (texto or "").splitlines()
    pos_abs = 0
    starts: List[Tuple[int, int]] = []  # (line_index, absolute_pos)

    for i, ln in enumerate(lines):
        if _ROW_START_RE.match(ln):
            starts.append((i, pos_abs))
        pos_abs += len(ln) + 1

    if not starts:
        return []

    # Sentinela (fin del texto)
    starts.append((len(lines), len(texto)))

    out: List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]] = []
    for k in range(len(starts) - 1):
        i_line, abs_pos = starts[k]
        j_line, _ = starts[k + 1]

        block_lines = lines[i_line:j_line]
        block_text = " ".join([re.sub(r"\s+", " ", x).strip() for x in block_lines if x.strip()])

        # número de renglón
        mnum = _ROW_START_RE.match(lines[i_line])
        num_r = int(mnum.group(1)) if (mnum and mnum.group(1).isdigit()) else None

        # cantidad (primer entero tras el número)
        qty = None
        if mnum:
            tail = lines[i_line][mnum.end():]
            mqty = _QTY_RE.search(tail)
            if mqty:
                try:
                    qty = int(mqty.group(0))
                except Exception:
                    qty = None

        # código (en todo el bloque)
        mcode = _CODE_RE.search(block_text)
        code = mcode.group(0) if mcode else None

        # descripción y especificaciones (limpia código/cantidad/número)
        desc = block_text
        if code:
            desc = re.sub(re.escape(code), "", desc)
        if qty is not None:
            desc = re.sub(rf"\b{qty}\b", "", desc)
        if num_r is not None:
            desc = re.sub(rf"^\s*{num_r}\b", "", desc)
        desc = re.sub(r"\s+", " ", (desc or "")).strip()

        # citas
        p  = _pagina_de_indice(idx_pag, abs_pos) if idx_pag else None
        ax = _anexo_en_pos(idx_ax, abs_pos) if idx_ax else None

        if num_r is not None:
            out.append((num_r, qty, code, desc, p, ax))

    out.sort(key=lambda t: t[0])
    return out


# ==================== Contactos (2.3) y Normativa (2.15) ====================
CONTACT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CONTACT_URL_RE   = re.compile(r"(https?://[^\s)]+|www\.[^\s)]+)")

def _extraer_contactos_con_paginas(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (tipo, valor, p, anexo) con tipo in {"email","url"}.
    Dedupe preservando orden.
    """
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)

    res: List[Tuple[str, str, int, Optional[int]]] = []
    for m in CONTACT_EMAIL_RE.finditer(texto or ""):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos) if idx_pag else None
        ax = _anexo_en_pos(idx_ax, pos) if idx_ax else None
        res.append(("email", m.group(0), p, ax))
    for m in CONTACT_URL_RE.finditer(texto or ""):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos) if idx_pag else None
        ax = _anexo_en_pos(idx_ax, pos) if idx_ax else None
        v = (m.group(0) or "").rstrip(").,;")
        res.append(("url", v, p, ax))

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
    items = _extraer_contactos_con_paginas(texto)
    if not items:
        return ""
    out = ["2.3 Contactos y portales:"]
    for (t, v, p, ax) in items:
        etiqueta = "Email" if t == "email" else "URL"
        cita = f"(Anexo {ax}, p. {p})" if (varios_anexos and ax) else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {etiqueta}: {v} {cita}")
    return "\n".join(out)


# Normativa (Ley/Decreto/Resolución/Disposición)
NORM_TIPOS = [
    (r"Ley", r"\bLey(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    (r"Decreto", r"\bDecreto(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    (r"Resolución", r"\bResoluci[oó]n(?:\s*(?:Ministerial|Conjunta))?\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
    (r"Disposición", r"\bDisposici[oó]n\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
]
NORM_PATTS = [(tipo, re.compile(patt, re.I)) for (tipo, patt) in NORM_TIPOS]

def _extraer_normativa(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (tipo, numero, p, anexo). Dedupe preservando orden.
    """
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)

    res: List[Tuple[str, str, int, Optional[int]]] = []
    for (tipo, cre) in NORM_PATTS:
        for m in cre.finditer(texto or ""):
            pos = m.start()
            p = _pagina_de_indice(idx_pag, pos) if idx_pag else None
            ax = _anexo_en_pos(idx_ax, pos) if idx_ax else None
            numero = (m.group(1) or "").strip()
            res.append((tipo, numero, p, ax))

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
    normas = _extraer_normativa(texto)
    if not normas:
        return ""
    out = ["2.15 Normativa aplicable:"]
    for (t, n, p, ax) in normas:
        cita = f"(Anexo {ax}, p. {p})" if (varios_anexos and ax) else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {t} {n} {cita}")
    return "\n".join(out)


# ==================== Normalización de citas (documento único vs. multi-anexo) ====================
_CITA_ANEXO_RE = re.compile(r"\(Anexo\s+([IVXLCDM\d]+)(?:,\s*p\.\s*(\d+))?\)", re.I)

def _normalize_citas_salida(texto: str, varios_anexos: bool) -> str:
    """
    Si es documento único, convierte "(Anexo X, p. N)" en "(p. N)" o "(Fuente: documento provisto)".
    En multi-anexo, deja las citas tal cual.
    """
    if varios_anexos:
        return texto

    def repl(m):
        pag = m.group(2)
        if pag:
            return f"(p. {pag})"
        return "(Fuente: documento provisto)"

    return _CITA_ANEXO_RE.sub(repl, texto)


# ==================== Reemplazo determinístico de secciones 2.3 y 2.15 ====================
def _replace_section(text: str, header_regex: str, replacement: str) -> str:
    """
    Reemplaza el bloque que inicia en header_regex hasta el próximo encabezado '2.' o fin.
    Si no existe, lo agrega al final.
    """
    m = re.search(header_regex, text or "", flags=re.I)
    if not m:
        return (text or "").rstrip() + "\n\n" + replacement.strip() + "\n"

    start = m.start()
    nxt = re.search(r"(?im)^\s*2\.(1[0-9]|[1-9])\s", (text or "")[m.end():])
    end = m.end() + (nxt.start() if nxt else 0)
    if end <= m.end():
        return (text or "")[:start] + replacement.strip() + "\n" + (text or "")[m.end():]
    return (text or "")[:start] + replacement.strip() + "\n" + (text or "")[end:]


def _ampliar_contactos_y_normativa(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Sustituye SIEMPRE:
      - 2.3 Contactos y portales
      - 2.15 Normativa aplicable
    usando extractores determinísticos (sin depender del modelo).
    """
    out = informe
    sec23 = _build_section_23(texto_fuente, varios_anexos)
    if sec23:
        out = _replace_section(out, r"(?im)^\s*2\.3\s+Contactos", sec23)

    sec215 = _build_section_215(texto_fuente, varios_anexos)
    if sec215:
        out = _replace_section(out, r"(?im)^\s*2\.15\s+Normativa", sec215)

    return out
# ==================== Helpers de cita y capturas con página/anexo ====================
def _cita(p: Optional[int], ax: Optional[int], varios_anexos: bool) -> str:
    if p and varios_anexos and ax:
        return f"(Anexo {ax}, p. {p})"
    if p:
        return f"(p. {p})"
    return "(Fuente: documento provisto)"

def _buscar_primera_captura(texto: str, patrones: List[str]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Busca el primer patrón con 1 grupo de captura y devuelve (valor, pagina, anexo).
    """
    if not texto:
        return None, None, None
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)
    for patt in patrones:
        for m in re.finditer(patt, texto, flags=re.I):
            val = (m.group(1) or "").strip()
            pos = m.start(1)
            p = _pagina_de_indice(idx_pag, pos) if idx_pag else None
            ax = _anexo_en_pos(idx_ax, pos) if idx_ax else None
            if val:
                return val, p, ax
    return None, None, None

# ==================== Extractor (determinístico) para "Objeto del gasto" ====================
_OBJ_GASTO_PATTERNS = [
    r"objeto\s+del\s+gasto\s*[:\-]\s*([^\n;]+)",
    r"partida\s+presupuestaria\s*[:\-]\s*([^\n;]+)",
    r"clasificador(?:\s*/?\s*objeto\s*del\s*gasto|\s*del\s*gasto)?\s*[:\-]\s*([^\n;]+)",
]

def _extraer_objeto_gasto_con_cita(texto: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    val, p, ax = _buscar_primera_captura(texto, _OBJ_GASTO_PATTERNS)
    if val:
        # Mantenerlo corto y útil
        val = _truncate_words(val, 20)
    return val, p, ax

# ==================== Reconstrucción determinística de la FICHA (0) ====================
FICHA_LABELS = [
    "N° de proceso",
    "Nombre de proceso",
    "Objeto de la contratación",
    "Procedimiento de selección",
    "Tipo de cotización",
    "Tipo de adjudicación",
    "Cantidad de ofertas permitidas",
    "Estado",
    "Plazo de mantenimiento de la oferta",
    "Número de renglón",
    "Objeto del gasto",
    "Código del ítem",
    "Descripción",
    "Cantidad",
    "Inicio y final de consultas",
    "Fecha y hora del acto de apertura",
    "Monto",
    "Moneda",
    "Duración del contrato",
]

def _bounds_ficha(texto: str) -> Tuple[int, int]:
    """
    Devuelve (start, end) del bloque '0) Ficha ...' hasta el próximo encabezado top-level (1) o (2) o fin.
    """
    if not texto:
        return -1, -1
    m = re.search(r"(?im)^\s*0\)\s*Ficha\s+estandarizada[^\n]*$", texto)
    if not m:
        return -1, -1
    start = m.start()
    nxt = re.search(r"(?im)^\s*[12]\)\s", texto[m.end():])
    end = m.end() + (nxt.start() if nxt else 0)
    return start, end if end > start else (start, len(texto))

def _parse_ficha_existente(bloque: str) -> Dict[str, str]:
    """
    Lee valores existentes en la ficha (si el modelo los trajo) para no perderlos.
    """
    out: Dict[str, str] = {}
    if not bloque:
        return out
    for label in FICHA_LABELS:
        # • Label: valor  |  - Label: valor  |  Label: valor
        patt = rf"(?im)^\s*(?:[-*•]\s*)?{re.escape(label)}\s*:\s*(.+?)\s*$"
        m = re.search(patt, bloque)
        if m:
            val = (m.group(1) or "").strip()
            if val:
                out[label] = val
    return out

def _reconstruir_ficha_en_informe(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Construye SIEMPRE la ficha (0) de forma determinística.
    - Completa 'Número de renglón' con 'Total de renglones: N; ver Sección 9 ...'.
    - Si hay renglones: setea 'Código del ítem' / 'Descripción' / 'Cantidad' a 'Ver Sección 9 ...' (no "NO ESPECIFICADO").
    - Intenta extraer 'Objeto del gasto' con regex. El resto conserva lo que haya aportado el modelo o 'NO ESPECIFICADO'.
    """
    s, e = _bounds_ficha(informe)
    bloque_existente = informe[s:e] if (s != -1 and e != -1) else ""

    # Valores de la ficha que haya traído el modelo
    existentes = _parse_ficha_existente(bloque_existente)

    # Renglones desde el texto fuente
    rows = _extraer_renglones_y_especificaciones(texto_fuente)
    n_rows = len(rows)
    p0 = rows[0][4] if n_rows else None
    ax0 = rows[0][5] if n_rows else None

    # Objeto del gasto
    obj_gasto, p_g, ax_g = _extraer_objeto_gasto_con_cita(texto_fuente)

    # Armar dict final
    vals: Dict[str, str] = {}
    for lab in FICHA_LABELS:
        vals[lab] = existentes.get(lab, "NO ESPECIFICADO")

    # Número de renglón (punto 2)
    if n_rows:
        vals["Número de renglón"] = f"Total de renglones: {n_rows}; ver Sección 9 para el detalle completo {_cita(p0, ax0, varios_anexos)}"
        for lab in ("Código del ítem", "Descripción", "Cantidad"):
            # punto 3: no marcar NO ESPECIFICADO si está a nivel renglón
            vals[lab] = f"Ver Sección 9 para el detalle por renglón {_cita(p0, ax0, varios_anexos)}"
    else:
        vals["Número de renglón"] = f"Total de renglones: 0; ver Sección 9 para el detalle completo {_cita(None, None, varios_anexos)}"

    # Objeto del gasto (punto 3)
    if obj_gasto:
        vals["Objeto del gasto"] = f"{obj_gasto} {_cita(p_g, ax_g, varios_anexos)}"

    # Render de la ficha
    lines = ["0) Ficha estandarizada del procedimiento (campos estandarizados)"]
    for lab in FICHA_LABELS:
        val = vals.get(lab, "NO ESPECIFICADO").strip()
        # Evitar que queden instrucciones literales
        val = re.sub(r"“|”", '"', val)  # normaliza comillas
        if re.search(r"indicar\s*\"?Total de renglones", val, flags=re.I):
            # por si el modelo dejó el texto literal
            if "renglones" in lab.lower():
                # ya lo sobrescribimos arriba; por seguridad:
                val = vals["Número de renglón"]
            else:
                val = "NO ESPECIFICADO"
        lines.append(f"- {lab}: {val}")

    nuevo_bloque = "\n".join(lines) + "\n"

    if s == -1:
        # No había ficha; la preprendemos al documento
        return (nuevo_bloque + "\n" + (informe or "")).strip()

    return (informe[:s] + nuevo_bloque + informe[e:]).strip()


# ==================== Reemplazo opcional de 2.13 y 2.16 ====================
def _reemplazar_213_216(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    out = informe

    # 2.13 Planilla / renglones
    sec213 = _build_section_213(texto_fuente, varios_anexos)
    if sec213:
        # También sincroniza la sección 9 en wording si existiera
        alt213 = sec213.replace("2.13 Planilla de cotización y renglones:", "9) Renglones y planilla de cotización:")
        out = _replace_section(out, r"(?im)^\s*2\.13\s+Planilla", sec213)
        out = _replace_section(out, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla", alt213)

    # 2.16 Catálogo de artículos
    sec216 = _build_section_216(texto_fuente, varios_anexos)
    if sec216:
        out = _replace_section(out, r"(?im)^\s*2\.16\s+Cat[aá]logo\s+de\s+art", sec216)
        out = re.sub(r"(?im)^\s*(ANEXO|Anexo)\s*[-–—]?\s*Cat[aá]logo\s+de\s+art[^\n]*\n?", "", out)

    return out

# ==================== Post-procesado final del informe ====================
def _posprocesar_informe_final(informe_raw: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Cadena final de pasos que:
    - Limpia meta-texto y normaliza citas.
    - Corrección focalizada de 'NO ESPECIFICADO'.
    - Sustituye 2.3 y 2.15 de forma determinística.
    - (Opcional) Reemplaza 2.13 / 2.16 si está habilitado.
    - Reconstruye la FICHA (0) de forma determinística para evitar textos literales.
    - Prepara para PDF.
    """
    if not informe_raw:
        return "No se recibió contenido para analizar."

    out = _limpiar_meta(informe_raw)
    out = _normalize_citas_salida(out, varios_anexos)
    out = _segundo_pase_si_falta(out, texto_fuente, varios_anexos)

    # Siempre contactos y normativa determinísticos
    out = _ampliar_contactos_y_normativa(out, texto_fuente, varios_anexos)

    # Opcional: 2.13 / 2.16 determinísticos
    if FORCE_DETERMINISTIC_213_216 or EXPAND_SECTIONS_213_216:
        out = _reemplazar_213_216(out, texto_fuente, varios_anexos)

    # Ficha (0) reconstruida
    out = _reconstruir_ficha_en_informe(out, texto_fuente, varios_anexos)

    # Limpieza menor y formato PDF
    out = re.sub(r"(?im)^\s*informe\s+original\s*$", "", out)
    out = preparar_texto_para_pdf(out)
    return out

# ==================== (REEMPLAZA) Analizador principal con post-procesado robusto ====================
def analizar_con_openai(texto: str) -> str:
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    texto_len = len(texto)
    n_anexos = _contar_anexos(texto)
    varios_anexos = n_anexos >= 2
    prompt_maestro = _prompt_andres(varios_anexos)

    hints = _build_regex_hints(texto) if ENABLE_REGEX_HINTS else ""
    hints_block = f"\n\n=== HALLAZGOS AUTOMÁTICOS (snippets literales para verificación) ===\n{hints}\n" if hints else ""

    # Heurística two-stage en multi-anexo largo
    force_two_stage = (varios_anexos and texto_len >= MULTI_FORCE_TWO_STAGE_MIN_CHARS)

    # === Single-pass cuando aplica ===
    if (not varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS) or \
       (varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS_MULTI and not force_two_stage):
        t0 = _t()
        max_out = _max_out_for_text(texto)
        messages = [
            {"role": "system", "content": "Actuá como equipo experto en derecho administrativo argentino (ámbitos nacional, provincial y municipal) y compras públicas; redactor técnico-jurídico. Cero invenciones."},
            {"role": "user", "content": f"{prompt_maestro}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve SOLO el informe final (texto), sin preámbulos ni títulos de estas instrucciones."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = (resp.choices[0].message.content or "").strip()
            out = _posprocesar_informe_final(bruto, texto, varios_anexos)
            _log_tiempo("analizar_single_pass" + ("_multi" if varios_anexos else ""), t0)
            return out
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # === Dos etapas (chunking dinámico + concurrencia) ===
    chunk_size = _compute_chunk_size(texto_len)
    partes = _particionar(texto, chunk_size)

    # Seguridad: si quedó 1 parte por tamaño, reintenta single-pass
    if len(partes) == 1:
        t0 = _t()
        max_out = _max_out_for_text(texto)
        messages = [
            {"role": "system", "content": "Actuá como equipo experto en derecho administrativo argentino (ámbitos nacional, provincial y municipal) y compras públicas; redactor técnico-jurídico. Cero invenciones."},
            {"role": "user", "content": f"{prompt_maestro}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve SOLO el informe final (texto), sin preámbulos ni títulos de estas instrucciones."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = (resp.choices[0].message.content or "").strip()
            return _posprocesar_informe_final(bruto, texto, varios_anexos)
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # A) Notas intermedias (concurrente)
    notas_list = _generar_notas_concurrente(partes)
    notas_integradas = "\n".join(notas_list)

    # B) Síntesis final
    t0_sint = _t()
    max_out = _max_out_for_text(texto)
    messages_final = [
        {"role": "system", "content": "Actuá como equipo experto en derecho administrativo argentino (ámbitos nacional, provincial y municipal) y compras públicas; redactor técnico-jurídico. Cero invenciones."},
        {"role": "user", "content": f"""{_prompt_andres(varios_anexos)}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

{("=== HALLAZGOS AUTOMÁTICOS (snippets literales) ===\n" + _build_regex_hints(texto)) if ENABLE_REGEX_HINTS else ""}

👉 Integra TODO en un **solo informe**; deduplica; cita una vez por dato con todas las fuentes.
👉 Prohibido meta-comentarios de fragmentos. No imprimas títulos de estas instrucciones.
👉 Devuelve SOLO el informe final en texto.
"""}
    ]
    try:
        resp_final = _llamada_openai(messages_final, max_completion_tokens=max_out, model=_pick_model("sintesis"))
        bruto = (resp_final.choices[0].message.content or "").strip()
        out = _posprocesar_informe_final(bruto, texto, varios_anexos)
        _log_tiempo("sintesis_final", t0_sint)
        return out
    except Exception as e:
        # Aún en error, devolvemos notas limpias para depurar
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias (limpias):\n{_limpiar_meta(notas_integradas)}"
# ==================== Contactos + Normativa (determinístico, usado en posproceso) ====================
def _ampliar_contactos_y_normativa(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    out = informe
    sec23 = _build_section_23(texto_fuente, varios_anexos)
    if sec23:
        out = _replace_section(out, r"(?im)^\s*2\.3\s+Contactos", sec23)
    sec215 = _build_section_215(texto_fuente, varios_anexos)
    if sec215:
        out = _replace_section(out, r"(?im)^\s*2\.15\s+Normativa", sec215)
    return out


# ==================== Multi-anexo ====================
def analizar_anexos(files: list) -> str:
    """
    Combina anexos y ejecuta análisis.
    - 1 archivo: NO marca "=== ANEXO ... ===" para habilitar single-pass y citas (p. N).
    - ≥2: marca ANEXOS para trazabilidad. Si el total entra en MAX_SINGLE_PASS_CHARS_MULTI, puede ir single-pass salvo que force_two_stage.
    """
    if not files:
        return "No se recibieron anexos para analizar."

    t0 = _t()
    bloques = []
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
                "Verificá si los documentos están escaneados y elevá VISION_MAX_PAGES/DPI, "
                "o subí archivos en texto nativo.")

    contenido_unico = _limpieza_basica_preanalisis(contenido_unico)
    _log_tiempo("anexos_armado_y_limpieza", t0)

    return analizar_con_openai(contenido_unico)


# ==================== Chat ====================
def responder_chat_openai(mensaje: str, contexto: str = "", usuario: str = "Usuario") -> str:
    descripcion_interfaz = f"""
Sos el asistente de "Suizo Argentina - Licitaciones IA". Ayudás con pliegos y dudas de uso.
Usuario actual: {usuario}
"""
    if not contexto:
        contexto = "(No hay historial disponible.)"

    prompt = f"""
{descripcion_interfaz}

📂 Historial de análisis previos:
{contexto}

🧠 Pregunta del usuario:
{mensaje}

Respondé natural y directo. Evitá repetir las funciones de la plataforma.
"""

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_CHAT", _pick_model("analisis")),
            messages=[
                {"role": "system", "content": "Asistente experto en licitaciones y soporte de plataforma."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1200
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Error al generar respuesta: {e}"


# ==================== PDF ====================
def _render_pdf_bytes(resumen: str, fecha_display: Optional[str] = None) -> bytes:
    """
    Renderiza el PDF. Si `fecha_display` viene informada (ej: '31/05/2025 14:03'),
    la usa tal cual. Si no, usa hora local de AR para evitar corrimientos.
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

    # Fecha en encabezado: prioriza la pasada por parámetro (AR), sino cae en local AR.
    if not fecha_display:
        try:
            fecha_display = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).strftime("%d/%m/%Y %H:%M")
        except Exception:
            fecha_display = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.drawCentredString(A4[0] / 2, A4[1] - 42 * mm, f"{fecha_display}")

    # Filtros de rótulos indeseados
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
            y -= alto_linea  # espacio entre párrafos / títulos
            continue
        # Heurística de títulos
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
        # espacio extra tras títulos
        if parrafo.strip().endswith(":") or parrafo.isupper() or re.match(r"^\d+(\.\d+)*\s", parrafo):
            y -= alto_linea // 2

    c.save()
    return buffer.getvalue()


def generar_pdf_con_plantilla(resumen: str, nombre_archivo: str, fecha_display: Optional[str] = None):
    """
    Genera el PDF en generated_pdfs/{nombre_archivo}
    - `fecha_display`: string ya formateado (DD/MM/YYYY HH:MM) que se imprime en el header.
                       Si no se pasa, se usa la hora local de Argentina.
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
    palabras = texto.split(" ")
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
