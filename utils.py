# utils.py
import io
import os
import re
import base64
import mimetypes
import time
from datetime import datetime
from typing import List, Tuple, Dict
from tempfile import NamedTemporaryFile

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor

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
# Defaults elevados para minimizar truncamientos
MAX_COMPLETION_TOKENS_SALIDA = int(os.getenv("MAX_COMPLETION_TOKENS_SALIDA", "8000"))
TEMPERATURE_ANALISIS = os.getenv("TEMPERATURE_ANALISIS", "").strip()
ANALISIS_MODO = os.getenv("ANALISIS_MODO", "").lower().strip()  # "fast" opcional

# Concurrencia
ANALISIS_CONCURRENCY = int(os.getenv("ANALISIS_CONCURRENCY", "3"))
NOTAS_MAX_TOKENS = int(os.getenv("NOTAS_MAX_TOKENS", "2000"))

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

# ---- OCR selectivo en paralelo ----
from concurrent.futures import ThreadPoolExecutor, as_completed

def _ocr_pagina_png_bytes(png_bytes: bytes, idx: int) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    txt = _ocr_openai_imagen_b64(b64)
    return f"[PÁGINA {idx+1}]\n{txt}" if txt else f"[PÁGINA {idx+1}] (sin texto OCR)"

def _ocr_selectivo_por_pagina(doc: fitz.Document, max_pages: int) -> str:
    n = len(doc)
    to_process = min(n, max_pages)
    resultados = [""] * to_process
    tareas = []

    with ThreadPoolExecutor(max_workers=OCR_CONCURRENCY) as ex:
        for i in range(to_process):
            p = doc.load_page(i)
            txt_nat = (p.get_text() or "").strip()
            if len(txt_nat) >= OCR_TEXT_MIN_CHARS:
                resultados[i] = f"[PÁGINA {i+1}]\n{txt_nat}"
            else:
                png_bytes = _rasterizar_pagina(p)
                tareas.append(ex.submit(_ocr_pagina_png_bytes, png_bytes, i))

        for fut in as_completed(tareas):
            try:
                s = fut.result()
                m = re.search(r"\[PÁGINA\s+(\d+)\]", s)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < to_process:
                        resultados[idx] = s
            except Exception:
                pass

    if n > to_process:
        resultados.append(f"\n[AVISO] Se procesaron {to_process}/{n} páginas por OCR selectivo.")
    return "\n\n".join([r for r in resultados if r]).strip()

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
[Guía de mapeo semántico]
- "Fecha de publicación" ≈ "fecha del llamado", "fecha de difusión del llamado", "fecha de convocatoria".
- "Número de proceso" ≈ "Expediente", "N° de procedimiento", "EX-...", "IF-...".
- "Presupuesto referencial" ≈ "presupuesto oficial", "monto estimado", "crédito disponible".
- "Presentación de ofertas" ≈ "acto de presentación", "límite de recepción".
- "Apertura" ≈ "acto de apertura de ofertas".
- "Mantenimiento de oferta" ≈ "validez de la oferta".
- "Garantía de cumplimiento" ≈ "garantía contractual".
- "Planilla de cotización" ≈ "formulario de oferta", "cuadro comparativo", "planilla de precios".
- "Tipo de cambio BNA" ≈ "Banco Nación vendedor del día anterior".
Usa esta guía: si un campo aparece con sinónimos/variantes, NO lo marques como "no especificado".
"""

# <<< Secciones útiles: se mantiene 2.1 → 2.15 y se agrega 2.16 Artículos >>>
_BASE_PROMPT_MAESTRO = r"""
# (Instrucciones internas: NO imprimir este encabezado ni estas reglas en la salida)
Reglas clave:
- Cero invenciones; si falta o es ambiguo: escribir "NO ESPECIFICADO" y explicarlo en la misma sección.
- Cada dato crítico debe terminar con su fuente entre paréntesis, según las Reglas de Citas.
- Cobertura completa (oferta → ejecución), con normativa citada.
- Deduplicar, fusionar, no repetir; un único informe integrado.
- Prohibido meta texto tipo "parte X de Y" o "revise el resto".
- No imprimir etiquetas internas como [PÁGINA N].
- No usar los títulos literales "Informe Completo" ni "Informe Original".

Formato de salida:
1) RESUMEN DE PLIEGO (≤200 palabras)
2) INFORME DETALLADO CON TRAZABILIDAD
   2.1 Identificación del llamado
   2.2 Calendario y lugares
   2.3 Contactos y portales (listar TODOS los e-mails y URLs detectados)
   2.4 Alcance y plazo contractual
   2.5 Tipología / modalidad (citar norma/artículos)
   2.6 Mantenimiento de oferta y prórroga
   2.7 Garantías (umbral UC, %, plazos, formas)
   2.8 Presentación de ofertas (soporte, firmas, docs obligatorias) e incluir costo/valor del pliego y mecanismo de adquisición/pago
   2.9 Apertura, evaluación y adjudicación (tipo de cambio BNA, comisión, criterios, preferencias)
   2.10 Subsanación (qué sí/no)
   2.11 Perfeccionamiento y modificaciones
   2.12 Entrega, lugares y plazos
   2.13 Planilla de cotización y renglones (enumerar TODOS los renglones; por renglón incluir cantidades, UM, descripción y **especificaciones técnicas** relevantes)
   2.14 Muestras
   2.15 Normativa aplicable (todas las leyes/decretos/resoluciones/disposiciones citadas, con número/año y fuente)
   2.16 Catálogo de artículos citados (Art. N — **síntesis literal 1–2 líneas del contenido**; una línea por artículo; con cita)

Estilo:
- Títulos con mayúsculas iniciales, listas claras, tablas simples. Sin "#".
- Aplicar la Guía de sinónimos.
"""

def _prompt_maestro(varios_anexos: bool) -> str:
    if varios_anexos:
        regla_citas = (
            "Reglas de Citas:\n"
            "- Al final de cada línea con dato, usar (Anexo X, p. N).\n"
            "- Para deducir p. N, utiliza la etiqueta [PÁGINA N] más cercana al dato dentro del texto provisto de ese ANEXO.\n"
            "- Si NO consta paginación pero sí el anexo, usar (Anexo X).\n"
            "- Si el campo es NO ESPECIFICADO, usar (Fuente: documento provisto) (no inventar página/anexo).\n"
        )
    else:
        regla_citas = (
            "Reglas de Citas:\n"
            "- Documento único: al final de cada línea con dato, usar (p. N).\n"
            "- Para deducir p. N, utiliza la etiqueta [PÁGINA N] más cercana al dato dentro del texto provisto.\n"
            "- Prohibido escribir 'Anexo I' u otros anexos en las citas.\n"
            "- Si el campo es NO ESPECIFICADO, usar (Fuente: documento provisto) (no inventar página).\n"
        )
    extras = (
        "\nCriterios anti-omisión:\n"
        "- En 'Contactos y portales': incluir absolutamente todos los e-mails/dominos/URLs detectados.\n"
        "- En 'Planilla de cotización y renglones': enumerar todos los renglones y sumar especificaciones técnicas por renglón.\n"
        "- En 'Normativa aplicable': listar todas las normas mencionadas (Ley/Decreto/Resolución/Disposición, número y año).\n"
        "- En 'Catálogo de artículos citados': incluir cada artículo que figure, con síntesis literal 1–2 líneas.\n"
    )
    return f"{_BASE_PROMPT_MAESTRO}\n{regla_citas}{extras}\nGuía de sinónimos:\n{SINONIMOS_CANONICOS}"

CRAFT_PROMPT_NOTAS = r"""
Genera NOTAS INTERMEDIAS en bullets, ultra concisas, con cita al final de cada bullet.
- SOLO bullets (sin encabezados, sin "parte x/y", sin conclusiones).
- Etiqueta tema + cita en paréntesis.
- Si NO hay paginación: (Fuente: documento provisto).
- Usa la Guía de sinónimos y conserva la terminología encontrada.
Ejemplos:
- [IDENTIFICACION] Organismo: ... (p. 1)
- [CALENDARIO] Presentación: DD/MM/AAAA HH:MM — Lugar: ... (p. 2)
- [GARANTIAS] Mant. 5%; Cumpl. ≥10% ≤7 días hábiles (p. 4)
- [CONTACTO] Email ... / Portal ... (p. 2)
- [COSTO PLIEGO] Valor $... — medio de pago: ... (p. N)
- [PRESUPUESTO] Monto: $... (p. N)
- [PLANILLA/RENGLONES] Renglón X: ... (p. N)
- [ESPEC TECNICAS] Renglón X: requisito ... (p. N)
- [NORMATIVA] Ley/Decreto/Resolución ... (p. N)
- [ARTICULO] Art. 17 — síntesis ... (p. N)
- [FALTA] campo X — NO ESPECIFICADO. (Fuente: documento provisto)
"""

# Filtrado de metafrases y títulos no deseados
_META_PATTERNS = [
    re.compile(r"(?i)\bparte\s+\d+\s+de\s+\d+"),
    re.compile(r"(?i)informe\s+basado\s+en\s+la\s+parte"),
    re.compile(r"(?i)revise\s+las\s+partes\s+restantes"),
    re.compile(r"(?i)información\s+puede\s+estar\s+incompleta"),
    re.compile(r"(?i)^\s*informe\s+completo\s*$"),
    re.compile(r"(?i)^\s*informe\s+original\s*$"),
]

def _limpiar_meta(texto: str) -> str:
    lineas = []
    for ln in texto.splitlines():
        if any(p.search(ln) for p in _META_PATTERNS):
            continue
        lineas.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()

def _particionar(texto: str, max_chars: int) -> list[str]:
    return [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]

_ANEXO_RE = re.compile(r"===\s*ANEXO\s+\d+", re.I)
def _contar_anexos(s: str) -> int:
    return len(_ANEXO_RE.findall(s or ""))

# =============== Post-procesamiento de citas para documento único ===============
_CITA_ANEXO_RE = re.compile(r"\(Anexo\s+([IVXLCDM\d]+)(?:,\s*p\.\s*(\d+))?\)", re.I)
def _normalize_citas_salida(texto: str, varios_anexos: bool) -> str:
    if varios_anexos:
        return texto
    def repl(m):
        pag = m.group(2)
        if pag:
            return f"(p. {pag})"
        return "(Fuente: documento provisto)"
    return _CITA_ANEXO_RE.sub(repl, texto)

# ===== Secciones obligatorias y helpers (relleno de faltantes) =====
SECCIONES_OBLIGATORIAS = [
    ("2.1", "Identificación del llamado"),
    ("2.2", "Calendario y lugares"),
    ("2.3", "Contactos y portales"),
    ("2.4", "Alcance y plazo contractual"),
    ("2.5", "Tipología / modalidad"),
    ("2.6", "Mantenimiento de oferta y prórroga"),
    ("2.7", "Garantías"),
    ("2.8", "Presentación de ofertas"),
    ("2.9", "Apertura, evaluación y adjudicación"),
    ("2.10", "Subsanación"),
    ("2.11", "Perfeccionamiento y modificaciones"),
    ("2.12", "Entrega, lugares y plazos"),
    ("2.13", "Planilla de cotización y renglones"),
    ("2.14", "Muestras"),
    ("2.15", "Normativa aplicable"),
    ("2.16", "Catálogo de artículos citados"),
]

_SECCION_KEY_RE = re.compile(r"(?m)^\s*(2\.(?:1|2|3|4|5|6|7|8|9|1[0-6]))\s+([^\n]+)")

def _secciones_presentes(texto: str) -> dict:
    presentes = {}
    for m in _SECCION_KEY_RE.finditer(texto or ""):
        key = m.group(1).strip()
        titulo = (m.group(2) or "").strip()
        presentes[key] = titulo
    return presentes

def _prompt_seccion_focalizada(key: str, titulo: str, varios_anexos: bool, hints: str) -> str:
    if varios_anexos:
        regla_citas = (
            "Usa citas (Anexo X, p. N). Si no hay paginación, usa (Anexo X). "
            "No inventes páginas/anexos. "
        )
    else:
        regla_citas = (
            "Usa citas (p. N) del documento único. Si no hay paginación, usa (Fuente: documento provisto). "
        )

    refuerzo = ""
    if key == "2.13":
        refuerzo = (
            "Enumera TODOS los renglones sin agrupar ni recortar; por renglón incluye: número, "
            "descripción, cantidad, unidad de medida, y las **especificaciones técnicas** literales relevantes. "
            "Si hay planilla de cotización, refleja la información completa. "
        )
    elif key == "2.16":
        refuerzo = (
            "Incluye cada Art. N citado y agrega una síntesis literal de 1–2 líneas por artículo, una línea por artículo. "
        )
    elif key == "2.3":
        refuerzo = "Incluye absolutamente TODOS los emails y URLs encontrados. "
    elif key == "2.8":
        refuerzo = "Incluye costo/valor del pliego y mecanismo de adquisición/pago si constan. "
    elif key == "2.15":
        refuerzo = "Lista todas las leyes/decretos/resoluciones/disposiciones con número y año. "

    return (
        f"(Generación focalizada de sección) Redacta únicamente la sección {key} {titulo}.\n"
        f"Cero invenciones. {regla_citas}"
        f"{refuerzo}"
        "Mantén estilo de listas claras y una sola cita por dato.\n"
        f"{'=== HALLAZGOS AUTOMÁTICOS ===\n' + hints if hints else ''}\n"
        "Devuelve solo el texto final de la sección, sin encabezados extra, sin preámbulos."
    )

def _generar_seccion_faltante(key: str, titulo: str, texto_fuente: str, varios_anexos: bool, hints: str) -> str:
    msg = [
        {"role": "system", "content": "Redactor técnico-jurídico; cero invenciones; usa citas literales."},
        {"role": "user", "content": f"{_prompt_seccion_focalizada(key, titulo, varios_anexos, hints)}\n\n=== CONTENIDO COMPLETO ===\n{texto_fuente}"}
    ]
    try:
        r = _llamada_openai(msg, model=_pick_model("sintesis"))
        sec = (r.choices[0].message.content or "").strip()
        return preparar_texto_para_pdf(_limpiar_meta(_normalize_citas_salida(sec, varios_anexos)))
    except Exception:
        return ""

def _rellenar_secciones_faltantes(reporte: str, texto_fuente: str, varios_anexos: bool, hints: str) -> str:
    presentes = _secciones_presentes(reporte)
    piezas = [reporte.rstrip(), ""]
    for key, titulo in SECCIONES_OBLIGATORIAS:
        if key not in presentes:
            sec_text = _generar_seccion_faltante(key, titulo, texto_fuente, varios_anexos, hints)
            if sec_text:
                piezas.append(f"{key} {titulo}\n{sec_text}\n")
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join([p for p in piezas if p])).strip()

# ==================== Normalización para PDF (sin '#') ====================
_HDR_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```.*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_ITALIC_RE = re.compile(r"(\*\*|\*|__|_)(.*?)\1")

def _title_case(s: str) -> str:
    return " ".join(w.capitalize() if w else w for w in re.split(r"(\s+)", s))

def preparar_texto_para_pdf(markdown_text: str) -> str:
    out_lines: List[str] = []
    for raw_ln in (markdown_text or "").splitlines():
        ln = raw_ln.rstrip()
        if _CODE_FENCE_RE.match(ln):
            continue
        # filtra títulos indeseados
        if re.match(r"(?i)^\s*informe\s+completo\s*$", ln):
            continue
        if re.match(r"(?i)^\s*informe\s+original\s*$", ln):
            continue
        m = _HDR_RE.match(ln)
        if m:
            titulo = _title_case(m.group(2).strip(": ").strip())
            out_lines.append(titulo)
            out_lines.append("")  # espacio tras título
            continue
        if _TABLE_SEP_RE.match(ln):
            continue
        if _BULLET_RE.match(ln):
            ln = _BULLET_RE.sub("• ", ln)
        ln = _LINK_RE.sub(lambda mm: f"{mm.group(1)} ({mm.group(2)})", ln)
        ln = _BOLD_ITALIC_RE.sub(lambda mm: mm.group(2), ln)
        out_lines.append(ln)
        if ln.strip().endswith(":"):
            out_lines.append("")  # espacio extra tras línea-título
    texto = "\n".join(out_lines)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto

# ==================== Hints regex (recall) ====================
_PAG_TAG_RE = re.compile(r"\[PÁGINA\s+(\d+)\]")
def _index_paginas(s: str) -> List[Tuple[int,int]]:
    return [(m.start(), int(m.group(1))) for m in _PAG_TAG_RE.finditer(s)]

def _pagina_de_indice(indices: List[Tuple[int,int]], pos: int) -> int:
    last = 1
    for i, p in indices:
        if i <= pos: last = p
        else: break
    return last

# Campos detectables (ampliados)
DETECTABLE_FIELDS: Dict[str, Dict] = {
    "mant_oferta": {"label":"Mantenimiento de oferta", "pats":[r"mantenim[ií]ento de la oferta", r"validez de la oferta"]},
    "gar_mant":    {"label":"Garantía de mantenimiento", "pats":[r"garant[ií]a.*manten", r"\b5 ?%"]},
    "gar_cumpl":   {"label":"Garantía de cumplimiento", "pats":[r"garant[ií]a.*cumpl", r"\b10 ?%"]},
    "plazo_ent":   {"label":"Plazo de entrega", "pats":[r"plazo de entrega", r"\b\d{1,3}\s*d[ií]as"]},
    "tipo_cambio": {"label":"Tipo de cambio BNA", "pats":[r"Banco\s+Naci[oó]n", r"tipo de cambio"]},
    "comision":    {"label":"Comisión de (Pre)?Adjudicación", "pats":[r"Comisi[oó]n.*(pre)?adjudicaci[oó]n"]},
    "muestras":    {"label":"Muestras", "pats":[r"\bmuestras?\b"]},
    "planilla":    {"label":"Planilla de cotización y renglones", "pats":[r"planilla.*cotizaci[oó]n", r"renglones?"]},
    "modalidad":   {"label":"Modalidad / art. 17", "pats":[r"Orden de compra cerrada", r"art[ií]culo\s*17"]},
    "plazo_contr": {"label":"Plazo contractual", "pats":[r"por el t[eé]rmino\s+de\s+\d+", r"\b185\s*d[ií]as"]},
    "prorroga":    {"label":"Prórroga", "pats":[r"pr[oó]rroga\s+de\s+hasta\s+el\s+100%"]},
    "presupuesto": {"label":"Presupuesto estimado", "pats":[r"presupuesto (estimado|oficial|referencial)", r"\$\s?\d{1,3}(\.\d{3})*(,\d{2})?"]},
    "expediente":  {"label":"Expediente", "pats":[r"\bEX-\d{4}-[A-Z0-9-]+"]},
    "fechas":      {"label":"Fechas y horas", "pats":[r"\b\d{2}/\d{2}/\d{4}\b", r"\b\d{1,2}:\d{2}\s*(hs|h)"]},
    "contacto":    {"label":"Contactos y portales", "pats":[r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", r"https?://[^\s)]+|www\.[^\s)]+"]},
    "costo_pliego":{"label":"Costo/valor del pliego", "pats":[r"(costo|valor)\s+del\s+pliego", r"adquisici[oó]n\s+del\s+pliego", r"\$\s?\d{1,3}(\.\d{3})*(,\d{2})?"]},
    "subsanacion": {"label":"Subsanación", "pats":[r"subsanaci[oó]n"]},
    "perf_modif":  {"label":"Perfeccionamiento/Modificaciones", "pats":[r"perfeccionamiento", r"modificaci[oó]n"]},
    "preferencias":{"label":"Preferencias", "pats":[r"preferencias"]},
    "criterios":   {"label":"Criterios de evaluación", "pats":[r"criterios?\s+de\s+evaluaci[oó]n"]},
    "renglones":   {"label":"Renglones y especificaciones", "pats":[r"Rengl[oó]n\s*\d+", r"Especificaciones?\s+t[ée]cnicas?"]},
    "articulos":   {"label":"Artículos citados", "pats":[r"\bArt(?:[íi]culo|\.)\s*\d+[A-Za-z]?\b"]},
    "normativa":   {"label":"Normativa aplicable", "pats":[
                        r"\bLey(?:\s*N[°º])?\s*\d{1,5}(?:\.\d{1,3})*(?:/\d{2,4})?",
                        r"\bDecreto(?:\s*N[°º])?\s*\d{1,5}(?:/\d{2,4})?",
                        r"\bResoluci[oó]n(?:\s*(?:Ministerial|Conjunta))?\s*(?:N[°º]\s*)?\d{1,6}(?:/\d{2,4})?",
                        r"\bDisposici[oó]n\s*(?:N[°º]\s*)?\d{1,6}(?:/\d{2,4})?"
                    ]},
}

# ====== NUEVO: utilidades para conteo y evidencia exhaustiva ======
def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))

_ART_HEAD_RE = re.compile(r"(?im)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.*)$")
_ART_BLOCK_RE = re.compile(
    r"(?ims)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.+?)(?=^\s*art(?:[íi]culo|\.?)\s*\d+[a-zº°]?|\Z)"
)

def _extraer_articulos_con_snippets(texto: str) -> List[Tuple[str, str, int]]:
    """
    Devuelve lista de (rótulo_articulo, snippet_200c, pagina_aprox)
    """
    idx = _index_paginas(texto)
    res = []
    for m in _ART_BLOCK_RE.finditer(texto):
        start = m.start()
        p = _pagina_de_indice(idx, start)
        rotulo = m.group(1).strip()
        contenido = (m.group(2) or "").strip()
        snippet = contenido[:200].replace("\n", " ").strip()
        res.append((rotulo, snippet, p))
    if not res:
        for m in _ART_HEAD_RE.finditer(texto):
            start = m.start()
            p = _pagina_de_indice(idx, start)
            rotulo = m.group(1).strip()
            snippet = (m.group(2) or "").strip()[:200].replace("\n", " ")
            res.append((rotulo, snippet, p))
    return res

_RENGLON_RE = re.compile(r"(?im)^\s*rengl[oó]n\s*(\d+)\s*[-–—:]?\s*(.*)$")

def _extraer_renglones_y_especificaciones(texto: str) -> List[Tuple[str, str, int]]:
    """
    Devuelve lista de (renglón_id, snippet_260c, pagina_aprox)
    Toma la línea de encabezado del renglón y un poco de contexto inmediato.
    """
    idx = _index_paginas(texto)
    res = []
    for m in _RENGLON_RE.finditer(texto):
        start = m.start()
        p = _pagina_de_indice(idx, start)
        encabezado = f"Renglón {m.group(1)}"
        tail_start = m.end()
        tail = texto[tail_start: tail_start + 600]
        prox = _RENGLON_RE.search(tail)
        if prox:
            tail = tail[:prox.start()]
        snippet = (m.group(2) + " " + tail).strip()
        snippet = re.sub(r"\s+", " ", snippet)[:260]
        res.append((encabezado, snippet, p))
    return res

def _construir_evidencia_ampliacion(texto: str) -> Tuple[str, int, int]:
    """
    Arma bloques de evidencia literal (con páginas) para renglones/planilla y artículos.
    Devuelve (bloque_evidencia, cant_renglones, cant_articulos).
    """
    renglones = _extraer_renglones_y_especificaciones(texto)
    articulos = _extraer_articulos_con_snippets(texto)

    ev_parts = []
    if renglones:
        ev_parts.append("### EVIDENCIA Renglones / Planilla (literal)\n" + "\n".join(
            [f"- {rid}: {sn} (p. {p})" for (rid, sn, p) in renglones]
        ))
    if articulos:
        ev_parts.append("### EVIDENCIA Artículos (literal)\n" + "\n".join(
            [f"- {rot}: {sn} (p. {p})" for (rot, sn, p) in articulos]
        ))

    return ("\n\n".join(ev_parts) if ev_parts else ""), len(renglones), len(articulos)

def _conteo_en_informe(informe: str) -> Tuple[int, int]:
    return _count(r"(?im)\brengl[oó]n\s*\d+", informe), _count(r"(?im)\bart(?:[íi]culo|\.?)\s*\d+", informe)

def _max_out_for_text(texto: str) -> int:
    """
    Ajusta el tope de salida según volumen y cantidad de renglones/artículos detectados.
    """
    base_chars = len(texto or "")
    r_count = _count(r"(?im)^\s*rengl[oó]n\s*\d+", texto)
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

def _ampliar_secciones_especificas(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Reemplaza/expande 2.13 (Planilla/Renglones) y 2.16 (Artículos) si el informe quedó corto.
    Usa evidencia literal detectada por regex para obligar a un listado exhaustivo.
    """
    evidencia, total_ren, total_art = _construir_evidencia_ampliacion(texto_fuente)
    if not evidencia:
        return informe

    en_informe_ren, en_informe_art = _conteo_en_informe(informe)

    debe_ampliar = (
        (total_ren and (en_informe_ren < max(1, total_ren - 1))) or
        (total_art and (en_informe_art < max(1, total_art - 1))) or
        (not re.search(r"(?im)^2\.13\s+Planilla", informe)) or
        (not re.search(r"(?im)^2\.16\s+Cat[aá]logo de art", informe))
    )
    if not debe_ampliar:
        return informe

    prompt = f"""
(Reforzador de cobertura) Sustituye y/o expande SOLO las secciones:
- 2.13 Planilla de cotización y renglones
- 2.16 Catálogo de artículos citados

Requisitos:
- 2.13: enumera TODOS los renglones detectados (una línea por renglón) con cantidades/UM/descripcion y **especificaciones técnicas** relevantes si aparecen; sin agrupar ni resumir.
- 2.16: lista TODOS los artículos citados como "Art. N — síntesis literal 1–2 líneas".
- Cita al final de cada línea: usa (Anexo X, p. N) o (p. N) según corresponda.
- NO alteres ninguna otra sección del informe. Mantén exactamente el resto del texto tal cual.
- NO imprimas encabezados tipo “Informe Original”.

=== INFORME ACTUAL ===
{informe}

=== EVIDENCIA LITERAL PARA AMPLIAR ===
{evidencia}
"""
    try:
        resp = _llamada_openai(
            [
                {"role": "system", "content": "Redactor técnico-jurídico. Expande solo 2.13 y 2.16 con listados exhaustivos y citas."},
                {"role": "user", "content": prompt},
            ],
            model=_pick_model("sintesis"),
            max_completion_tokens=_max_out_for_text(texto_fuente)
        )
        out = (resp.choices[0].message.content or "").strip()
        out = _normalize_citas_salida(_limpiar_meta(out), varios_anexos)
        out = re.sub(r"(?im)^\s*informe\s+original\s*$", "", out)
        return out
    except Exception:
        return informe.rstrip() + "\n\nANEXO — Evidencia literal (ampliación 2.13/2.16):\n" + evidencia

def _buscar_candidatos(texto: str, pats: List[str], idx_pag: List[Tuple[int,int]], limit: int) -> List[str]:
    hits = []
    for pat in pats:
        for m in re.finditer(pat, texto, flags=re.I):
            pos = m.start()
            p = _pagina_de_indice(idx_pag, pos)
            start = max(0, pos - 160)
            end = min(len(texto), pos + 240)
            snippet = texto[start:end].replace("\n", " ").strip()
            hits.append(f"- p. {p}: {snippet}")
            if len(hits) >= limit:
                return hits
    return hits[:limit]

def _build_regex_hints(texto: str, limit_per_field: int = None, max_chars: int = None) -> str:
    if not texto: return ""
    if limit_per_field is None: limit_per_field = HINTS_PER_FIELD
    if max_chars is None: max_chars = HINTS_MAX_CHARS
    idx_pag = _index_paginas(texto)
    secciones = []
    for key, meta in DETECTABLE_FIELDS.items():
        hits = _buscar_candidatos(texto, meta["pats"], idx_pag, limit_per_field)
        if hits:
            secciones.append(f"[{meta['label']}]\n" + "\n".join(hits))
        if sum(len(s) for s in secciones) > max_chars:
            break
    return "\n\n".join(secciones[:])

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
    """
    stage_default: 'analisis' | 'notas' | 'sintesis'
    Aplica FAST_FORCE_MODEL si corresponde.
    """
    if ANALISIS_MODO == "fast" and FAST_FORCE_MODEL:
        return FAST_FORCE_MODEL
    if stage_default == "notas":
        return MODEL_NOTAS
    if stage_default == "sintesis":
        return MODEL_SINTESIS
    return MODEL_ANALISIS

def _llamada_openai(messages, model=None, temperature_str=TEMPERATURE_ANALISIS,
                    max_completion_tokens=None, retries=2, fallback_model="gpt-4o-mini"):
    mdl = model or _pick_model("analisis")

    def _build_kwargs(m):
        kw = dict(model=m, messages=messages, max_completion_tokens=max_completion_tokens or MAX_COMPLETION_TOKENS_SALIDA)
        if ANALISIS_MODO == "fast":
            kw["temperature"] = 0
        elif temperature_str != "":
            try:
                kw["temperature"] = float(temperature_str)
            except:
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
    raise RuntimeError(str(last_error) if last_error else "Fallo desconocido en _llamada_openai")

# ==================== Concurrencia para NOTAS ====================
def _compute_chunk_size(total_chars: int) -> int:
    if TARGET_PARTS <= 0:
        return CHUNK_SIZE_BASE
    ideal = (total_chars + TARGET_PARTS - 1) // TARGET_PARTS
    return max(CHUNK_SIZE_BASE, ideal)

def _generar_notas_concurrente(partes: List[str]) -> List[str]:
    resultados = [None] * len(partes)
    t0 = _t()

    def worker(idx: int, parte: str):
        msg = [
            {"role": "system", "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user", "content": f"{CRAFT_PROMPT_NOTAS}\n\n## Guía de sinónimos/normalización\n{SINONIMOS_CANONICOS}\n\n=== FRAGMENTO {idx+1}/{len(partes)} ===\n{parte}"}
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
    return resultados

# ==================== Segundo pase (opcional y focalizado) ====================
_NOESP_RE = re.compile(r"(?i)\bNO ESPECIFICADO\b")
def _posibles_paginas_para(clave: str, texto: str) -> List[int]:
    idx = _index_paginas(texto)
    pags = set()
    for pat in DETECTABLE_FIELDS.get(clave, {}).get("pats", []):
        for m in re.finditer(pat, texto, flags=re.I):
            pos = m.start()
            pags.add(_pagina_de_indice(idx, pos))
    return sorted(pags)

def _segundo_pase_si_falta(original_report: str, texto_fuente: str, varios_anexos: bool) -> str:
    if not ENABLE_SECOND_PASS_COMPLETION:
        return original_report
    if not _NOESP_RE.search(original_report):
        return original_report

    evidencia = []
    for clave, meta in DETECTABLE_FIELDS.items():
        label = meta["label"]
        if re.search(rf"{re.escape(label)}.*NO ESPECIFICADO", original_report, flags=re.I) or \
           re.search(rf"{re.escape(label)}\s*:\s*NO ESPECIFICADO", original_report, flags=re.I):
            hits = _buscar_candidatos(texto_fuente, meta["pats"], _index_paginas(texto_fuente), 10)
            if hits:
                evidencia.append(f"### {label}\n" + "\n".join(hits))
    if not evidencia:
        return original_report

    prompt_corr = f"""
(Revisión focalizada) Completa ÚNICAMENTE los campos marcados como "NO ESPECIFICADO" en el informe,
usando SOLO la evidencia literal que te paso abajo. Mantén exactamente la estructura y secciones del
informe original, sin agregar nuevas secciones. Donde la evidencia sea ambigua, deja "NO ESPECIFICADO".
Respeta las reglas de citas del informe original (usa (Anexo X, p. N) o (p. N) según corresponda).
NO imprimas los rótulos de bloques como 'Informe Original' o similares.

=== CONTENIDO A CORREGIR (NO IMPRIMIR ESTE TÍTULO) ===
{original_report}

=== EVIDENCIA LITERAL (snippets con páginas) ===
{'\n\n'.join(evidencia)}
"""
    try:
        resp = _llamada_openai(
            [{"role": "system", "content": "Actúa como redactor técnico-jurídico, cero invenciones; corrige campos faltantes con citas."},
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

# ==================== Analizador principal ====================
def analizar_con_openai(texto: str) -> str:
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    texto_len = len(texto)
    n_anexos = _contar_anexos(texto)
    varios_anexos = n_anexos >= 2
    prompt_maestro = _prompt_maestro(varios_anexos)

    # Hints regex (opcionales, capados por tamaño)
    hints = _build_regex_hints(texto) if ENABLE_REGEX_HINTS else ""
    hints_block = f"\n\n=== HALLAZGOS AUTOMÁTICOS (snippets literales para verificación, NO resumir renglones) ===\n{hints}\n" if hints else ""

    # ¿forzar dos etapas en multi-anexo grande?
    force_two_stage = (varios_anexos and texto_len >= MULTI_FORCE_TWO_STAGE_MIN_CHARS)

    # === Single-pass cuando aplica ===
    if (not varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS) or \
       (varios_anexos and texto_len <= MAX_SINGLE_PASS_CHARS_MULTI and not force_two_stage):
        t0 = _t()
        max_out = _max_out_for_text(texto)
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{prompt_maestro}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve SOLO el informe final (texto), sin preámbulos ni títulos de estas instrucciones."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = resp.choices[0].message.content.strip()
            bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
            bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
            # Completar secciones ausentes y luego ampliar 2.13/2.16
            bruto = _rellenar_secciones_faltantes(bruto, texto, varios_anexos, hints)
            bruto = _ampliar_secciones_especificas(bruto, texto, varios_anexos)
            out = preparar_texto_para_pdf(bruto)
            _log_tiempo("analizar_single_pass" + ("_multi" if varios_anexos else ""), t0)
            return out
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # === Dos etapas (chunking dinámico + concurrencia) ===
    chunk_size = _compute_chunk_size(texto_len)
    partes = _particionar(texto, chunk_size)

    # Seguridad: si por tamaño quedó 1 parte, reintenta single-pass
    if len(partes) == 1:
        t0 = _t()
        max_out = _max_out_for_text(texto)
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{prompt_maestro}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve SOLO el informe final (texto), sin preámbulos ni títulos de estas instrucciones."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = resp.choices[0].message.content.strip()
            bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
            bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
            bruto = _rellenar_secciones_faltantes(bruto, texto, varios_anexos, hints)
            bruto = _ampliar_secciones_especificas(bruto, texto, varios_anexos)
            out = preparar_texto_para_pdf(bruto)
            _log_tiempo("analizar_single_pass_len1", t0)
            return out
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # A) Notas intermedias (CONCURRENTE)
    notas_list = _generar_notas_concurrente(partes)
    notas_integradas = "\n".join(notas_list)

    # B) Síntesis final
    t0_sint = _t()
    max_out = _max_out_for_text(texto)
    messages_final = [
        {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
        {"role": "user", "content": f"""{prompt_maestro}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

{("=== HALLAZGOS AUTOMÁTICOS (snippets literales) ===\n" + hints) if hints else ""}

👉 Integra TODO en un **solo informe**; deduplica; cita una vez por dato con todas las fuentes.
👉 Prohibido meta-comentarios de fragmentos. No imprimas títulos de estas instrucciones.
👉 Devuelve SOLO el informe final en texto.
"""}
    ]
    try:
        resp_final = _llamada_openai(messages_final, max_completion_tokens=max_out, model=_pick_model("sintesis"))
        bruto = (resp_final.choices[0].message.content or "").strip()
        bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
        bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
        bruto = _rellenar_secciones_faltantes(bruto, texto, varios_anexos, hints)
        bruto = _ampliar_secciones_especificas(bruto, texto, varios_anexos)
        out = preparar_texto_para_pdf(bruto)
        _log_tiempo("sintesis_final", t0_sint)
        return out
    except Exception as e:
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias (limpias):\n{_limpiar_meta(notas_integradas)}"

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
def _render_pdf_bytes(resumen: str) -> bytes:
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
    fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
    c.drawCentredString(A4[0] / 2, A4[1] - 42 * mm, f"{fecha_actual}")

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

def generar_pdf_con_plantilla(resumen: str, nombre_archivo: str):
    output_dir = os.path.join("generated_pdfs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, nombre_archivo)

    data = _render_pdf_bytes(resumen)

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
