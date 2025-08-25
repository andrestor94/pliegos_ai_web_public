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
from zoneinfo import ZoneInfo  # fallback local AR

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

# --- Limitar enumeraciones y desactivar expansiones automáticas (nuevo) ---
EXPAND_SECTIONS_213_216 = int(os.getenv("EXPAND_SECTIONS_213_216", "0"))
MAX_RENGLONES_OUT       = int(os.getenv("MAX_RENGLONES_OUT", "12"))
MAX_ARTICULOS_OUT       = int(os.getenv("MAX_ARTICULOS_OUT", "12"))

# Nuevo: reforzar FICHA (arregla placeholders literales y completa renglones/Sección 9)
ENFORCE_FICHA = int(os.getenv("ENFORCE_FICHA", "1"))

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

# ---- OCR selectivo en paralelo (muestreo distribuido) ----
from concurrent.futures import ThreadPoolExecutor, as_completed

def _ocr_pagina_png_bytes(png_bytes: bytes, idx: int) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    txt = _ocr_openai_imagen_b64(b64)
    return f"[PÁGINA {idx+1}]\n{txt}" if txt else f"[PÁGINA {idx+1}] (sin texto OCR)"

def _ocr_selectivo_por_pagina(doc: fitz.Document, max_pages: int) -> str:
    """
    Muestrea páginas repartidas a lo largo de todo el documento (inicio/medio/fin).
    Así no se pierde la planilla si está al final.
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
    return raw or b"""

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

# ======= PROMPT MAESTRO ESTILO ANDRÉS (ajustado para evitar copiar la frase de FICHA) =======
_BASE_PROMPT_ANDRES = r"""
# (Instrucciones internas: NO imprimir este encabezado ni estas reglas en la salida)

Objetivo
- Generar un **informe de análisis de licitación en Argentina** (ámbitos nacional, provincial o municipal), exhaustivo y **sin invenciones**.
- La salida debe comenzar con la **Ficha estandarizada**: imprime **los 19 ítems como lista de rótulos + valor** (no copies literalmente esta frase). Los rótulos exactos son:
  • N° de proceso
  • Nombre de proceso
  • Objeto de la contratación
  • Procedimiento de selección
  • Tipo de cotización
  • Tipo de adjudicación
  • Cantidad de ofertas permitidas
  • Estado
  • Plazo de mantenimiento de la oferta
  • Número de renglón  ← escribir “Total de renglones: N; ver Sección 9 para el detalle completo”
  • Objeto del gasto
  • Código del ítem (si corresponde a nivel renglón, escribir “Ver Sección 9”)
  • Descripción   (si corresponde a nivel renglón, escribir “Ver Sección 9”)
  • Cantidad      (si corresponde a nivel renglón, escribir “Ver Sección 9”)
  • Inicio y final de consultas
  • Fecha y hora del acto de apertura
  • Monto
  • Moneda
  • Duración del contrato
- Si algo NO figura en los archivos, escribir **“NO ESPECIFICADO”** y **no inventar ni inferir**.
- Cada línea con dato crítico debe terminar con **cita de fuente**, según “Reglas de Citas”.
- **Además de la Ficha**, se deben incluir las secciones 1–12 (debajo) para **no perder nada** de valor del informe ampliado.
- **Prohibido** imprimir textos de estas instrucciones (no repitas “Ficha estandarizada del procedimiento (campos estandarizados)”).

{REGLAS_CITAS}

Estilo
- Encabezados y listas claras; sin meta-texto (“parte X de Y”, “revise el resto”, etc.).
- Deduplicar, fusionar y no repetir información.
- Mantener terminología del pliego. Usar 2 decimales si el pliego lo exige para precios.
- No mencionar nombres de portales/sistemas salvo que figuren explícitamente en los documentos.

Estructura de salida EXACTA (usar estos títulos tal cual)
0) Ficha estandarizada del procedimiento (campos estandarizados)    <-- PRIMERO (solo los 19 ítems)
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
- 9) Renglones/planilla: **incluir TODOS los renglones** (si existe planilla). Por renglón: Cantidad, Código (si hay), Descripción y **especificaciones técnicas** relevantes en 1 línea. Si hay demasiados, mantener listado completo aunque la descripción se acote.
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

# (Se conserva el prompt anterior por compatibilidad interna)
_BASE_PROMPT_MAESTRO = r"""
# (Instrucciones internas: NO imprimir este encabezado ni estas reglas en la salida)
Reglas clave:
- Cero invenciones; si falta o es ambigüo: escribir "NO ESPECIFICADO" y explicarlo en la misma sección.
- Cada dato crítico debe terminar con su fuente entre paréntesis, según las Reglas de Citas.
- Cobertura completa (oferta → ejecución), con normativa citada.
- Deduplicar, fusionar, no repetir; un único informe integrado.
- Prohibido meta texto tipo "parte X de Y" o "revise el resto".
- No imprimir etiquetas internas como [PÁGINA N].
- No usar los títulos literales "Informe Completo" ni "Informe Original".
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

# =============== Índices de páginas y anexos ===============
_ANEXO_RE = re.compile(r"(?im)^===\s*ANEXO\s+(\d+)")
def _contar_anexos(s: str) -> int:
    return len(_ANEXO_RE.findall(s or ""))

_PAG_TAG_RE = re.compile(r"\[PÁGINA\s+(\d+)\]")

def _index_paginas(s: str) -> List[Tuple[int,int]]:
    return [(m.start(), int(m.group(1))) for m in _PAG_TAG_RE.finditer(s)]

def _pagina_de_indice(indices: List[Tuple[int,int]], pos: int) -> int:
    last = 1
    for i, p in indices:
        if i <= pos: last = p
        else: break
    return last

def _index_anexos(s: str) -> List[Tuple[int,int]]:
    return [(m.start(), int(m.group(1))) for m in _ANEXO_RE.finditer(s)]

def _anexo_en_pos(indices: List[Tuple[int,int]], pos: int) -> Optional[int]:
    last = None
    for i, a in indices:
        if i <= pos:
            last = a
        else:
            break
    return last

# =============== Normalización de citas cuando es 1 solo documento ===============
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

# Campos detectables (ampliados y adaptados a AR)
DETECTABLE_FIELDS: Dict[str, Dict] = {
    "mant_oferta": {"label":"Mantenimiento de oferta", "pats":[r"mantenim[ií]ento de la oferta", r"validez de la oferta"]},
    "gar_mant":    {"label":"Garantía de mantenimiento", "pats":[r"garant[ií]a.*manten", r"\b5 ?%"]},
    "gar_cumpl":   {"label":"Garantía de cumplimiento", "pats":[r"garant[ií]a.*cumpl", r"\b10 ?%"]},
    "plazo_ent":   {"label":"Plazo de entrega", "pats":[r"plazo de entrega", r"\b\d{1,3}\s*d[ií]as"]},
    "tipo_cambio": {"label":"Tipo de cambio", "pats":[r"Banco\s+Naci[oó]n", r"tipo de cambio", r"BNA"]},
    "comision":    {"label":"Comisión de (Pre)?Adjudicación", "pats":[r"Comisi[oó]n.*(pre)?adjudicaci[oó]n"]},
    "muestras":    {"label":"Muestras", "pats":[r"\bmuestras?\b"]},
    "planilla":    {"label":"Planilla de cotización y renglones", "pats":[r"planilla.*cotizaci[oó]n", r"renglones?"]},
    "modalidad":   {"label":"Procedimiento/Modalidad", "pats":[r"licitaci[oó]n\s+(p[úu]blica|privada)", r"contrataci[oó]n\s+directa", r"compra\s+menor", r"subasta", r"modalidad"]},
    "plazo_contr": {"label":"Duración del contrato", "pats":[r"duraci[oó]n del contrato", r"plazo contractual", r"por el t[eé]rmino\s+de\s+\d+", r"\b\d{1,4}\s*d[ií]as"]},
    "prorroga":    {"label":"Prórroga/Ampliación", "pats":[r"pr[oó]rroga", r"ampliaci[oó]n", r"hasta\s+el\s+100%"]},
    "presupuesto": {"label":"Monto / Presupuesto", "pats":[r"presupuesto (estimado|oficial|referencial)", r"monto\s+estimado", r"cr[eé]dito\s+disponible", r"\$\s?\d{1,3}(\.\d{3})*(,\d{2})?"]},
    "expediente":  {"label":"Expediente / N° proceso", "pats":[r"\bEX-\d{4}-[A-Z0-9-]+", r"\bN[°º]\s*de\s*(proceso|procedimiento|expediente)"]},
    "fechas":      {"label":"Fechas y horas", "pats":[r"\b\d{2}/\d{2}/\d{4}\b", r"\b\d{1,2}:\d{2}\s*(hs|h)"]},
    "contacto":    {"label":"Contactos y portales", "pats":[r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", r"https?://[^\s)]+|www\.[^\s)]+"]},
    "costo_pliego":{"label":"Costo/valor del pliego", "pats":[r"(costo|valor)\s+del\s+pliego", r"adquisici[oó]n\s+del\s+pliego", r"\$\s?\d{1,3}(\.\d{3})*(,\d{2})?"]},
    "subsanacion": {"label":"Subsanación", "pats":[r"subsanaci[oó]n"]},
    "perf_modif":  {"label":"Perfeccionamiento/Modificaciones", "pats":[r"perfeccionamiento", r"modificaci[oó]n"]},
    "preferencias":{"label":"Preferencias", "pats":[r"preferencias"]},
    "criterios":   {"label":"Criterios de evaluación", "pats":[r"criterios?\s+de\s+evaluaci[oó]n"]},
    "renglones":   {"label":"Renglones y especificaciones", "pats":[r"Rengl[oó]n\s*\d+", r"Especificaciones?\s+t[ée]cnicas?"]},
    "articulos":   {"label":"Artículos citados", "pats":[r"\bArt(?:[íi]culo|\.)\s*\d+[A-Za-z]?\b"]},
    "estado":      {"label":"Estado del trámite", "pats":[r"\bestado\b", r"\bvigente\b", r"\b(adjudicado|desierto|fracasado|cerrado)\b"]},
    "consultas":   {"label":"Inicio y final de consultas", "pats":[r"\bconsultas\b", r"aclaraciones", r"preguntas"]},
    "apertura":    {"label":"Acto de apertura", "pats":[r"acto\s+de\s+apertura", r"\bapertura\b"]},
    "tipo_cotiz":  {"label":"Tipo de cotización", "pats":[r"forma\s+de\s+cotizaci[oó]n", r"tipo\s+de\s+cotizaci[oó]n", r"cotizaci[oó]n\s+por"]},
    "tipo_adj":    {"label":"Tipo de adjudicación", "pats":[r"adjudicaci[oó]n\s+por\s+(rengl[oó]n|lote|total)"]},
    "moneda":      {"label":"Moneda", "pats":[r"\bmoneda\b", r"\bARS\b", r"\bUSD\b"]},
    "obj_gasto":   {"label":"Objeto del gasto", "pats":[r"objeto\s+del\s+gasto", r"partida\s+presupuestaria", r"clasificador"]},
    "ofertas_perm":{"label":"Ofertas permitidas", "pats":[r"m[aá]s\s+de\s+una\s+oferta", r"ofertas?\s+alternativas", r"una\s+sola\s+oferta"]},
}

# ====== NUEVO: utilidades para conteo y evidencia exhaustiva ======
def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))

_ART_HEAD_RE = re.compile(r"(?im)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.*)$")
_ART_BLOCK_RE = re.compile(
    r"(?ims)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.+?)(?=^\s*art(?:[íi]culo|\.?)\s*\d+[a-zº°]?|\Z)"
)

def _extraer_articulos_con_snippets(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (rótulo_articulo, snippet_200c, pagina_aprox, anexo_num)
    """
    idx = _index_paginas(texto)
    idx_ax = _index_anexos(texto)
    res = []
    for m in _ART_BLOCK_RE.finditer(texto):
        start = m.start()
        p = _pagina_de_indice(idx, start)
        ax = _anexo_en_pos(idx_ax, start)
        rotulo = m.group(1).strip()
        contenido = (m.group(2) or "").strip()
        snippet = contenido[:200].replace("\n", " ").strip()
        res.append((rotulo, snippet, p, ax))
    if not res:
        for m in _ART_HEAD_RE.finditer(texto):
            start = m.start()
            p = _pagina_de_indice(idx, start)
            ax = _anexo_en_pos(idx_ax, start)
            rotulo = m.group(1).strip()
            snippet = (m.group(2) or "").strip()[:200].replace("\n", " ")
            res.append((rotulo, snippet, p, ax))
    return res

# --- Renglones robustos (exigir literalmente "Renglón") ---
_ROW_START_RE = re.compile(r"(?im)^(?:reng(?:l[oó]n)?\.?\s*)(\d{1,4})\b")
_CODE_RE = re.compile(r"\b[A-Z]{1,3}\d{5,8}\b")  # p.ej. D0330113, GB079001, E5001253
_QTY_RE = re.compile(r"\b\d{1,6}\b")

def _extraer_renglones_y_especificaciones(texto: str) -> List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]]:
    """
    Devuelve lista de (num_renglon, cantidad, codigo, descripcion_full, pagina_aprox, anexo_num)
    - Reconoce filas numeradas QUE EMPIEZAN CON "Renglón".
    - Agrega líneas subsiguientes hasta el próximo comienzo de fila.
    """
    idx = _index_paginas(texto)
    idx_ax = _index_anexos(texto)
    res: List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]] = []

    lines = texto.splitlines()
    pos = 0
    starts: List[Tuple[int,int]] = []  # (line_index, abs_pos)
    for i, ln in enumerate(lines):
        m = _ROW_START_RE.match(ln)
        if m:
            starts.append((i, pos))
        pos += len(ln) + 1

    if not starts:
        return res

    # sentinel
    starts.append((len(lines), len(texto)))

    for k in range(len(starts) - 1):
        i_line, abs_pos = starts[k]
        j_line, _abs_pos_next = starts[k+1]
        block_lines = lines[i_line:j_line]
        block_text = " ".join([re.sub(r"\s+", " ", x).strip() for x in block_lines if x.strip()])

        # número de renglón
        mnum = _ROW_START_RE.match(lines[i_line])
        try:
            num_r = int(mnum.group(1)) if mnum else None
        except Exception:
            num_r = None

        # cantidad (primer entero de la línea tras el número)
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

        # descripción y especificaciones
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
    Arma bloques de evidencia literal (con páginas y anexos) para renglones/planilla y artículos.
    Devuelve (bloque_evidencia, cant_renglones, cant_articulos).
    """
    renglones = _extraer_renglones_y_especificaciones(texto)
    articulos = _extraer_articulos_con_snippets(texto)

    ev_parts = []
    if renglones:
        ev = []
        for (num, qty, code, desc, p, ax) in renglones:
            cit = f"(Anexo {ax}, p. {p})" if ax else f"(p. {p})"
            det = []
            if qty is not None: det.append(f"cant {qty}")
            if code: det.append(f"cód {code}")
            det_txt = " — ".join(det) if det else ""
            linea = f"- Renglón {num}{(' — ' + det_txt) if det_txt else ''}: {desc} {cit}"
            ev.append(linea)
        ev_parts.append("### EVIDENCIA Renglones / Planilla (literal)\n" + "\n".join(ev))
    if articulos:
        ev = []
        for (rot, sn, p, ax) in articulos:
            cit = f"(Anexo {ax}, p. {p})" if ax else f"(p. {p})"
            ev.append(f"- {rot} — {sn} {cit}")
        ev_parts.append("### EVIDENCIA Artículos (literal)\n" + "\n".join(ev))

    return ("\n\n".join(ev_parts) if ev_parts else ""), len(renglones), len(articulos)

def _conteo_en_informe(informe: str) -> Tuple[int, int]:
    return _count(r"(?im)\brengl[oó]n\s*\d+", informe), _count(r"(?im)\bart(?:[íi]culo|\.?)\s*\d+", informe)

def _max_out_for_text(texto: str) -> int:
    base_chars = len(texto or "")
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

# ====== Utilidades de compresión no literal ======
def _truncate_words(s: str, max_words: int) -> str:
    try:
        words = re.findall(r"\S+", s or "")
        if len(words) <= max_words:
            return (s or "").strip()
        return " ".join(words[:max_words]).rstrip(",.;:") + "…"
    except Exception:
        return (s or "").strip()

# Palabras-clave para filtrar artículos útiles (nuevo)
_ART_KEYS = re.compile(
    r"(objeto|tipolog|modalidad|mantenim|pr[oó]rroga|oferta|apertura|evaluaci[oó]n|empate|mejora|adjudicaci[oó]n|"
    r"garant[ií]a|entrega|plazo|pago|factura|sancion|penalidad|rescis[ií]n|perfeccionamiento|subsanaci[oó]n)",
    re.I
)

# ====== Generadores determinísticos (capados) de 2.13 y 2.16 ======
def _build_section_213(texto: str, varios_anexos: bool) -> str:
    rows = _extraer_renglones_y_especificaciones(texto)
    if not rows:
        return ""
    rows = rows[:max(1, MAX_RENGLONES_OUT)]  # tope
    lines = ["2.13 Planilla de cotización y renglones:"]
    for (num, qty, code, desc, p, ax) in rows:
        desc_corta = _truncate_words(desc, RENGLON_DESC_MAX_WORDS)
        partes = [f"Renglón {num}"]
        if qty is not None: partes.append(f"Cantidad: {qty}")
        if code: partes.append(f"Código: {code}")
        partes.append(f"Descripción/Especificaciones: {desc_corta}")
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        lines.append(" - " + " — ".join(partes) + f" {cita}")
    return "\n".join(lines)

def _build_section_216(texto: str, varios_anexos: bool) -> str:
    arts = _extraer_articulos_con_snippets(texto)
    if not arts:
        return ""
    # filtrar por relevancia práctica
    arts = [(rot, sn, p, ax) for (rot, sn, p, ax) in arts if _ART_KEYS.search(sn or "") or _ART_KEYS.search(rot or "")]
    if not arts:
        return ""
    arts = arts[:max(1, MAX_ARTICULOS_OUT)]  # tope
    lines = ["2.16 Catálogo de artículos citados:"]
    for (rot, sn, p, ax) in arts:
        rot_norm = re.sub(r"(?i)art(?:[íi]culo|\.)\s*", "Art. ", rot).strip()
        sn = _truncate_words(sn, ART_SNIPPET_MAX_WORDS)
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        lines.append(f" - {rot_norm} — {sn} {cita}")
    return "\n".join(lines)

# ====== Contactos (emails/URLs) con página/anexo ======
CONTACT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CONTACT_URL_RE   = re.compile(r"(https?://[^\s)]+|www\.[^\s)]+)")

def _extraer_contactos_con_paginas(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    """
    Devuelve lista de (tipo, valor, p, anexo) con tipo in {"email","url"}
    """
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)
    res: List[Tuple[str,str,int,Optional[int]]] = []
    for m in CONTACT_EMAIL_RE.finditer(texto):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos)
        ax = _anexo_en_pos(idx_ax, pos)
        res.append(("email", m.group(0), p, ax))
    for m in CONTACT_URL_RE.finditer(texto):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos)
        ax = _anexo_en_pos(idx_ax, pos)
        v = m.group(0).rstrip(").,;")
        res.append(("url", v, p, ax))
    # dedupe preservando orden
    seen = set()
    dedup = []
    for t,v,p,ax in res:
        key = (t, v.lower())
        if key in seen: continue
        seen.add(key); dedup.append((t,v,p,ax))
    return dedup

def _build_section_23(texto: str, varios_anexos: bool) -> str:
    items = _extraer_contactos_con_paginas(texto)
    if not items:
        return ""
    out = ["2.3 Contactos y portales:"]
    for (t,v,p,ax) in items:
        etiqueta = "Email" if t=="email" else "URL"
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {etiqueta}: {v} {cita}")
    return "\n".join(out)

# ====== Normativa aplicable (Ley/Decreto/Resolución/Disposición) ======
NORM_TIPOS = [
    (r"Ley", r"\bLey(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    (r"Decreto", r"\bDecreto(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    (r"Resolución", r"\bResoluci[oó]n(?:\s*(?:Ministerial|Conjunta))?\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
    (r"Disposición", r"\bDisposici[oó]n\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
]
NORM_PATTS = [(tipo, re.compile(patt, re.I)) for (tipo, patt) in NORM_TIPOS]

def _extraer_normativa(texto: str) -> List[Tuple[str,str,int,Optional[int]]]:
    """
    Devuelve lista de (tipo, numero, p, anexo)
    """
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)
    res = []
    for (tipo, cre) in NORM_PATTS:
        for m in cre.finditer(texto):
            pos = m.start()
            p = _pagina_de_indice(idx_pag, pos)
            ax = _anexo_en_pos(idx_ax, pos)
            numero = m.group(1).strip()
            res.append((tipo, numero, p, ax))
    # dedupe preservando orden
    seen = set(); dedup = []
    for t,n,p,ax in res:
        key = (t.lower(), n)
        if key in seen: continue
        seen.add(key); dedup.append((t,n,p,ax))
    return dedup

def _build_section_215(texto: str, varios_anexos: bool) -> str:
    normas = _extraer_normativa(texto)
    if not normas:
        return ""
    out = ["2.15 Normativa aplicable:"]
    for (t,n,p,ax) in normas:
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {t} {n} {cita}")
    return "\n".join(out)

# ====== Reemplazo de secciones en el informe ======
def _find_section_bounds(text: str, header_regex: str) -> Tuple[int,int]:
    """Devuelve (start, end) del bloque que inicia en header_regex hasta el próximo '2.' encabezado o fin."""
    m = re.search(header_regex, text, flags=re.I)
    if not m:
        return (-1, -1)
    start = m.start()
    nxt = re.search(r"(?im)^\s*2\.(1[0-9]|[1-9])\s", text[m.end():])
    if not nxt:
        return (start, len(text))
    return (start, m.end() + nxt.start())

def _replace_section(text: str, header_regex: str, replacement: str) -> str:
    s, e = _find_section_bounds(text, header_regex)
    if s == -1:
        # si no existe, lo anexamos al final con un salto
        return text.rstrip() + "\n\n" + replacement.strip() + "\n"
    return text[:s] + replacement.strip() + "\n" + text[e:]

# ==================== Ampliación / sustitución de 2.13 y 2.16 (capadas) ====================
def _ampliar_secciones_especificas(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Por defecto (EXPAND_SECTIONS_213_216=0) NO toca 2.13 ni 2.16.
    Mantiene la salida concisa (estilo Andrés) y sólo normaliza Contactos (2.3) y Normativa (2.15).
    """
    out = informe

    # Siempre: actualizar determinísticamente Contactos (2.3) y Normativa (2.15)
    sec23 = _build_section_23(texto_fuente, varios_anexos)
    if sec23:
        out = _replace_section(out, r"(?im)^\s*2\.3\s+Contactos", sec23)

    sec215 = _build_section_215(texto_fuente, varios_anexos)
    if sec215:
        out = _replace_section(out, r"(?im)^\s*2\.15\s+Normativa", sec215)

    # Si no se solicita expansión de renglones/artículos, retornar
    if not EXPAND_SECTIONS_213_216:
        return out

    # Construir 2.13 y 2.16 con topes/cap y reemplazar sin duplicar variantes
    sec213 = _build_section_213(texto_fuente, varios_anexos)
    if sec213:
        alt213 = sec213.replace("2.13 Planilla de cotización y renglones:", "9) Renglones y planilla de cotización:")
        out = _replace_section(out, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla", alt213)
        out = _replace_section(out, r"(?im)^\s*2\.13\s+Planilla", sec213)

    sec216 = _build_section_216(texto_fuente, varios_anexos)
    if sec216:
        out = _replace_section(out, r"(?im)^\s*2\.16\s+Cat[aá]logo\s+de\s+art", sec216)
        # eliminar posibles títulos alternativos "ANEXO — Catálogo ..."
        out = re.sub(r"(?im)^\s*(ANEXO|Anexo)\s*[-–—]?\s*Cat[aá]logo\s+de\s+art[^\n]*\n?", "", out)

    out = re.sub(r"(?im)^\s*informe\s+original\s*$", "", out)
    return out

# ==================== AUTOCORRECCIÓN DE LA FICHA (puntos 1–3) ====================
_FICHA_HEADER_RE = re.compile(r"(?im)^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento.*$", re.M)
_SIGUIENTE_HEADER_1_RE = re.compile(r"(?im)^\s*1\)\s*Resumen\s+ejecutivo", re.M)

def _extraer_bloque_ficha(informe: str) -> Tuple[int, int]:
    m0 = _FICHA_HEADER_RE.search(informe or "")
    if not m0:
        return (-1, -1)
    start = m0.end()
    m1 = _SIGUIENTE_HEADER_1_RE.search(informe, start)
    end = m1.start() if m1 else len(informe)
    return start, end

def _cita_de_renglones(rows: List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]], varios_anexos: bool) -> str:
    if not rows:
        return "(Fuente: documento provisto)"
    p = rows[0][4] or 1
    ax = rows[0][5]
    if varios_anexos and ax:
        return f"(Anexo {ax}, p. {p})"
    return f"(p. {p})"

def _buscar_objeto_gasto_snippet(texto: str, varios_anexos: bool) -> Tuple[Optional[str], Optional[str]]:
    idx = _index_paginas(texto); idx_ax = _index_anexos(texto)
    for pat in DETECTABLE_FIELDS["obj_gasto"]["pats"]:
        m = re.search(pat, texto, flags=re.I)
        if m:
            pos = m.start()
            p = _pagina_de_indice(idx, pos)
            ax = _anexo_en_pos(idx_ax, pos)
            start = max(0, pos - 60); end = min(len(texto), pos + 160)
            snippet = texto[start:end].replace("\n", " ")
            snippet = re.sub(r"\s+", " ", snippet).strip()
            # recortar tras ':'
            after = snippet.split(":", 1)[-1].strip() if ":" in snippet else snippet
            after = after.strip(" -—").strip()
            after = _truncate_words(after, 18)
            cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else f"(p. {p})"
            return after if after else None, cita
    return None, None

def _autocorregir_ficha(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    - Evita que aparezca como ítem literal 'Ficha estandarizada del procedimiento (campos estandarizados)'.
    - Completa 'Número de renglón' con 'Total de renglones: N; ver Sección 9...' y agrega cita.
    - Si hay renglones, fuerza en Ficha: Código/Descripción/Cantidad -> 'Ver Sección 9' con cita (evita 'NO ESPECIFICADO').
    - Intenta completar 'Objeto del gasto' con snippet si lo detecta en los anexos.
    """
    if not informe:
        return informe

    s, e = _extraer_bloque_ficha(informe)
    if s == -1:
        return informe

    bloque = informe[s:e]
    lineas = [ln for ln in bloque.splitlines()]

    # 1) Quitar posibles repeticiones literales dentro de la Ficha
    lineas = [ln for ln in lineas if not re.search(r"(?i)ficha\s+estandarizada.*campos\s+estandarizados", ln.strip())]

    # 2) Datos de renglones
    rows = _extraer_renglones_y_especificaciones(texto_fuente)
    total_r = len(rows)
    cita_rows = _cita_de_renglones(rows, varios_anexos)

    def _patch(label_regex: str, nuevo_valor: str) -> None:
        nonlocal lineas
        rx = re.compile(label_regex, flags=re.I)
        for i, ln in enumerate(lineas):
            if rx.search(ln):
                # normalizar a "Etiqueta: valor ..."
                if ":" in ln:
                    pref = ln.split(":", 1)[0]
                    lineas[i] = f"{pref}: {nuevo_valor}"
                else:
                    lineas[i] = f"{ln.strip()}: {nuevo_valor}"
                return
        # si no estaba, lo insertamos al inicio del bloque de la ficha
        lineas.insert(0, re.sub(r".*?:", "", label_regex, flags=re.I))

    # 3) Número de renglón -> total + cita
    if total_r > 0:
        _patch(r"^\s*[-•]?\s*N[úu]mero\s+de\s+rengl[oó]n\s*:?", f"Total de renglones: {total_r}; ver Sección 9 para el detalle completo {cita_rows}")

    # 4) Código/Descripción/Cantidad -> 'Ver Sección 9' si hay renglones
    if total_r > 0:
        marcador = f"Ver Sección 9 {cita_rows}"
        _patch(r"^\s*[-•]?\s*C[oó]digo\s+del\s+ítem\s*:?", marcador)
        _patch(r"^\s*[-•]?\s*Descripci[oó]n\s*:?", marcador)
        _patch(r"^\s*[-•]?\s*Cantidad\s*:?", marcador)

    # 5) Objeto del gasto si lo tenemos
    og_val, og_cita = _buscar_objeto_gasto_snippet(texto_fuente, varios_anexos)
    if og_val:
        _patch(r"^\s*[-•]?\s*Objeto\s+del\s+gasto\s*:?", f"{og_val} {og_cita}")

    # Reconstruir informe
    nuevo_bloque = "\n".join(lineas)
    return informe[:s] + nuevo_bloque + informe[e:]
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
    # Usar el nuevo prompt estilo Andrés (con Ficha estandarizada + 1–12)
    prompt_maestro = _prompt_andres(varios_anexos)

    # Hints regex (opcionales, capados por tamaño)
    hints = _build_regex_hints(texto) if ENABLE_REGEX_HINTS else ""
    hints_block = f"\n\n=== HALLAZGOS AUTOMÁTICOS (snippets literales para verificación) ===\n{hints}\n" if hints else ""

    # ¿forzar dos etapas en multi-anexo grande?
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
            bruto = resp.choices[0].message.content.strip()
            bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
            # Autocorrección de Ficha antes del 2º pase
            bruto = _autocorregir_ficha(bruto, texto, varios_anexos)
            # 2º pase para completar NO ESPECIFICADO
            bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
            # Autocorrección de Ficha nuevamente (por si el 2º pase alteró algo)
            bruto = _autocorregir_ficha(bruto, texto, varios_anexos)
            # Normalizar 2.3/2.15 y (si está habilitado) 2.13/2.16
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
            {"role": "system", "content": "Actuá como equipo experto en derecho administrativo argentino (ámbitos nacional, provincial y municipal) y compras públicas; redactor técnico-jurídico. Cero invenciones."},
            {"role": "user", "content": f"{prompt_maestro}{hints_block}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve SOLO el informe final (texto), sin preámbulos ni títulos de estas instrucciones."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out, model=_pick_model("analisis"))
            bruto = resp.choices[0].message.content.strip()
            bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
            bruto = _autocorregir_ficha(bruto, texto, varios_anexos)
            bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
            bruto = _autocorregir_ficha(bruto, texto, varios_anexos)
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
        bruto = _normalize_citas_salida(_limpiar_meta(bruto), varios_anexos)
        bruto = _autocorregir_ficha(bruto, texto, varios_anexos)
        bruto = _segundo_pase_si_falta(bruto, texto, varios_anexos)
        bruto = _autocorregir_ficha(bruto, texto, varios_anexos)
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
    - ≥2: marca ANEXOS para trazabilidad. Si el total entra en MAX_SINGLE_PASS_CHARS_MULTI,
          puede ir single-pass salvo que force_two_stage.
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
