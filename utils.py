import io
import os
import re
import base64
import mimetypes
import time
from datetime import datetime
from typing import List

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
MODEL_ANALISIS = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-4o-mini")
VISION_MODEL   = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")

MAX_SINGLE_PASS_CHARS = int(os.getenv("MAX_SINGLE_PASS_CHARS", "120000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "24000"))
MAX_COMPLETION_TOKENS_SALIDA = int(os.getenv("MAX_COMPLETION_TOKENS_SALIDA", "3500"))
TEMPERATURE_ANALISIS = os.getenv("TEMPERATURE_ANALISIS", "").strip()
ANALISIS_MODO = os.getenv("ANALISIS_MODO", "").lower().strip()  # "fast" opcional

# OCR
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "8"))
VISION_DPI = int(os.getenv("VISION_DPI", "150"))
OCR_TEXT_MIN_CHARS = int(os.getenv("OCR_TEXT_MIN_CHARS", "120"))
OCR_CONCURRENCY = int(os.getenv("OCR_CONCURRENCY", "4"))

# Control de paginado en texto nativo
PAGINAR_TEXTO_NATIVO = int(os.getenv("PAGINAR_TEXTO_NATIVO", "1"))

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

_BASE_PROMPT_MAESTRO = r"""
# (Instrucciones internas: NO imprimir este encabezado ni estas reglas en la salida)
Reglas clave:
- No mencionar "C.R.A.F.T." ni títulos de estas instrucciones.
- Cada dato crítico debe terminar con su fuente entre paréntesis, según las Reglas de Citas.
- Cero invenciones; si falta o es ambiguo: escribir "NO ESPECIFICADO" y mover la duda a "Consultas sugeridas".
- Cobertura completa (oferta → ejecución), con normativa citada.
- Deduplicar, fusionar, no repetir; un único informe integrado.
- Prohibido meta texto tipo "parte X de Y" o "revise el resto".
- No imprimir etiquetas internas como [PÁGINA N].

Formato de salida:
1) RESUMEN EJECUTIVO (≤200 palabras)
2) INFORME EXTENSO CON TRAZABILIDAD
   2.1 Identificación del llamado
   2.2 Calendario y lugares
   2.3 Contactos y portales
   2.4 Alcance y plazo contractual
   2.5 Tipología / modalidad (citar norma/artículos)
   2.6 Mantenimiento de oferta y prórroga
   2.7 Garantías (umbral UC, %, plazos, formas)
   2.8 Presentación de ofertas (soporte, firmas, docs obligatorias, etc.)
   2.9 Apertura, evaluación y adjudicación (tipo de cambio BNA, comisión, criterios, preferencias)
   2.10 Subsanación (qué sí/no)
   2.11 Perfeccionamiento y modificaciones
   2.12 Entrega, lugares y plazos
   2.13 Planilla de cotización y renglones
   2.14 Muestras
   2.15 Cláusulas adicionales
   2.16 Matriz de Cumplimiento (tabla: requisito | encontrado (sí/no) | fuente | notas)
   2.17 Mapa de Anexos (tabla)
   2.18 Semáforo de Riesgos (con justificación y fuente)
   2.19 Checklist operativo
   2.20 Ambigüedades/Inconsistencias y Consultas Sugeridas
   2.21 Anexos del Informe (índice de trazabilidad)

Estilo:
- Títulos en mayúsculas iniciales, listas claras, tablas simples. Sin "#".
- Aplicar la Guía de sinónimos.
"""

def _prompt_maestro(varios_anexos: bool) -> str:
    """
    Reglas de citas según contexto + mapeo explícito de [PÁGINA N] -> (p. N)
    """
    if varios_anexos:
        regla_citas = (
            "Reglas de Citas:\n"
            "- Al final de cada línea con dato, usar (Anexo X, p. N).\n"
            "- Para deducir p. N, utiliza la etiqueta [PÁGINA N] más cercana al dato dentro del texto provisto.\n"
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
    return f"{_BASE_PROMPT_MAESTRO}\n{regla_citas}\nGuía de sinónimos:\n{SINONIMOS_CANONICOS}"

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
- [FALTA] campo X — NO ESPECIFICADO. (Fuente: documento provisto)
"""

_META_PATTERNS = [
    re.compile(r"(?i)\bparte\s+\d+\s+de\s+\d+"),
    re.compile(r"(?i)informe\s+basado\s+en\s+la\s+parte"),
    re.compile(r"(?i)revise\s+las\s+partes\s+restantes"),
    re.compile(r"(?i)información\s+puede\s+estar\s+incompleta")
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
    """
    Si es documento único:
      - (Anexo I, p. N) -> (p. N)
      - (Anexo I) -> (Fuente: documento provisto)
    """
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
        if _CODE_FENCE_RE.match(ln): continue
        m = _HDR_RE.match(ln)
        if m:
            titulo = _title_case(m.group(2).strip(": ").strip())
            out_lines.append(titulo); continue
        if _TABLE_SEP_RE.match(ln): continue
        if _BULLET_RE.match(ln): ln = _BULLET_RE.sub("• ", ln)
        ln = _LINK_RE.sub(lambda mm: f"{mm.group(1)} ({mm.group(2)})", ln)
        ln = _BOLD_ITALIC_RE.sub(lambda mm: mm.group(2), ln)
        out_lines.append(ln)
    texto = "\n".join(out_lines)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto

# ==================== Llamada a OpenAI robusta ====================
def _max_tokens_salida_adaptivo(longitud_chars: int) -> int:
    """
    En modo fast, reduce salida para bajar latencia:
      - < 15k chars: 2200
      - < 40k chars: 2800
      - resto: MAX_COMPLETION_TOKENS_SALIDA
    """
    base = MAX_COMPLETION_TOKENS_SALIDA
    if ANALISIS_MODO != "fast":
        return base
    if longitud_chars < 15000:
        return min(base, 2200)
    if longitud_chars < 40000:
        return min(base, 2800)
    return base

def _llamada_openai(messages, model=MODEL_ANALISIS, temperature_str=TEMPERATURE_ANALISIS,
                    max_completion_tokens=None, retries=2, fallback_model="gpt-4o-mini"):
    def _build_kwargs(mdl):
        kw = dict(model=mdl, messages=messages, max_completion_tokens=max_completion_tokens or MAX_COMPLETION_TOKENS_SALIDA)
        if ANALISIS_MODO == "fast":
            kw["temperature"] = 0
        elif temperature_str != "":
            try:
                kw["temperature"] = float(temperature_str)
            except:
                pass
        return kw

    models_to_try = [model]
    if fallback_model and fallback_model != model:
        models_to_try.append(fallback_model)

    last_error = None
    for mdl in models_to_try:
        for attempt in range(retries + 1):
            try:
                resp = client.chat.completions.create(**_build_kwargs(mdl))
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

# ==================== Analizador principal ====================
def analizar_con_openai(texto: str) -> str:
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    n_anexos = _contar_anexos(texto)
    varios_anexos = n_anexos >= 2
    prompt_maestro = _prompt_maestro(varios_anexos)

    # Pasada única (rápida)
    if len(texto) <= MAX_SINGLE_PASS_CHARS and not varios_anexos:
        t0 = _t()
        max_out = _max_tokens_salida_adaptivo(len(texto))
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{prompt_maestro}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve SOLO el informe final (texto), sin preámbulos ni títulos de estas instrucciones."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out)
            bruto = resp.choices[0].message.content.strip()
            limpio = _limpiar_meta(bruto)
            normalizado = _normalize_citas_salida(limpio, varios_anexos)
            out = preparar_texto_para_pdf(normalizado)
            _log_tiempo("analizar_single_pass", t0)
            return out
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # Si quedó en 1 parte, también usamos single-pass
    partes = _particionar(texto, CHUNK_SIZE)
    if len(partes) == 1:
        t0 = _t()
        max_out = _max_tokens_salida_adaptivo(len(texto))
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{prompt_maestro}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve SOLO el informe final (texto), sin preámbulos ni títulos de estas instrucciones."}
        ]
        try:
            resp = _llamada_openai(messages, max_completion_tokens=max_out)
            bruto = resp.choices[0].message.content.strip()
            limpio = _limpiar_meta(bruto)
            normalizado = _normalize_citas_salida(limpio, varios_anexos)
            out = preparar_texto_para_pdf(normalizado)
            _log_tiempo("analizar_single_pass_len1", t0)
            return out
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # Dos etapas (solo si realmente hay que particionar)
    notas = []
    t0_notas = _t()
    for i, parte in enumerate(partes, 1):
        msg = [
            {"role": "system", "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user", "content": f"{CRAFT_PROMPT_NOTAS}\n\n## Guía de sinónimos/normalización\n{SINONIMOS_CANONICOS}\n\n=== FRAGMENTO {i}/{len(partes)} ===\n{parte}"}
        ]
        try:
            r = _llamada_openai(msg, max_completion_tokens=1800)
            notas.append(r.choices[0].message.content.strip())
        except Exception as e:
            notas.append(f"[ERROR] No se pudieron generar notas de la parte {i}: {e}")
    _log_tiempo(f"notas_intermedias_{len(partes)}_partes", t0_notas)

    notas_integradas = "\n".join(notas)

    t0_sint = _t()
    max_out = _max_tokens_salida_adaptivo(len(texto))
    messages_final = [
        {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
        {"role": "user", "content": f"""{prompt_maestro}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

👉 Integra TODO en un **solo informe**; deduplica; cita una vez por dato con todas las fuentes.
👉 Prohibido meta-comentarios de fragmentos. No imprimas títulos de estas instrucciones.
👉 Devuelve SOLO el informe final en texto.
"""}
    ]
    try:
        resp_final = _llamada_openai(messages_final, max_completion_tokens=max_out)
        bruto = resp_final.choices[0].message.content.strip()
        limpio = _limpiar_meta(bruto)
        normalizado = _normalize_citas_salida(limpio, varios_anexos)
        out = preparar_texto_para_pdf(normalizado)
        _log_tiempo("sintesis_final", t0_sint)
        return out
    except Exception as e:
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias (limpias):\n{_limpiar_meta(notas_integradas)}"

# ==================== Multi-anexo ====================
def analizar_anexos(files: list) -> str:
    """
    Combina anexos y ejecuta análisis.
    - Si hay 1 solo archivo: NO agrega marcadores "=== ANEXO ... ===" (permite single-pass y citas por p. N).
    - Si hay ≥2: agrega marcadores para trazabilidad.
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
            model=os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini"),
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
def generar_pdf_con_plantilla(resumen: str, nombre_archivo: str):
    output_dir = os.path.join("generated_pdfs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, nombre_archivo)

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

    resumen = preparar_texto_para_pdf((resumen or "").replace("**", ""))

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
        if parrafo.strip().endswith(":") or parrafo.strip().istitle():
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
        y -= 6

    c.save()
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())
    return output_path

def dividir_texto(texto, canvas_obj, max_width):
    palabras = texto.split(" ")
    lineas, linea_actual = [], ""
    for palabra in palabras:
        prueba = (linea_actual + " " + palabra) if linea_actual else palabra
        if canvas_obj.stringWidth(prueba, canvas_obj._fontname, canvas_obj._fontsize) <= max_width:
            linea_actual = prueba
        else:
            lineas.append(linea_actual)
            linea_actual = palabra
    if linea_actual:
        lineas.append(linea_actual)
    return lineas
