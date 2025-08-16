import io
import os
import re
import base64
import mimetypes
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
# NO rompe si no está instalado; simplemente se salta extracción DOCX.
try:
    import docx  # python-docx
except Exception:
    docx = None

load_dotenv()

# ========================= OpenAI client =========================
# Timeout defensivo para evitar llamadas que queden colgadas
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "90"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT)

# Modelos configurables por .env (con defaults seguros)
MODEL_ANALISIS = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-4o-mini")
VISION_MODEL   = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")

# Heurísticas
MAX_SINGLE_PASS_CHARS = int(os.getenv("MAX_SINGLE_PASS_CHARS", "55000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "14000"))  # tamaño razonable en chars
MAX_COMPLETION_TOKENS_SALIDA = int(os.getenv("MAX_COMPLETION_TOKENS_SALIDA", "3500"))
TEMPERATURE_ANALISIS = os.getenv("TEMPERATURE_ANALISIS", "").strip()
# si está vacío, no la mandamos (evita error en modelos que no soportan temperature)

# OCR
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "8"))
VISION_DPI = int(os.getenv("VISION_DPI", "170"))

# ==================== Utilidades de OCR / Raster ====================
def _rasterizar_pagina(page, dpi=VISION_DPI) -> bytes:
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def _ocr_openai_imagen_b64(b64_png: str) -> str:
    """
    OCR literal de una imagen (base64).
    Conserva títulos, tablas como líneas, listas y números. No resume.
    """
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

def _ocr_con_vision(doc: fitz.Document, max_pages: int = VISION_MAX_PAGES) -> str:
    """
    OCR por OpenAI Vision (primeras N páginas del PDF). Devuelve texto literal por página.
    """
    textos = []
    n = len(doc)
    to_process = min(n, max_pages)
    for i in range(to_process):
        page = doc.load_page(i)
        png_bytes = _rasterizar_pagina(page)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        content = _ocr_openai_imagen_b64(b64)
        if content:
            textos.append(f"[PÁGINA {i+1}]\n{content}")
        else:
            textos.append(f"[PÁGINA {i+1}] (sin texto OCR)")
    if n > to_process:
        textos.append(f"\n[AVISO] Se procesaron {to_process}/{n} páginas por OCR (ajustable con VISION_MAX_PAGES).")
    return "\n\n".join(textos).strip()

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

def extraer_texto_de_pdf(file) -> str:
    """
    1) Texto nativo con PyMuPDF.
    2) Si es muy poco (PDF escaneado), OCR con Vision para primeras N páginas.
    """
    raw = _leer_todo(file)
    if not raw:
        return ""
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            nativo = []
            for p in doc:
                # get_text() suele respetar rotaciones; extrae texto si es nativo
                t = p.get_text() or ""
                if t.strip():
                    nativo.append(t)
            plain = "\n".join(nativo).strip()
            # Heurística simple de “poco texto” => probablemente escaneado
            if len(plain) < 500:
                ocr_text = _ocr_con_vision(doc)
                return ocr_text if len(ocr_text) > len(plain) else plain
            return plain
    except Exception:
        # Si no abre como PDF, intentá decodificar como texto
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def extraer_texto_de_docx(file) -> str:
    """
    Extrae texto de DOCX (párrafos + tablas).
    Si python-docx no está instalado, intenta decode plano.
    """
    raw = _leer_todo(file)
    if not raw:
        return ""
    if docx is None:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    try:
        document = docx.Document(io.BytesIO(raw))
        partes: List[str] = []
        # Párrafos
        for p in document.paragraphs:
            txt = (p.text or "").strip()
            if txt:
                partes.append(txt)
        # Tablas -> líneas tipo "col1 | col2 | col3"
        for tbl in document.tables:
            for row in tbl.rows:
                celdas = []
                for cell in row.cells:
                    celdas.append((cell.text or "").strip())
                partes.append(" | ".join(celdas))
        return "\n".join(partes).strip()
    except Exception:
        try:
            return raw.decode("utf-8", errors="ignore")
        except Exception:
            return ""

def extraer_texto_de_imagen(file) -> str:
    """
    OCR de imagen (png/jpg/jpeg/webp) con Vision.
    """
    raw = _leer_todo(file)
    if not raw:
        return ""
    # Si no es PNG, convertimos a PNG en memoria usando PyMuPDF si es posible
    # PyMuPDF permite abrir imágenes como docs para rasterizar a PNG
    b64 = None
    try:
        # Intento 1: abrir como image-doc y re-exportar a PNG
        img_doc = fitz.open(stream=raw, filetype=_ext_de_archivo(file).lstrip(".") or None)
        page = img_doc.load_page(0)
        png = page.get_pixmap(alpha=False).tobytes("png")
        b64 = base64.b64encode(png).decode("utf-8")
    except Exception:
        # Intento 2: usar bytes tal cual si ya es PNG
        ext = _ext_de_archivo(file)
        if ext == ".png":
            b64 = base64.b64encode(raw).decode("utf-8")
        else:
            # Fallback: enviamos como png aunque no sea ideal
            b64 = base64.b64encode(raw).decode("utf-8")
    return _ocr_openai_imagen_b64(b64)

def extraer_texto_universal(file) -> str:
    """
    Lee múltiples tipos de archivo:
    - PDF (texto nativo u OCR)
    - DOCX (párrafos + tablas)
    - Imágenes PNG/JPG/JPEG/WEBP (OCR)
    - TXT / RTF básico
    - Otros → intenta decode UTF-8
    """
    ext = _ext_de_archivo(file)
    mime = _mime_guess(file)

    # PDF
    if ext == ".pdf" or (mime == "application/pdf"):
        return extraer_texto_de_pdf(file)

    # DOCX
    if ext == ".docx" or (mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]):
        return extraer_texto_de_docx(file)

    # Imágenes comunes
    if ext in [".png", ".jpg", ".jpeg", ".webp"] or mime.startswith("image/"):
        return extraer_texto_de_imagen(file)

    # TXT / RTF (muy básico: removemos marcas RTF simples)
    raw = _leer_todo(file)
    if not raw:
        return ""
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = ""
    if ext == ".rtf":
        # Limpieza simple de RTF — no es un parser completo
        text = re.sub(r"{\\rtf1.*?\\viewkind4\\uc1", "", text, flags=re.S)
        text = re.sub(r"\\[a-z]+-?\d* ?", "", text)
        text = text.replace("{", "").replace("}", "")
    return (text or "").strip()

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

CRAFT_PROMPT_MAESTRO = r"""
# C.R.A.F.T. — Informe quirúrgico de pliegos (multi-anexo)
Reglas clave:
- Trazabilidad: cada dato crítico con fuente `(Anexo X[, p. Y])` o `(Fuente: documento provisto)` si no hay paginación.
- Cero invenciones; si falta/ambigua, indicarlo y sugerir consulta.
- Cobertura completa de ciclo (oferta → ejecución), con normativa citada.
- Deduplicar, fusionar, no repetir; un único informe integrado.
- Prohibido meta texto tipo "parte X de Y" o "revise el resto".

Formato:
1) Resumen Ejecutivo (≤200 palabras)
2) Informe Extenso con Trazabilidad
   2.1 Identificación del llamado
   2.2 Calendario y lugares
   2.3 Contactos y portales (marca inconsistencias)
   2.4 Alcance y plazo contractual
   2.5 Tipología / modalidad (citar norma y artículos)
   2.6 Mantenimiento de oferta y prórroga
   2.7 Garantías (umbral UC, %, plazos, formas)
   2.8 Presentación de ofertas (soporte, firmas, neto/letras, origen/envases, parcial por renglón, docs obligatorias)
   2.9 Apertura, evaluación y adjudicación (tipo de cambio BNA, comisión, criterio, única oferta, preferencias)
   2.10 Subsanación (qué sí/no)
   2.11 Perfeccionamiento y modificaciones
   2.12 Entrega, lugares y plazos
   2.13 Planilla de cotización y renglones
   2.14 Muestras
   2.15 Cláusulas adicionales
   2.16 Matriz de Cumplimiento (tabla)
   2.17 Mapa de Anexos (tabla)
   2.18 Semáforo de Riesgos
   2.19 Checklist operativo
   2.20 Ambigüedades/Inconsistencias y Consultas Sugeridas
   2.21 Anexos del Informe (índice de trazabilidad)
3) Calidad: citas junto a cada dato; aplicar Guía de sinónimos.
"""

CRAFT_PROMPT_NOTAS = r"""
Genera NOTAS INTERMEDIAS CRAFT en bullets, ultra concisas, con cita al final de cada bullet.
- SOLO bullets (sin encabezados, sin "parte x/y", sin conclusiones).
- Etiqueta tema + cita en paréntesis. Si no hay paginación/ID: (Fuente: documento provisto).
- Aplica la Guía de sinónimos y conserva la terminología encontrada.
Ejemplos:
- [IDENTIFICACION] Organismo: ... (Anexo ?, p. ?)
- [CALENDARIO] Presentación: DD/MM/AAAA HH:MM — Lugar: ... (Fuente: documento provisto)
- [GARANTIAS] Mant. 5%; Cumpl. ≥10% ≤7 días hábiles (Anexo ?, p. ?)
- [INCONSISTENCIA] dominios ...gba.gov.ar vs ...pba.gov.ar (Fuente: documento provisto)
- [FALTA] campo X — no consta.
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

# ==================== Normalización para PDF (sin '#') ====================
_HDR_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_NUM_RE = re.compile(r"^\s*\d+[\.\)]\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```.*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_ITALIC_RE = re.compile(r"(\*\*|\*|__|_)(.*?)\1")

def _title_case(s: str) -> str:
    # Title Case simple para que .istitle() sea True → tu PDF lo pinta como encabezado
    return " ".join(w.capitalize() if w else w for w in re.split(r"(\s+)", s))

def preparar_texto_para_pdf(markdown_text: str) -> str:
    """
    Convierte Markdown a texto plano prolijo para tu PDF:
    - Quita '#' de encabezados y los pasa a Title Case (activando estilo en tu plantilla).
    - Convierte bullets a '•'.
    - Mantiene listas numeradas.
    - Elimina líneas separadoras de tablas (---|---).
    - Quita fences de código.
    - Convierte [texto](url) → 'texto (url)'.
    - Quita marcas ** **, * *, __ __ y _ _ (mantiene el contenido).
    """
    out_lines: List[str] = []
    for raw_ln in (markdown_text or "").splitlines():
        ln = raw_ln.rstrip()

        # Code fences → eliminar línea
        if _CODE_FENCE_RE.match(ln):
            continue

        # Encabezados '#'
        m = _HDR_RE.match(ln)
        if m:
            titulo = _title_case(m.group(2).strip(": ").strip())
            out_lines.append(titulo)
            continue

        # Remover separadores de tablas markdown
        if _TABLE_SEP_RE.match(ln):
            continue

        # Bullets → '• '
        if _BULLET_RE.match(ln):
            ln = _BULLET_RE.sub("• ", ln)

        # Links [t](u) → t (u)
        ln = _LINK_RE.sub(lambda mm: f"{mm.group(1)} ({mm.group(2)})", ln)

        # Quitar marcas de negrita/cursiva (dejar solo texto)
        ln = _BOLD_ITALIC_RE.sub(lambda mm: mm.group(2), ln)

        out_lines.append(ln)

    texto = "\n".join(out_lines)

    # Compactar saltos múltiples
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto

# ==================== Llamada a OpenAI robusta ====================
def _llamada_openai(messages, model=MODEL_ANALISIS, temperature_str=TEMPERATURE_ANALISIS,
                    max_completion_tokens=MAX_COMPLETION_TOKENS_SALIDA, retries=2, fallback_model="gpt-4o-mini"):
    """
    - Usa max_completion_tokens (no max_tokens).
    - Si temperature_str == "" no manda 'temperature' (evita error en modelos que no lo soportan).
    - Reintenta si choices vienen vacías o content vacío.
    - Fallback de modelo si el principal falla en el primer intento.
    """
    def _build_kwargs(mdl):
        kw = dict(model=mdl, messages=messages, max_completion_tokens=max_completion_tokens)
        if temperature_str != "":
            try:
                temp_val = float(temperature_str)
                kw["temperature"] = temp_val
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
                    import time; time.sleep(1.2 * (attempt + 1))
                else:
                    break
    raise RuntimeError(str(last_error) if last_error else "Fallo desconocido en _llamada_openai")

# ==================== Analizador principal ====================
def analizar_con_openai(texto: str) -> str:
    """
    Devuelve un único informe CRAFT.
    - Si el texto es corto y no hay indicios de multi-anexo → 1 pasada.
    - Si es largo o multi-anexo → notas intermedias + síntesis.
    """
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    # Detectar si hay varios anexos
    separadores = ["===ANEXO===", "=== ANEXO ===", "### ANEXO", "## ANEXO", "\nAnexo "]
    varios_anexos = any(sep.lower() in texto.lower() for sep in separadores)

    # Pasada única
    if len(texto) <= MAX_SINGLE_PASS_CHARS and not varios_anexos:
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{CRAFT_PROMPT_MAESTRO}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve ÚNICAMENTE el informe final (texto), sin preámbulos."}
        ]
        try:
            resp = _llamada_openai(messages)
            bruto = resp.choices[0].message.content.strip()
            limpio = _limpiar_meta(bruto)
            return preparar_texto_para_pdf(limpio)
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # Dos etapas (notas → síntesis)
    partes = _particionar(texto, CHUNK_SIZE)
    notas = []

    # A) Notas intermedias
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

    notas_integradas = "\n".join(notas)

    # B) Síntesis final
    messages_final = [
        {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
        {"role": "user", "content": f"""{CRAFT_PROMPT_MAESTRO}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

👉 Integra TODO en un **solo informe**; deduplica; cita una vez por dato con todas las fuentes.
👉 Prohibido meta-comentarios de fragmentos.
👉 Devuelve SOLO el informe final en texto.
"""}
    ]

    try:
        resp_final = _llamada_openai(messages_final, max_completion_tokens=MAX_COMPLETION_TOKENS_SALIDA)
        bruto = resp_final.choices[0].message.content.strip()
        limpio = _limpiar_meta(bruto)
        return preparar_texto_para_pdf(limpio)
    except Exception as e:
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias (limpias):\n{_limpiar_meta(notas_integradas)}"

# ==================== Multi-anexo ====================
def analizar_anexos(files: list) -> str:
    """
    Combina todos los anexos en un solo texto con marcadores y ejecuta el análisis integrado.
    Acepta PDF, DOCX, imágenes (PNG/JPG/JPEG/WEBP), TXT/RTF, etc.
    """
    if not files:
        return "No se recibieron anexos para analizar."

    bloques = []
    for idx, f in enumerate(files, 1):
        try:
            texto = extraer_texto_universal(f)
        except Exception:
            # Fallback de lectura plana
            try:
                f.file.seek(0)
                texto = f.file.read().decode("utf-8", errors="ignore")
            except Exception:
                texto = ""

        nombre = getattr(f, "filename", f"anexo_{idx}")
        if not nombre:
            nombre = f"anexo_{idx}"
        bloques.append(f"=== ANEXO {idx:02d}: {nombre} ===\n{texto}\n")

    contenido_unico = "\n".join(bloques).strip()
    if len(contenido_unico) < 100:
        return ("No se pudo extraer texto útil de los anexos. "
                "Verificá si los documentos están escaneados y elevá VISION_MAX_PAGES/DPI, "
                "o subí archivos en texto nativo.")

    return analizar_con_openai(contenido_unico)

# ==================== Chat (sin cambios sustanciales) ====================
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

# ==================== PDF (misma plantilla; texto más prolijo) ====================
def generar_pdf_con_plantilla(resumen: str, nombre_archivo: str):
    """
    Mantiene tu plantilla. Normaliza el cuerpo antes de dibujarlo
    para evitar '#', bullets feos y marcas de formato.
    """
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

    # Limpieza extra por si llega Markdown:
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
        # Heurística de encabezado: Title Case => istitle() True → azul y bold
        if parrafo.strip().endswith(":") or parrafo.strip().istitle():
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(azul)
        else:
            c.setFont("Helvetica", 11)
            c.setFillColor("black")
        for linea in dividir_texto(parrafo.strip(), c, ancho_texto):
            if y <= 20 * mm:
                c.showPage()
                if os.path.exists(plantilla_path):
                    c.drawImage(plantilla, 0, 0, width=A4[0], height=A4[1])
                c.setFont("Helvetica", 11)
                c.setFillColor("black")
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
