import io
import os
import re
import base64
from datetime import datetime

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor

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

# ==================== Utilidades de extracción ====================
def _rasterizar_pagina(page, dpi=VISION_DPI) -> bytes:
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def _ocr_con_vision(doc: fitz.Document, max_pages: int = VISION_MAX_PAGES) -> str:
    """
    OCR por OpenAI Vision (primeras N páginas). Devuelve texto literal por página.
    """
    textos = []
    n = len(doc)
    to_process = min(n, max_pages)
    for i in range(to_process):
        page = doc.load_page(i)
        png_bytes = _rasterizar_pagina(page)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        prompt = (
            "Extraé el TEXTO literal de esta página escaneada de un pliego. "
            "Conservá títulos, tablas como líneas, listas y números. No resumas ni interpretes."
        )
        try:
            resp = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                    ]
                }],
                max_completion_tokens=2400
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                textos.append(f"[PÁGINA {i+1}]\n{content}")
            else:
                textos.append(f"[PÁGINA {i+1}] (sin texto OCR)")
        except Exception as e:
            textos.append(f"[PÁGINA {i+1}] [OCR-ERROR] {e}")
    if n > to_process:
        textos.append(f"\n[AVISO] Se procesaron {to_process}/{n} páginas por OCR (ajustable con VISION_MAX_PAGES).")
    return "\n\n".join(textos).strip()

def extraer_texto_de_pdf(file) -> str:
    """
    1) Texto nativo con PyMuPDF.
    2) Si es muy poco (PDF escaneado), OCR con Vision para primeras N páginas.
    """
    raw = file.file.read()
    if not raw:
        return ""
    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            nativo = []
            for p in doc:
                t = p.get_text() or ""
                if t:
                    nativo.append(t)
            plain = "\n".join(nativo).strip()
            # Heurística simple de “poco texto”
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

Guía de sinónimos:
{SINONIMOS_CANONICOS}
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

# ==================== Llamada a OpenAI robusta ====================
def _llamada_openai(messages, model=MODEL_ANALISIS, temperature_str=TEMPERATURE_ANALISIS,
                    max_completion_tokens=MAX_COMPLETION_TOKENS_SALIDA, retries=2, fallback_model="gpt-4o-mini"):
    """
    - Usa max_completion_tokens (no max_tokens).
    - Si temperature_str == "" no manda 'temperature' (evita error de modelos que no lo soportan).
    - Reintenta si choices vienen vacías o content vacío.
    - Fallback de modelo si el principal falla en el primer intento.
    """
    # Construir kwargs sin temperature si está vacía
    def _build_kwargs(mdl):
        kw = dict(model=mdl, messages=messages, max_completion_tokens=max_completion_tokens)
        if temperature_str != "":
            try:
                # si no se puede parsear, no la mandamos
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
            return _limpiar_meta(resp.choices[0].message.content.strip())
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
        return _limpiar_meta(resp_final.choices[0].message.content.strip())
    except Exception as e:
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias (limpias):\n{_limpiar_meta(notas_integradas)}"

# ==================== Multi-anexo ====================
def analizar_anexos(files: list) -> str:
    """
    Combina todos los anexos en un solo texto con marcadores y ejecuta el análisis integrado.
    """
    if not files:
        return "No se recibieron anexos para analizar."

    bloques = []
    for idx, f in enumerate(files, 1):
        try:
            f.file.seek(0)
            texto = extraer_texto_de_pdf(f)
        except Exception:
            f.file.seek(0)
            try:
                texto = f.file.read().decode("utf-8", errors="ignore")
            except Exception:
                texto = ""

        nombre = getattr(f, "filename", f"anexo_{idx}.pdf")
        bloques.append(f"=== ANEXO {idx:02d}: {nombre} ===\n{texto}\n")

    contenido_unico = "\n".join(bloques).strip()
    if len(contenido_unico) < 100:
        return ("No se pudo extraer texto útil de los anexos. "
                "Verificá si los PDF están escaneados y elevá VISION_MAX_PAGES/ DPI, "
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

# ==================== PDF (igual que antes) ====================
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
    resumen = resumen.replace("**", "")
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
        if (
            parrafo.strip().endswith(":")
            or parrafo.strip().startswith("📘")
            or parrafo.strip().istitle()
        ):
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
