# utils.py
import fitz  # PyMuPDF
import io
import os
import re
import unicodedata
from datetime import datetime
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# Utilidades de E/S
# ============================================================

def extraer_texto_de_pdf(file) -> str:
    """Lee un PDF (file-like con atributo .file) y devuelve texto concatenado."""
    texto_completo = ""
    file.file.seek(0)
    data = file.file.read()
    with fitz.open(stream=data, filetype="pdf") as doc:
        for pagina in doc:
            texto_completo += pagina.get_text()
    return texto_completo


# ============================================================
# ANALIZADOR (C.R.A.F.T. + GPT-5) con integración multi-anexo
# ============================================================

MODEL_ANALISIS = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-5")

# Heurísticas de particionado
MAX_SINGLE_PASS_CHARS = 55000
CHUNK_SIZE = 16000
TEMPERATURE_ANALISIS = 0.2
MAX_TOKENS_SALIDA = 4000

# Guía de sinónimos/normalización
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

# -------- Prompt maestro --------
CRAFT_PROMPT_MAESTRO = r"""
# C.R.A.F.T. — Informe quirúrgico de pliegos (múltiples anexos)

## C — Contexto
Trabajas con **pliegos** con **varios anexos**. La info es crítica (fechas, montos, normativa, garantías, etc.). Debes **leer TODO** e integrar en **un único informe** con **trazabilidad**.

**Reglas clave**
- **Trazabilidad total**: cada dato crítico con **fuente** (Anexo X[, p. Y]). Si no hay paginación/ID, usar (Fuente: documento provisto).
- **Cero invenciones**; si algo falta/ambigua, indícalo y sugiere consulta.
- **Cobertura total** (oferta, evaluación, adjudicación, perfeccionamiento, ejecución).
- **Normativa** citada por tipo/numero/artículo con fuente.
- **No repetir** contenido: deduplicar y fusionar datos si aparecen en varios anexos.
- **No repitas encabezados genéricos** (p. ej. “Informe Estandarizado de pliego de Licitación”). Si aparece, incluirlo **una sola vez** al inicio del documento o **omitirlo**.

## R — Rol
Equipo experto (Derecho Administrativo, Analista de Licitaciones Sanitarias, Redactor técnico-jurídico). Estilo técnico, sobrio y preciso.

## A — Acción
1) Indexar y normalizar (fechas DD/MM/AAAA, horas HH:MM, precios 2 decimales).
2) Extraer todos los campos (checklist).
3) Verificación cruzada: faltantes, **inconsistencias** (dominios email, horarios, montos).
4) Análisis jurídico-operativo citando normativa y fuentes.
5) **Un único informe integrado** (sin encabezados repetidos).
6) Consultas al comitente para vacíos/ambigüedades.

## F — Formato (salida en texto)
### 1) Resumen Ejecutivo (≤200 palabras)
Objeto, organismo, proceso/modalidad, fechas clave, riesgos, acciones inmediatas.

### 2) Informe Extenso con Trazabilidad
2.1 Identificación del llamado  
2.2 Calendario y lugares  
2.3 Contactos y portales (marcar inconsistencias de dominios si las hay)  
2.4 Alcance y plazo contractual  
2.5 Tipología / modalidad (con normativa y artículos citados)  
2.6 Mantenimiento de oferta y prórroga  
2.7 Garantías (umbral UC, %, plazos, formas)  
2.8 Presentación de ofertas (soporte, firmas, neto/letras, origen/envases, parcial por renglón, docs obligatorias: catálogos, LD 13.074, ARBA A-404, CBU BAPRO, AFIP/ARBA/CM, Registro, pago pliego, preferencias art. 22)  
2.9 Apertura, evaluación y adjudicación (tipo de cambio BNA, comisión, criterio, única oferta, facultades, preferencias)  
2.10 Subsanación (qué sí/no)  
2.11 Perfeccionamiento y modificaciones (plazos, topes, notificaciones y garantías)  
2.12 Entrega, lugar y plazos (dirección/horarios; inmediato/≤10 días O.C.; logística)  
2.13 Planilla de cotización y renglones  
2.14 Muestras  
2.15 Cláusulas adicionales  
2.16 Matriz de Cumplimiento (tabla)  
2.17 Mapa de Anexos (tabla)  
2.18 Semáforo de Riesgos  
2.19 Checklist operativo  
2.20 Ambigüedades / Inconsistencias y Consultas Sugeridas  
2.21 Anexos del Informe (índice de trazabilidad; glosario/normativa)

### 3) Calidad
- Citas junto a cada dato crítico.
- **No marcar "Información no especificada"** si el dato aparece con sinónimos/variantes (ver **Guía**).
- Si hay discordancia unitario vs total, explicar la regla (con cita).

## Guía de sinónimos/normalización
{SINONIMOS_CANONICOS}
"""

# -------- Prompt para "Notas intermedias" --------
CRAFT_PROMPT_NOTAS = r"""
Genera **NOTAS INTERMEDIAS CRAFT** ultra concisas para síntesis posterior, a partir del fragmento.
Reglas:
- SOLO bullets (sin encabezados, sin "parte x/y", sin conclusiones).
- Etiqueta del tema + **cita** entre paréntesis. Si no hay paginación/ID, usa (Fuente: documento provisto).
- Aplica la **Guía de sinónimos/normalización**: si aparece con nombre alternativo, consérvalo.

Ejemplos:
- [IDENTIFICACION] Organismo: ... (Anexo ?, p. ?)
- [CALENDARIO] Presentación: DD/MM/AAAA HH:MM — Lugar: ... (Fuente: documento provisto)
- [GARANTIAS] Mant. 5%; Cumpl. ≥10% ≤7 días hábiles (Anexo ?, p. ?)
- [NORMATIVA] Decreto 59/19, art. X (Anexo ?, p. ?)
- [INCONSISTENCIA] dominios ...gba.gov.ar vs ...pba.gov.ar (Fuente: documento provisto)
- [MUESTRAS] renglones 23 y 24 (Anexo ?, p. ?)

Si falta, anota: [FALTA] campo X — no consta.
"""

def _particionar(texto: str, max_chars: int) -> list[str]:
    return [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]


# ============================================================
# Helper LLM: usa Responses para gpt-5 / gpt-4.1 / o4
# ============================================================

def call_llm(model: str, messages: list, max_tokens: int = 2000, temperature: float = 0.2) -> str:
    """
    Usa Responses API para modelos que lo requieren (p.ej. gpt-5, gpt-4.1, o4).
    Solo usa Chat Completions si el modelo NO requiere Responses.
    Evita el error: Unsupported parameter 'max_tokens' (usar 'max_completion_tokens').
    """
    def _requires_responses(m: str) -> bool:
        m = (m or "").lower()
        return m.startswith("gpt-5") or m.startswith("gpt-4.1") or m.startswith("o4")

    if _requires_responses(model):
        # Debe existir client.responses.create en openai>=1.40.0
        if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
            raise RuntimeError(
                "Este modelo requiere la Responses API. Actualizá el paquete 'openai' a >= 1.40.0 y redeploy."
            )
        resp = client.responses.create(
            model=model,
            input=[{"role": m["role"], "content": m["content"]} for m in messages],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        # Unificar texto (varía según SDK)
        chunks = []
        for o in getattr(resp, "output", []) or []:
            for b in getattr(o, "content", []) or []:
                if getattr(b, "type", "") == "output_text":
                    chunks.append(getattr(b, "text", ""))
        if chunks:
            return "".join(chunks).strip()
        if hasattr(resp, "output_text"):
            return (resp.output_text or "").strip()
        if hasattr(resp, "text"):
            return (resp.text or "").strip()
        return str(resp).strip()

    # Modelos clásicos (Chat Completions)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _llamada_openai(messages, model=MODEL_ANALISIS, temperature=TEMPERATURE_ANALISIS, max_tokens=MAX_TOKENS_SALIDA) -> str:
    """Wrapper usado por el flujo de análisis."""
    return call_llm(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)


# --- Filtros de meta y post-procesado anti-duplicados ---------------------

_META_PATTERNS = [
    re.compile(r"(?i)\bparte\s+\d+\s+de\s+\d+"),
    re.compile(r"(?i)informe\s+basado\s+en\s+la\s+parte"),
    re.compile(r"(?i)revise\s+las\s+partes\s+restantes"),
    re.compile(r"(?i)informaci[oó]n\s+puede\s+estar\s+incompleta"),
]

# Viñetas/símbolos comunes al inicio
_BULLETS = r"[•●◦▪▫■□▶»\-–—#]*"

# Patrón ancho para detectar variaciones del encabezado
_HEADER_PATTERNS = [
    re.compile(
        rf"(?i)^\s*(?:{_BULLETS}\s*)*informe\s+(?:estandarizado\s+de\s+)?pliego\s+de\s+licitaci[oó]n\s*:?.*$"
    ),
    re.compile(
        rf"(?i)^\s*(?:{_BULLETS}\s*)*informe\s+de\s+pliego\s+de\s+licitaci[oó]n\s*:?.*$"
    ),
]

# Patrón global robusto para “parte X de Y”
_PARTES_GLOBAL = re.compile(r"(?is)^.*\bparte\s*\W*\s*\d+\s*\W*\s*de\s*\W*\s*\d+\b.*$", re.MULTILINE)

# Placeholders que deben eliminarse en PDF/informe
_PLACEHOLDERS_VACIO = {
    "información no especificada en la parte proporcionada",
    "información no especificada en el pliego",
    "información no especificada",
    "no especificado en la parte proporcionada",
    "no especificado en el pliego",
    "no especificado",
    "no se considera necesario en este informe",
    "-",
}

def _normalize(s: str) -> str:
    """Normaliza unicode, baja a minúsculas, colapsa espacios y elimina BOM/espacios raros."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\ufeff", "")
    s = re.sub(rf"^\s*{_BULLETS}\s*", "", s)  # quito bullets/ hashes iniciales
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    table = str.maketrans("áéíóúüñ", "aeiouun")
    s = s.translate(table)
    return s

def _es_header_informe(ln: str) -> bool:
    if any(p.match(ln) for p in _HEADER_PATTERNS):
        return True
    norm = _normalize(ln)
    return ("informe" in norm and "pliego" in norm and "licitacion" in norm and norm.startswith("informe"))

def _limpiar_meta(texto: str) -> str:
    # 1) eliminar líneas con “parte X de Y”
    texto = _PARTES_GLOBAL.sub("", texto)
    # 2) eliminar otras advertencias meta
    lineas = []
    for ln in texto.splitlines():
        if any(p.search(ln) for p in _META_PATTERNS):
            continue
        lineas.append(ln)
    limpio = re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()
    return limpio

def _dedupe_headers(texto: str, keep_first: bool = True) -> str:
    """
    Deja solo la PRIMERA aparición del encabezado "Informe ... Pliego ..." y
    elimina todas las siguientes (con o sin viñetas/## etc.).
    """
    seen = False
    out = []
    for ln in texto.splitlines():
        if _es_header_informe(ln):
            if keep_first and not seen:
                seen = True
                out.append(ln.strip())  # mantenemos una sola vez
            continue
        out.append(ln)
    res = "\n".join(out)
    res = re.sub(r"\n{3,}", "\n\n", res).strip()
    return res

def _eliminar_placeholders_vacios(texto: str) -> str:
    """Quita líneas que sean exactamente placeholders de 'no especificado...'."""
    out = []
    for ln in texto.splitlines():
        if _normalize(ln) in _PLACEHOLDERS_VACIO:
            continue
        out.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()

def _dedupe_consecutivos(texto: str) -> str:
    """Elimina duplicados consecutivos de la MISMA línea (caso repeticiones pegadas)."""
    out = []
    prev = None
    for ln in texto.splitlines():
        if prev is not None and _normalize(prev) == _normalize(ln):
            continue
        out.append(ln)
        prev = ln
    return "\n".join(out)

def _postprocesar_informe(texto: str) -> str:
    """Pipeline de limpieza del informe final."""
    texto = _limpiar_meta(texto)
    texto = _dedupe_headers(texto, keep_first=True)
    texto = _eliminar_placeholders_vacios(texto)
    texto = _dedupe_consecutivos(texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto


def analizar_con_openai(texto: str) -> str:
    """
    Analiza el contenido completo y devuelve **un único informe** en texto listo para PDF.
    - Si es corto y no hay múltiples anexos: 1 pasada.
    - Si es largo o con anexos: notas intermedias + síntesis final única.
    """
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    separadores = ["===ANEXO===", "=== ANEXO ===", "### ANEXO", "## ANEXO", "\nAnexo "]
    varios_anexos = any(sep.lower() in texto.lower() for sep in separadores)

    # Pasada única
    if len(texto) <= MAX_SINGLE_PASS_CHARS and not varios_anexos:
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{CRAFT_PROMPT_MAESTRO}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve ÚNICAMENTE el informe final (texto), sin preámbulos."}
        ]
        try:
            resp_text = _llamada_openai(messages)
            return _postprocesar_informe(resp_text)
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # Dos etapas (notas → síntesis)
    partes = _particionar(texto, CHUNK_SIZE)
    notas = []

    # Etapa A: notas intermedias
    for i, parte in enumerate(partes, 1):
        msg = [
            {"role": "system", "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user", "content": f"{CRAFT_PROMPT_NOTAS}\n\n(No incluyas ningún encabezado genérico como 'Informe Estandarizado de pliego de Licitación' u otros títulos repetitivos.)\n\n## Guía de sinónimos/normalización\n{SINONIMOS_CANONICOS}\n\n=== FRAGMENTO {i}/{len(partes)} ===\n{parte}"}
        ]
        try:
            r_text = _llamada_openai(msg, max_tokens=2000)
            notas.append(_postprocesar_informe(r_text))
        except Exception as e:
            notas.append(f"[ERROR] No se pudieron generar notas de la parte {i}: {e}")

    notas_integradas = _postprocesar_informe("\n".join(notas))

    # Etapa B: síntesis final única
    messages_final = [
        {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
        {"role": "user", "content": f"""{CRAFT_PROMPT_MAESTRO}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

👉 Integra TODO en un **solo informe**; deduplica; cita una vez por dato con todas las fuentes.
👉 **Prohibido** incluir frases del tipo "parte X de Y" o meta-comentarios sobre fragmentos.
👉 Devuelve SOLO el informe final en texto.
"""}
    ]

    try:
        resp_final = _llamada_openai(messages_final, max_tokens=MAX_TOKENS_SALIDA)
        return _postprocesar_informe(resp_final)
    except Exception as e:
        # Fallback: al menos devolver las notas (ya limpias)
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias:\n{_postprocesar_informe(notas_integradas)}"


# ============================
# Integración multi-anexo
# ============================
def analizar_anexos(files: list) -> str:
    if not files:
        return "No se recibieron anexos para analizar."

    bloques = []
    for idx, f in enumerate(files, 1):
        # Extraer texto cuidando el puntero
        try:
            texto = extraer_texto_de_pdf(f)
        except Exception:
            try:
                f.file.seek(0)
                texto = f.file.read().decode("utf-8", errors="ignore")
            except Exception:
                texto = ""

        nombre = getattr(f, "filename", f"anexo_{idx}.pdf")
        bloques.append(f"=== ANEXO {idx:02d}: {nombre} ===\n{texto}\n")

    contenido_unico = "\n".join(bloques)
    return analizar_con_openai(contenido_unico)


# ============================================================
# Chat IA (usa call_llm y evita 400)
# ============================================================
def responder_chat_openai(mensaje: str, contexto: str = "", usuario: str = "Usuario") -> str:
    descripcion_interfaz = f"""
Sos el asistente inteligente de la plataforma web "Suizo Argentina - Licitaciones IA". Esta plataforma permite:

- Cargar y analizar múltiples archivos PDF que conforman un pliego.
- Obtener un resumen profesional con estructura estandarizada.
- Consultar un historial de análisis realizados por cada usuario.
- Descargar informes en PDF con diseño institucional.
- Crear tickets de soporte.
- Administrar usuarios (solo rol admin).
- Usar este chat para responder consultas sobre pliegos o la interfaz.

Tu función principal es asistir al usuario en el entendimiento y lectura de los pliegos analizados. También brindás soporte sobre el uso general de la plataforma.

El usuario actual es: {usuario}
""".strip()

    if not contexto:
        contexto = "(No hay historial disponible actualmente.)"

    prompt = f"""
{descripcion_interfaz}

📂 Historial de análisis previos:
{contexto}

🧠 Pregunta del usuario:
{mensaje}

📌 Respondé de manera natural, directa y profesional. No repitas lo que hace la plataforma. Respondé exactamente lo que se te pregunta.
""".strip()

    try:
        messages = [
            {"role": "system", "content": "Actuás como un asistente experto en análisis de pliegos de licitación y soporte de plataformas digitales."},
            {"role": "user", "content": prompt}
        ]
        # Para chat seguimos con gpt-4o (acepta chat.completions). Ajustá si querés.
        return call_llm(model="gpt-4o", messages=messages, max_tokens=1200, temperature=0.3)
    except Exception as e:
        return f"⚠️ Error al generar respuesta: {e}"


# ============================================================
# Generación de PDF (con limpieza previa y títulos controlados)
# ============================================================

# Patrones de "línea título" aceptados para negrita
_RE_TITULO = re.compile(
    r"""(?ix)
    ( # cualquiera de estos:
      ^\s*(\d+(\.\d+){0,3})\s+[^\:]{2,}\:?\s*$   # "2.1 Identificación ..." (opcional :)
     |^\s*[-–—•]\s+[^\:]{2,}\:\s*$              # bullet + texto:
     |^\s*[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑ0-9\s]{3,}\:\s*$ # Texto Capitalizado que termina en ":"
    )
    """.strip()
)

def _preparar_texto_para_pdf(texto: str) -> str:
    """Limpieza y normalización adicionales específicas para PDF."""
    t = _postprocesar_informe(texto)
    # Quitar espacios al borde y normalizar saltos
    t = "\n".join([ln.rstrip() for ln in t.splitlines()])
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

def generar_pdf_con_plantilla(resumen: str, nombre_archivo: str):
    """
    Genera un PDF simple (institucional) a partir de texto.
    Reglas:
    - Aplica limpieza para evitar encabezados repetidos y placeholders.
    - Considera título solo si termina en ":" o coincide con patrón de sección.
    """
    output_dir = os.path.join("generated_pdfs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, nombre_archivo)

    # --- Limpieza previa del texto ---
    resumen = _preparar_texto_para_pdf((resumen or "").replace("**", ""))

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    # Fondo institucional (opcional)
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

    # Tipografías base
    c.setFont("Helvetica", 11)
    margen_izquierdo = 20 * mm
    margen_superior = A4[1] - 54 * mm
    ancho_texto = 170 * mm
    alto_linea = 14
    y = margen_superior

    def _es_titulo_linea(s: str) -> bool:
        s_strip = s.strip()
        if not s_strip:
            return False
        if s_strip.endswith(":"):
            return True
        # patrón de sección (2.1, 2.10, etc.) o bullets formales
        return bool(_RE_TITULO.match(s_strip))

    for parrafo in resumen.split("\n"):
        p = parrafo.strip()
        if not p:
            y -= alto_linea
            continue

        # Seteo estilo según sea título o cuerpo
        if _es_titulo_linea(p):
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(azul)
        else:
            c.setFont("Helvetica", 11)
            c.setFillColor("black")

        # Escrito con wrap
        for linea in dividir_texto(p, c, ancho_texto):
            if y <= 20 * mm:
                c.showPage()
                if os.path.exists(plantilla_path):
                    c.drawImage(plantilla, 0, 0, width=A4[0], height=A4[1])
                c.setFont("Helvetica", 11)
                c.setFillColor("black")
                y = margen_superior
            c.drawString(margen_izquierdo, y, linea)
            y -= alto_linea
        y -= 6  # espacio entre párrafos

    c.save()
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())

    return output_path


def dividir_texto(texto, canvas_obj, max_width):
    """Word-wrap simple basado en ancho medido por ReportLab."""
    palabras = (texto or "").split(" ")
    lineas = []
    linea_actual = ""

    for palabra in palabras:
        test_line = (linea_actual + " " + palabra).strip() if linea_actual else palabra
        if canvas_obj.stringWidth(test_line, canvas_obj._fontname, canvas_obj._fontsize) <= max_width:
            linea_actual = test_line
        else:
            if linea_actual:
                lineas.append(linea_actual)
            linea_actual = palabra

    if linea_actual:
        lineas.append(linea_actual)

    return lineas
