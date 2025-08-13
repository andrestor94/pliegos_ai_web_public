import fitz  # PyMuPDF
import io
import os
import re
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

def extraer_texto_de_pdf(file) -> str:
    texto_completo = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for pagina in doc:
            texto_completo += pagina.get_text()
    return texto_completo


# ============================================================
# ANALIZADOR (C.R.A.F.T. + GPT-5) con integración multi-anexo
# ============================================================

MODEL_ANALISIS = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-5")

# Heurísticas de particionado
MAX_SINGLE_PASS_CHARS = 55000   # subimos umbral para forzar síntesis única
CHUNK_SIZE = 16000              # menos cortes → mejor contexto
TEMPERATURE_ANALISIS = 0.2
MAX_TOKENS_SALIDA = 4000

# Guía de sinónimos/normalización para evitar "no especificado" falsos
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

# -------- Prompt maestro (síntesis final única) --------
CRAFT_PROMPT_MAESTRO = r"""
# C.R.A.F.T. — Informe quirúrgico de pliegos (múltiples anexos)

## C — Contexto
Trabajas con **pliegos** con **varios anexos**. La info es crítica (fechas, montos, normativa, garantías, etc.). Debes **leer TODO** e integrar en **un único informe** con **trazabilidad**.

**Reglas clave**
- **Trazabilidad total**: cada dato crítico con **fuente** `(Anexo X[, p. Y])`. Si no hay paginación/ID, usar `(Fuente: documento provisto)`.
- **Cero invenciones**; si algo falta/ambigua, indícalo y sugiere consulta.
- **Cobertura total** (oferta, evaluación, adjudicación, perfeccionamiento, ejecución).
- **Normativa** citada por tipo/numero/artículo con fuente.
- **No repetir** contenido: deduplicar y fusionar datos si aparecen en varios anexos.
- **Prohibido** incluir frases tipo: "parte 1 de 7", "informe basado en parte x/y", "revise el resto".

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

# -------- Prompt para "Notas intermedias" por chunk --------
CRAFT_PROMPT_NOTAS = r"""
Genera **NOTAS INTERMEDIAS CRAFT** ultra concisas para síntesis posterior, a partir del fragmento.
Reglas:
- SOLO bullets (sin encabezados, sin "parte x/y", sin conclusiones).
- Etiqueta del tema + **cita** entre paréntesis. Si no hay paginación/ID, usa `(Fuente: documento provisto)`.
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

def _llamada_openai(messages, model=MODEL_ANALISIS, temperature=TEMPERATURE_ANALISIS, max_tokens=MAX_TOKENS_SALIDA):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

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
    # Compactar múltiples líneas vacías consecutivas
    limpio = re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()
    return limpio

def analizar_con_openai(texto: str) -> str:
    """
    Analiza el contenido completo y devuelve **un único informe** en texto listo para PDF.
    - Si es corto y no hay múltiples anexos: 1 pasada.
    - Si es largo o con anexos: notas intermedias + síntesis final única.
    """
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    # Detectar si parece haber múltiples anexos (marcadores comunes)
    separadores = ["===ANEXO===", "=== ANEXO ===", "### ANEXO", "## ANEXO", "\nAnexo "]
    varios_anexos = any(sep.lower() in texto.lower() for sep in separadores)

    # Pasada única SOLO si es corto y no hay indicios de multi-anexo
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

    # Etapa A: notas intermedias
    for i, parte in enumerate(partes, 1):
        msg = [
            {"role": "system", "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user", "content": f"{CRAFT_PROMPT_NOTAS}\n\n## Guía de sinónimos/normalización\n{SINONIMOS_CANONICOS}\n\n=== FRAGMENTO {i}/{len(partes)} ===\n{parte}"}
        ]
        try:
            r = _llamada_openai(msg, max_tokens=2000)
            notas.append(r.choices[0].message.content.strip())
        except Exception as e:
            notas.append(f"[ERROR] No se pudieron generar notas de la parte {i}: {e}")

    notas_integradas = "\n".join(notas)

    # Etapa B: síntesis final única y deduplicada
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
        resp_final = _llamada_openai(messages_final)
        return _limpiar_meta(resp_final.choices[0].message.content.strip())
    except Exception as e:
        # Fallback: al menos devolver las notas (limpias de meta)
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias:\n{_limpiar_meta(notas_integradas)}"


# ============================
# NUEVO: integración multi-anexo
# ============================
def analizar_anexos(files: list) -> str:
    """
    Recibe una lista de UploadFile (Starlette/FastAPI), combina TODOS los anexos
    en un único texto con marcadores de anexo y ejecuta el análisis integrado.
    """
    if not files:
        return "No se recibieron anexos para analizar."

    bloques = []
    for idx, f in enumerate(files, 1):
        try:
            # Importante: hay que reposicionar el puntero para cada lectura
            f.file.seek(0)
            texto = extraer_texto_de_pdf(f)
        except Exception:
            # si no es PDF, intentar leer bytes y decodificar a texto simple
            f.file.seek(0)
            try:
                texto = f.file.read().decode("utf-8", errors="ignore")
            except Exception:
                texto = ""

        nombre = getattr(f, "filename", f"anexo_{idx}.pdf")
        bloques.append(f"=== ANEXO {idx:02d}: {nombre} ===\n{texto}\n")

    contenido_unico = "\n".join(bloques)
    return analizar_con_openai(contenido_unico)


# ============================================================
# (NO TOCAR) — Chat IA
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
"""

    if not contexto:
        contexto = "(No hay historial disponible actualmente.)"

    prompt = f"""
{descripcion_interfaz}

📂 Historial de análisis previos:
{contexto}

🧠 Pregunta del usuario:
{mensaje}

📌 Respondé de manera natural, directa y profesional. No repitas lo que hace la plataforma. Respondé exactamente lo que se te pregunta.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Actuás como un asistente experto en análisis de pliegos de licitación y soporte de plataformas digitales."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error al generar respuesta: {e}"


# ============================================================
# (SIN CAMBIOS) — Generación de PDF
# ============================================================
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
    lineas = []
    linea_actual = ""

    for palabra in palabras:
        test_line = linea_actual + " " + palabra if linea_actual else palabra
        if canvas_obj.stringWidth(test_line, canvas_obj._fontname, canvas_obj._fontsize) <= max_width:
            linea_actual = test_line
        else:
            lineas.append(linea_actual)
            linea_actual = palabra

    if linea_actual:
        lineas.append(linea_actual)

    return lineas
