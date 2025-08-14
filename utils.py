# utils.py
import fitz  # PyMuPDF
import io
import os
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
    """Lee un PDF (file-like con .file) y devuelve el texto concatenado."""
    texto_completo = ""
    try:
        file.file.seek(0)
    except Exception:
        pass
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for pagina in doc:
            texto_completo += pagina.get_text()
    return texto_completo


# ============================================================
# ANALIZADOR (C.R.A.F.T.) — compatible con gpt-5 y chat.completions
# ============================================================

# Modelo (puedes override con OPENAI_MODEL_ANALISIS=gpt-5)
MODEL_ANALISIS = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-5")

# Heurísticas de particionado
MAX_SINGLE_PASS_CHARS = 45000   # si el texto total es menor, va en una sola pasada
CHUNK_SIZE = 12000              # tamaño por parte si es muy grande
TEMPERATURE_ANALISIS = 0.2
MAX_TOKENS_SALIDA = 4000        # margen para informes extensos

# -------- Prompt maestro (síntesis final única) --------
CRAFT_PROMPT_MAESTRO = r"""
# C.R.A.F.T. — Prompt maestro para leer, analizar y generar un **informe quirúrgico** de pliegos (con múltiples anexos)

## C — Contexto
Estás trabajando con **pliegos de licitación** (a menudo sanitarios) con **varios anexos**. La info es crítica: fechas, montos, artículos legales, decretos/resoluciones, modalidad, garantías, etc. Debes **leer todo**, **organizar**, **indexar** y producir un **informe técnico-jurídico completo**, claro y trazable.

**Reglas clave**
- **Trazabilidad total**: cada dato crítico con **fuente** `(Anexo X[, p. Y])`. Si el material provisto no trae paginación ni IDs, usa un marcador claro: `(Fuente: documento provisto)` o `(Anexo: no especificado)`.
- **Cero invenciones**: si un dato no aparece o es ambiguo, indicarlo y, si corresponde, proponer **consulta**.
- **Consistencia y cobertura total**: detectar incongruencias y cubrir oferta, evaluación, adjudicación, perfeccionamiento, ejecución.
- **Normativa**: citar (ley/decreto/resolución + artículo) con **fuente**.

## R — Rol
Actúas como equipo experto (Derecho Administrativo, Analista de Licitaciones Sanitarias, Redactor técnico-jurídico). Escritura técnica, sobria y precisa.

## A — Acción (resumen)
1) Indexar y normalizar (fechas DD/MM/AAAA, horas HH:MM, precios con 2 decimales).
2) Extraer **todos** los campos críticos (checklist).
3) Verificación cruzada: faltantes, ambigüedades, **inconsistencias** (dominios email, horarios, montos, etc.).
4) Análisis jurídico-operativo (modalidades, garantías, plazos, criterios, preferencias, etc.), citando normativa y fuentes.
5) **Construir un único informe** (sin repetir secciones), con **tablas** donde corresponda y **citas** en cada dato crítico.
6) Elaborar **consultas al comitente** para vacíos o ambigüedades.

## F — Formato (salida esperada, en texto)
### 1) Resumen Ejecutivo (≤200 palabras)
Objeto, organismo, proceso/modalidad, fechas clave, riesgos mayores, acciones inmediatas.

### 2) Informe Extenso con Trazabilidad
2.1 Identificación del llamado  
2.2 Calendario y lugares  
2.3 Contactos y portales (marcar inconsistencias de dominios si las hay)  
2.4 Alcance y plazo contractual  
2.5 Tipología / modalidad (con normativa y artículos citados)  
2.6 Mantenimiento de oferta y prórroga  
2.7 Garantías (umbral por UC, %, plazos, formas de constitución)  
2.8 Presentación de ofertas (soporte, firmas, neto/letras, origen/envases, parcial por renglón, documentación obligatoria: catálogos, LD 13.074, ARBA A-404, CBU BAPRO, AFIP/ARBA/CM, Registro, pago pliego, preferencias art. 22)  
2.9 Apertura, evaluación y adjudicación (tipo de cambio BNA, comisión, criterios, única oferta, facultades, preferencias)  
2.10 Subsanación (qué es subsanable vs no)  
2.11 Perfeccionamiento y modificaciones (plazos, topes, notificaciones y garantías)  
2.12 Entrega, lugar y plazos (dirección/horarios; inmediato/≤10 días O.C.; logística)  
2.13 Planilla de cotización y renglones (cantidad; estructura; totales en números y letras)  
2.14 Muestras (renglones con muestra y facultades del comitente)  
2.15 Cláusulas adicionales (anticorrupción; facturación/pago, etc.)  
2.16 Matriz de Cumplimiento (tabla)  
2.17 Mapa de Anexos (tabla)  
2.18 Semáforo de Riesgos (alto/medio/bajo)  
2.19 Checklist operativo para cotizar  
2.20 Ambigüedades / Inconsistencias y Consultas Sugeridas  
2.21 Anexos del Informe (índice de trazabilidad; glosario/normativa)

### 3) Estándares de calidad
- **Citas** al lado de cada dato crítico `(Anexo X[, p. Y])`. Si no hay paginación/ID en el insumo, indicarlo.
- **No repetir** contenido: deduplicar y usar referencias internas.
- Si hay discordancia unitario vs total, **explicar la regla aplicable** con cita.

## T — Público objetivo
Áreas de Compras/Contrataciones, Farmacia/Abastecimiento, Asesoría Legal y Dirección; proveedores del rubro. Español (AR), precisión jurídica y operatividad.

## Checklist de campos a extraer (mínimo)
Identificación; Calendario; Contactos/Portales; Alcance/Plazo; Modalidad/Normativa; Mantenimiento de oferta; Garantías; Presentación de ofertas; Apertura/Evaluación/Adjudicación; Subsanación; Perfeccionamiento/Modificaciones; Entrega; Planilla/Renglones; Muestras; Cláusulas adicionales; **Normativa citada**.

## Nota
- Devuelve **solo el informe final en texto**, perfectamente organizado. **No incluyas JSON**.
- No incluyas “parte 1/2/3” ni encabezados repetidos por cada segmento del documento.
"""

# -------- Prompt para "Notas intermedias" por chunk --------
CRAFT_PROMPT_NOTAS = r"""
Genera **NOTAS INTERMEDIAS CRAFT** ultra concisas para síntesis posterior, a partir del fragmento dado.
Reglas:
- Sin prosa larga ni secciones completas.
- Usa bullets con etiqueta del tema y la **cita** entre paréntesis.
- Si no hay paginación/ID disponible, usa `(Fuente: documento provisto)`.

Ejemplos de bullets:
- [IDENTIFICACION] Organismo: ... (Anexo ?, p. ?)
- [CALENDARIO] Presentación: DD/MM/AAAA HH:MM — Lugar: ... (Fuente: documento provisto)
- [GARANTIAS] Mantenimiento 5%; Cumplimiento ≥10% ≤7 días hábiles (Anexo ?, p. ?)
- [NORMATIVA] Decreto 59/19, art. X (Anexo ?, p. ?)
- [INCONSISTENCIA] Emails dominio ...gba.gov.ar vs ...pba.gov.ar (Fuente: documento provisto)
- [MUESTRAS] Renglones 23 y 24 (Anexo ?, p. ?)

No inventes. Si falta, anota: [FALTA] campo X — no consta.
Devuelve **solo bullets** (sin encabezados ni conclusiones).
"""


def _particionar(texto: str, max_chars: int) -> list[str]:
    return [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]


# ============== Compatibilidad con GPT-5 (Responses) / clásicos (Chat) ==============

def _requires_responses(model_name: str) -> bool:
    m = (model_name or "").lower()
    return m.startswith("gpt-5") or m.startswith("gpt-4.1") or m.startswith("o4")

def _chat_call(messages, model, temperature=0.2, max_tokens=2000) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def _responses_call(messages, model, max_tokens: int | None = None) -> str:
    # NO enviar 'temperature' (gpt-5 lo rechaza). 'max_output_tokens' es opcional según SDK.
    if not hasattr(client, "responses") or not hasattr(client.responses, "create"):
        raise RuntimeError(
            "Este modelo requiere la Responses API. Actualizá 'openai' a >= 1.40.0 y redeploy."
        )
    payload = {
        "model": model,
        "input": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    if max_tokens:
        payload["max_output_tokens"] = max_tokens
    resp = client.responses.create(**payload)

    # Unificar texto (distintas versiones de SDK)
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()

    chunks = []
    for o in getattr(resp, "output", []) or []:
        for b in getattr(o, "content", []) or []:
            if getattr(b, "type", "") == "output_text":
                chunks.append(getattr(b, "text", ""))
    if chunks:
        return "".join(chunks).strip()

    if hasattr(resp, "text") and resp.text:
        return resp.text.strip()

    return str(resp).strip()

def _llamada_openai(messages, model=MODEL_ANALISIS, temperature=TEMPERATURE_ANALISIS, max_tokens=MAX_TOKENS_SALIDA) -> str:
    """Llama a OpenAI usando Responses si el modelo lo requiere; si no, Chat Completions."""
    if _requires_responses(model):
        return _responses_call(messages, model, max_tokens=max_tokens)
    return _chat_call(messages, model, temperature=temperature, max_tokens=max_tokens)


def analizar_con_openai(texto: str) -> str:
    """
    Analiza el contenido completo y devuelve un único informe en texto,
    listo para renderizar a PDF.
    """
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    # Caso 1: una sola pasada
    if len(texto) <= MAX_SINGLE_PASS_CHARS:
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{CRAFT_PROMPT_MAESTRO}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve UNICAMENTE el informe final (texto), sin preámbulos."}
        ]
        try:
            contenido = _llamada_openai(messages)
            return contenido.strip()
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # Caso 2: dos etapas (notas intermedias + síntesis)
    partes = _particionar(texto, CHUNK_SIZE)
    notas = []

    # Etapa A: notas intermedias
    for i, parte in enumerate(partes, 1):
        msg = [
            {"role": "system", "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user", "content": f"{CRAFT_PROMPT_NOTAS}\n\n=== FRAGMENTO {i}/{len(partes)} ===\n{parte}"}
        ]
        try:
            r_texto = _llamada_openai(msg, max_tokens=2000)
            notas.append(r_texto.strip())
        except Exception as e:
            notas.append(f"[ERROR] No se pudieron generar notas de la parte {i}: {e}")

    notas_integradas = "\n".join(notas)

    # Etapa B: síntesis final única (informe completo)
    messages_final = [
        {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
        {"role": "user", "content": f"""{CRAFT_PROMPT_MAESTRO}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

👉 Usa UNICAMENTE estas notas para elaborar el informe final unico (sin repetir encabezados por fragmento, sin meta-comentarios).
👉 Devuelve SOLO el informe final en texto."""}
    ]

    try:
        informe = _llamada_openai(messages_final, max_tokens=MAX_TOKENS_SALIDA)
        return informe.strip()
    except Exception as e:
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias:\n{notas_integradas}"


# ============================================================
# NUEVO: integrar múltiples anexos (necesario para main.py)
# ============================================================
def analizar_anexos(files: list) -> str:
    """
    Recibe una lista de UploadFile y produce un único contenido etiquetado por anexo,
    luego llama a analizar_con_openai.
    """
    if not files:
        return "No se recibieron anexos para analizar."

    bloques = []
    for idx, f in enumerate(files, 1):
        # Intentar extraer como PDF
        try:
            texto = extraer_texto_de_pdf(f)
        except Exception:
            # Fallback: intentar leer texto plano
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
# Chat IA — ahora usando GPT-5 (con Responses si corresponde)
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
        messages = [
            {"role": "system", "content": "Actuás como un asistente experto en análisis de pliegos de licitación y soporte de plataformas digitales."},
            {"role": "user", "content": prompt}
        ]
        chat_model = "gpt-5"
        if _requires_responses(chat_model):
            return _responses_call(messages, chat_model, max_tokens=1200)
        return _chat_call(messages, chat_model, temperature=0.3, max_tokens=1200)
    except Exception as e:
        return f"⚠️ Error al generar respuesta: {e}"


# ============================================================
# Generación de PDF (igual que tu versión estable)
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
    resumen = (resumen or "").replace("**", "")
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
    palabras = (texto or "").split(" ")
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
