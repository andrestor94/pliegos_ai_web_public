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

# ======================= PDF -> TEXTO =======================
def extraer_texto_de_pdf(file) -> str:
    """
    Lee un UploadFile (FastAPI) y devuelve texto plano.
    Si el PDF es escaneado sin OCR, devolverá poco o nada.
    """
    try:
        data = file.file.read()
        file.file.seek(0)
        texto_completo = ""
        with fitz.open(stream=data, filetype="pdf") as doc:
            for pagina in doc:
                texto_completo += pagina.get_text() or ""
        # Limpieza mínima
        texto_completo = "".join(ch for ch in texto_completo if ch >= " " or ch in "\n\r\t")
        return texto_completo.strip()
    except Exception as e:
        return f"[ERROR_EXTRAER_PDF] {e}"

# ============================================================
# ANALIZADOR (C.R.A.F.T.) — con fallbacks de modelo
# ============================================================
MODEL_ANALISIS = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-5")
# Fallbacks por si el principal falla o devuelve vacío
FALLBACK_MODELS = [m.strip() for m in os.getenv("OPENAI_MODEL_FALLBACKS", "gpt-4o,gpt-4o-mini").split(",") if m.strip()]

MAX_SINGLE_PASS_CHARS = int(os.getenv("MAX_SINGLE_PASS_CHARS", "45000"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "12000"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "4000"))  # mapeado a max_output_tokens

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
2.8 Presentación de ofertas (soporte, firmas, neto/letras, origen/envases, parcial por renglón, documentación obligatoria)  
2.9 Apertura, evaluación y adjudicación  
2.10 Subsanación  
2.11 Perfeccionamiento y modificaciones  
2.12 Entrega, lugar y plazos  
2.13 Planilla de cotización y renglones  
2.14 Muestras  
2.15 Cláusulas adicionales  
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
- No incluyas “parte 1/2/3” ni encabezados repetidos por fragmento, ni meta-comentarios.
"""

CRAFT_PROMPT_NOTAS = r"""
Genera **NOTAS INTERMEDIAS CRAFT** ultra concisas para síntesis posterior, a partir del fragmento dado.
Reglas:
- Sin prosa larga ni secciones completas.
- Usa bullets con etiqueta del tema y la **cita** entre paréntesis.
- Si no hay paginación/ID disponible, usa `(Fuente: documento provisto)`.

Ejemplos:
- [IDENTIFICACION] Organismo: ... (Anexo ?, p. ?)
- [CALENDARIO] Presentación: DD/MM/AAAA HH:MM — Lugar: ... (Fuente: documento provisto)
- [GARANTIAS] Mantenimiento 5%; Cumplimiento ≥10% ≤7 días hábiles (Anexo ?, p. ?)
- [NORMATIVA] Decreto 59/19, art. X (Anexo ?, p. ?)
- [INCONSISTENCIA] Emails ...gba.gov.ar vs ...pba.gov.ar (Fuente: documento provisto)
- [MUESTRAS] Renglones 23 y 24 (Anexo ?, p. ?)

No inventes. Si falta, anota: [FALTA] campo X — no consta.
Devuelve **solo bullets** (sin encabezados ni conclusiones).
"""

def _particionar(texto: str, max_chars: int) -> list[str]:
    return [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]

def _sanitize(s: str) -> str:
    if not s:
        return ""
    s = "".join(ch for ch in s if ch >= " " or ch in "\n\r\t")
    return s.strip()

def _responses_call(input_payload: list[dict], model: str) -> str:
    """
    Llama a Responses API (válida para GPT-5 y modelos recientes).
    Usa 'max_output_tokens' (equivalente moderno).
    Devuelve string (posiblemente vacío) o '__ERROR__::...'.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=input_payload,
            max_output_tokens=MAX_COMPLETION_TOKENS,
            # Si tu modelo lo tolera y querés usarlo, podrías agregar temperature aquí.
            # temperature=0.2,
        )
        content = getattr(resp, "output_text", "") or ""
        content = _sanitize(content)
        print(f"[ANALISIS] model={model} content_len={len(content)}")
        return content
    except Exception as e:
        print(f"[ANALISIS] exception model={model}: {repr(e)}")
        return f"__ERROR__::{e}"

def _call_openai(messages, model: str) -> str:
    """
    Adaptador: convierte 'messages' estilo chat a 'input' de Responses.
    """
    input_payload = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        input_payload.append({"role": role, "content": content})
    return _responses_call(input_payload, model)

def _try_models(messages) -> str:
    """
    Intenta con el modelo principal y luego con los fallbacks
    hasta obtener texto no vacío. Si todos fallan, devuelve "".
    """
    models = [MODEL_ANALISIS] + [m for m in FALLBACK_MODELS if m and m != MODEL_ANALISIS]
    for m in models:
        out = _call_openai(messages, m)
        if out and not out.startswith("__ERROR__::"):
            return out
    return ""

def analizar_con_openai(texto: str) -> str:
    """
    Devuelve un **informe único** en texto. Si los modelos no devuelven
    contenido, retorna un mensaje claro para el front.
    """
    texto = _sanitize(texto)
    if not texto:
        return "No se recibió contenido para analizar."

    # ---------- Caso 1: una sola pasada ----------
    if len(texto) <= MAX_SINGLE_PASS_CHARS:
        print(f"[ANALISIS] single-pass len={len(texto)}")
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{CRAFT_PROMPT_MAESTRO}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve ÚNICAMENTE el informe final (texto), sin preámbulos."}
        ]
        out = _try_models(messages)
        if not out:
            print("[ANALISIS] single-pass vacío → fallback")
            return ("No se pudo generar un resumen automático. Verificá que los PDFs "
                    "tengan texto seleccionable (no imágenes escaneadas) y que el contenido sea legible.")
        return out

    # ---------- Caso 2: particionado + síntesis ----------
    partes = _particionar(texto, CHUNK_SIZE)
    notas = []

    # A) Notas intermedias
    for i, parte in enumerate(partes, 1):
        msg = [
            {"role": "system", "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user", "content": f"{CRAFT_PROMPT_NOTAS}\n\n=== FRAGMENTO {i}/{len(partes)} ===\n{parte}"}
        ]
        out = _try_models(msg)
        notas.append(out or f"[FALTA] No se pudieron generar notas de la parte {i} — (Fuente: documento provisto)")

    notas_integradas = "\n".join(notas)

    # B) Síntesis final única
    messages_final = [
        {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
        {"role": "user", "content": f"""{CRAFT_PROMPT_MAESTRO}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

👉 Usa ÚNICAMENTE estas notas para elaborar el **informe final único** (sin repetir encabezados por fragmento, sin meta-comentarios).
👉 Devuelve SOLO el informe final en texto."""}
    ]
    out_final = _try_models(messages_final)
    if not out_final:
        return ("No se pudo generar un resumen automático. Verificá que los PDFs "
                "tengan texto seleccionable (no imágenes escaneadas) y que el contenido sea legible.")
    return out_final

# ============================================================
# Chat IA (migrado a Responses API)
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
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL_CHAT", "gpt-4o"),
            input=[
                {"role": "system", "content": "Actuás como un asistente experto en análisis de pliegos de licitación y soporte de plataformas digitales."},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=1200
        )
        return (getattr(resp, "output_text", "") or "").strip()
    except Exception as e:
        return f"⚠️ Error al generar respuesta: {e}"

# ============================================================
# Generación de PDF — sin cambios
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

    # Limpieza mínima: quitar ** y espacios raros
    resumen = (resumen or "").replace("**", "").strip()

    c.setFont("Helvetica", 11)
    margen_izquierdo = 20 * mm
    margen_superior = A4[1] - 54 * mm
    ancho_texto = 170 * mm
    alto_linea = 14
    y = margen_superior

    for parrafo in (resumen.split("\n") if resumen else ["(sin contenido)"]):
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
    lineas, linea_actual = [], ""
    for palabra in palabras:
        test = (linea_actual + " " + palabra).strip() if linea_actual else palabra
        if canvas_obj.stringWidth(test, canvas_obj._fontname, canvas_obj._fontsize) <= max_width:
            linea_actual = test
        else:
            if linea_actual:
                lineas.append(linea_actual)
            linea_actual = palabra
    if linea_actual:
        lineas.append(linea_actual)
    return lineas
