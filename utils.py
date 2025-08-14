import fitz  # PyMuPDF
import io
import os
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List
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
# Extracción básica de texto por tipo de archivo
# ============================================================

def extraer_texto_de_pdf(file) -> str:
    """Lee un UploadFile PDF y devuelve su texto."""
    try:
        file.file.seek(0)
        with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)
    except Exception:
        return ""

def _read_bytes(upload_file) -> bytes:
    try:
        upload_file.file.seek(0)
    except Exception:
        pass
    return upload_file.file.read()

def _txt_from_txt(upload_file) -> str:
    data = _read_bytes(upload_file)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _txt_from_csv(upload_file) -> str:
    data = _read_bytes(upload_file)
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _txt_from_docx(upload_file) -> str:
    """
    Extractor DOCX sin dependencias externas:
    Lee word/document.xml y concatena los nodos w:t.
    """
    try:
        data = _read_bytes(upload_file)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open("word/document.xml") as f:
                xml = f.read()
        root = ET.fromstring(xml)
        ns = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        }
        texts = []
        for t in root.findall(".//w:t", ns):
            if t.text:
                texts.append(t.text)
        return " ".join(texts)
    except Exception:
        return ""

def _txt_from_pptx(upload_file) -> str:
    """
    Extractor PPTX sin dependencias externas:
    Recorre ppt/slides/slide*.xml y concatena a:t (text runs).
    """
    try:
        data = _read_bytes(upload_file)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            slide_names = [n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            slide_names.sort()
            chunks = []
            for name in slide_names:
                with z.open(name) as f:
                    xml = f.read()
                root = ET.fromstring(xml)
                ns = {
                    "a": "http://schemas.openxmlformats.org/drawingml/2006/main"
                }
                for t in root.findall(".//a:t", ns):
                    if t.text:
                        chunks.append(t.text)
            return "\n".join(chunks)
    except Exception:
        return ""

def _txt_from_xlsx(upload_file) -> str:
    """
    Extractor XLSX simple sin dependencias externas:
    - Lee sharedStrings.xml para recuperar strings compartidas.
    - Recorre worksheets y mapea c/@t == 's' con sharedStrings.
    Nota: Es un best-effort básico.
    """
    try:
        data = _read_bytes(upload_file)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            # sharedStrings
            shared = []
            if "xl/sharedStrings.xml" in z.namelist():
                with z.open("xl/sharedStrings.xml") as f:
                    xml = f.read()
                root = ET.fromstring(xml)
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in root.findall(".//s:si", ns):
                    # Un si puede tener múltiples t (con formato), los unimos
                    parts = []
                    for t in si.findall(".//s:t", ns):
                        if t.text:
                            parts.append(t.text)
                    shared.append("".join(parts))
            # hojas
            ws_names = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
            ws_names.sort()
            out_lines = []
            ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for name in ws_names:
                with z.open(name) as f:
                    xml = f.read()
                root = ET.fromstring(xml)
                for row in root.findall(".//s:row", ns):
                    row_vals = []
                    for c in row.findall("s:c", ns):
                        typ = c.get("t")
                        v = c.find("s:v", ns)
                        if v is None or v.text is None:
                            row_vals.append("")
                            continue
                        if typ == "s":
                            # índice a shared strings
                            try:
                                idx = int(v.text)
                                row_vals.append(shared[idx] if 0 <= idx < len(shared) else "")
                            except Exception:
                                row_vals.append("")
                        else:
                            # números/fechas (sin formato)
                            row_vals.append(v.text)
                    if any(cell.strip() for cell in row_vals):
                        out_lines.append(" | ".join(row_vals))
            return "\n".join(out_lines)
    except Exception:
        return ""

def _ext(fname: str) -> str:
    return os.path.splitext((fname or "").lower())[1]

def _safe_name(fname: str) -> str:
    base = os.path.basename(fname or "archivo")
    return "".join(c for c in base if c.isalnum() or c in ("-", "_", ".", " "))

# ============================================================
# Analizador multi-anexo para la ruta /analizar-pliego
# ============================================================

def analizar_anexos(archivos: List) -> str:
    """
    Recibe una lista de UploadFile (FastAPI) y:
      1) Extrae texto de cada uno (PDF, TXT, CSV, DOCX, PPTX, XLSX best-effort).
      2) Concatena con separadores y metadatos mínimos.
      3) Llama a analizar_con_openai() para producir el informe final único.
    Devuelve el informe como str. Si no hay texto legible, retorna mensaje claro.
    """
    if not archivos:
        return "No se recibió ningún archivo."

    piezas = []
    legibles = 0

    for i, a in enumerate(archivos, start=1):
        if not a or not getattr(a, "filename", ""):
            continue
        nombre = _safe_name(a.filename)
        ext = _ext(nombre)
        texto = ""

        if ext == ".pdf":
            texto = extraer_texto_de_pdf(a)
        elif ext in {".txt"}:
            texto = _txt_from_txt(a)
        elif ext in {".csv"}:
            texto = _txt_from_csv(a)
        elif ext in {".docx"}:
            texto = _txt_from_docx(a)
        elif ext in {".pptx"}:
            texto = _txt_from_pptx(a)
        elif ext in {".xlsx"}:
            texto = _txt_from_xlsx(a)
        else:
            # Otros no soportados de forma nativa
            texto = ""

        if texto.strip():
            legibles += 1
            piezas.append(f"\n\n===== ANEXO {i}: {nombre} =====\n{texto.strip()}\n")
        else:
            piezas.append(f"\n\n===== ANEXO {i}: {nombre} =====\n[No se pudo extraer texto legible de este archivo]\n")

    corpus = "\n".join(piezas).strip()

    if not corpus:
        return "No se pudo extraer texto de los anexos provistos. Verificá que sean PDFs o formatos de texto legibles."

    # Llamada al analizador C.R.A.F.T.
    return analizar_con_openai(corpus)

# ============================================================
# NUEVO ANALIZADOR (usa tu C.R.A.F.T. mejorado + GPT-5)
# ============================================================

MODEL_ANALISIS = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-5")

MAX_SINGLE_PASS_CHARS = 45000
CHUNK_SIZE = 12000
MAX_TOKENS_SALIDA = 4000

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
Actuás como equipo experto (Derecho Administrativo, Analista de Licitaciones Sanitarias, Redactor técnico-jurídico). Escritura técnica, sobria y precisa.

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

def _llamada_openai(messages, model=MODEL_ANALISIS, max_tokens=MAX_TOKENS_SALIDA):
    # Importante: usar max_completion_tokens y no pasar temperature para evitar errores del modelo.
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=max_tokens
    )

def analizar_con_openai(texto: str) -> str:
    """
    Si es corto: una sola pasada (síntesis final).
    Si es largo: notas intermedias por chunk + síntesis final única.
    """
    if not texto or not texto.strip():
        return "No se recibió contenido para analizar."

    # Caso 1: una sola pasada
    if len(texto) <= MAX_SINGLE_PASS_CHARS:
        messages = [
            {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
            {"role": "user", "content": f"{CRAFT_PROMPT_MAESTRO}\n\n=== CONTENIDO COMPLETO DEL PLIEGO ===\n{texto}\n\n👉 Devuelve ÚNICAMENTE el informe final (texto), sin preámbulos."}
        ]
        try:
            resp = _llamada_openai(messages)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"⚠️ Error al generar el análisis: {e}"

    # Caso 2: dos etapas
    partes = _particionar(texto, CHUNK_SIZE)
    notas = []

    # Etapa A: notas intermedias
    for i, parte in enumerate(partes, 1):
        msg = [
            {"role": "system", "content": "Eres un analista jurídico que extrae bullets técnicos con citas; cero invenciones; máxima concisión."},
            {"role": "user", "content": f"{CRAFT_PROMPT_NOTAS}\n\n=== FRAGMENTO {i}/{len(partes)} ===\n{parte}"}
        ]
        try:
            r = _llamada_openai(msg, max_tokens=2000)
            notas.append((r.choices[0].message.content or "").strip())
        except Exception as e:
            notas.append(f"[ERROR] No se pudieron generar notas de la parte {i}: {e}")

    notas_integradas = "\n".join(notas)

    # Etapa B: síntesis final única
    messages_final = [
        {"role": "system", "content": "Actúa como equipo experto en derecho administrativo y licitaciones sanitarias; redactor técnico-jurídico."},
        {"role": "user", "content": f"""{CRAFT_PROMPT_MAESTRO}

=== NOTAS INTERMEDIAS INTEGRADAS (DEDUPE Y TRAZABILIDAD) ===
{notas_integradas}

👉 Usa ÚNICAMENTE estas notas para elaborar el **informe final único** (sin repetir encabezados por fragmento, sin meta-comentarios). 
👉 Devuelve SOLO el informe final en texto."""}
    ]

    try:
        resp_final = _llamada_openai(messages_final)
        return (resp_final.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Error en la síntesis final: {e}\n\nNotas intermedias:\n{notas_integradas}"

# ============================================================
# Chat IA (gpt-5)
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
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Actuás como un asistente experto en análisis de pliegos de licitación y soporte de plataformas digitales."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=1200
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Error al generar respuesta: {e}"

# ============================================================
# Generación de PDF
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
        test_line = (linea_actual + " " + palabra) if linea_actual else palabra
        if canvas_obj.stringWidth(test_line, canvas_obj._fontname, canvas_obj._fontsize) <= max_width:
            linea_actual = test_line
        else:
            if linea_actual:
                lineas.append(linea_actual)
            linea_actual = palabra

    if linea_actual:
        lineas.append(linea_actual)

    return lineas
