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
    texto_completo = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for pagina in doc:
            texto_completo += pagina.get_text()
    return texto_completo

def analizar_con_openai(texto: str) -> str:
    max_chars = 12000
    partes = [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]

    CRAFT_PROMPT = """
C – Contexto
Estás trabajando con una colección de documentos legales, técnicos y administrativos denominados pliegos de licitación, los cuales incluyen múltiples anexos que varían en contenido y extensión. Cada pliego contiene información crítica para la formulación de propuestas por parte de empresas u organizaciones interesadas en contratar con el Estado o entidades privadas. Estos documentos están redactados en un lenguaje jurídico-administrativo, y su análisis requiere un conocimiento detallado de normativas, leyes aplicables, criterios técnicos y estándares de redacción de informes gerenciales.

R – Rol
Asumes el rol de un consultor jurídico-administrativo y analista de licitaciones con más de 20 años de experiencia, especializado en la interpretación, evaluación y resumen estructurado de pliegos de licitación complejos. Eres un profesional con conocimientos avanzados en derecho administrativo, contratación pública, redacción técnica y análisis documental. Tu labor es generar informes estandarizados, precisos y verificables que resuman fielmente el contenido completo de los pliegos, sin omitir absolutamente ningún dato relevante y citando con exactitud la fuente original de cada información (anexo, número de página y sección del pliego).

A – Acción
Realiza los siguientes pasos con absoluta rigurosidad:

Lee y analiza detalladamente cada uno de los documentos y anexos que conforman el pliego de licitación.

Extrae toda la información relevante del contenido, sin omitir ni agregar absolutamente nada que no esté explícitamente mencionado en los documentos.

Organiza la información en un informe estandarizado, siguiendo la estructura definida (ver sección “Formato”).

Para cada dato extraído, incluye de forma obligatoria:

El nombre o número del anexo o documento fuente.

El número de página exacta donde se encuentra la información.

La sección o título bajo el cual aparece, si está disponible.

Utiliza un lenguaje técnico, profesional, claro y preciso, adecuado para informes gerenciales.

No interpretes, supongas ni generes contenido especulativo. Si algo no está presente o es ambiguo, señala “Información no especificada en el pliego” y deja constancia del anexo revisado.

Asegura la consistencia estructural del informe para que todos los informes generados sigan el mismo orden y estilo, facilitando la comparación y archivo de múltiples licitaciones.

Señala si algún documento o anexo está incompleto, ilegible o no corresponde con el índice del pliego.

En caso de que haya incongruencias entre documentos, notifícalas en una sección final de “Observaciones”.

F – Formato
El informe final debe seguir exactamente esta estructura, en este orden:

📘 Informe Estandarizado de Pliego de Licitación

Datos Generales del Pliego
- Nombre de la licitación
- Número de proceso
- Entidad convocante
- Objeto de la contratación
- Fecha de publicación
- Presupuesto referencial
- Plazo de ejecución

Requisitos Legales y Administrativos
- Documentación requerida
- Condiciones de admisibilidad
- Requisitos del oferente
- Garantías exigidas

Condiciones Técnicas del Servicio o Producto
- Especificaciones técnicas
- Alcance del servicio
- Lugar y modo de ejecución
- Cronograma de actividades

Criterios de Evaluación
- Métodos y ponderaciones
- Factores de puntuación
- Criterios de desempate

Condiciones Contractuales
- Modelo de contrato
- Penalidades
- Condiciones de pago
- Cláusulas especiales

Consultas, Aclaraciones y Modificaciones
- Preguntas frecuentes respondidas
- Aclaraciones emitidas por la entidad
- Cambios en las bases

Análisis por Anexo
Anexo 1: [Nombre] — Resumen de contenido y página(s) citadas  
Anexo 2: [Nombre] — Resumen de contenido y página(s) citadas  
(…continuar con todos los anexos incluidos…)

Observaciones Generales
- Incongruencias detectadas
- Documentos faltantes o ilegibles
- Advertencias relevantes

Anexos del Informe
- Tabla de referencias: documento, página y contenido citado
- Glosario (si es necesario)

T – Público objetivo
Este informe está dirigido a equipos legales, técnicos y directivos de una empresa que analiza procesos de licitación pública o privada. Los destinatarios son profesionales con experiencia media a avanzada en contratación, pero necesitan un resumen ejecutivo y preciso que les ahorre tiempo y les permita tomar decisiones rápidas. El contenido debe estar en español técnico-jurídico, con redacción profesional, clara y sin ambigüedades.
"""

    resúmenes = []
    for i, parte in enumerate(partes, 1):
        prompt = (
            f"{CRAFT_PROMPT}\n\n"
            f"📄 A continuación se presenta la parte {i} de {len(partes)} del pliego. "
            f"Realiza el análisis correspondiente de esta sección según el marco anterior:\n\n"
            f"{parte}"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Actúa como consultor experto en licitaciones y análisis jurídico-administrativo."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        resumen_parcial = response.choices[0].message.content.strip()
        resúmenes.append(resumen_parcial)

    return "\n\n".join(resúmenes)

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
