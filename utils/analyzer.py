import os
from PyPDF2 import PdfReader
from docx import Document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def analyze_document(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = read_pdf(path)
    elif ext in [".docx", ".doc"]:
        text = read_docx(path)
    elif ext == ".txt":
        text = read_txt(path)
    else:
        return {"error": "Formato no soportado"}

    prompt = f"""
Eres un asistente experto en análisis de licitaciones. Extraé de este documento toda la información importante, bien organizada y sin omitir ningún detalle relevante. Mostralo de forma clara y estructurada en español:

{text[:8000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sos un analista de licitaciones experto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1800
        )

        result = response.choices[0].message.content.strip()
        return {"análisis": result}

    except Exception as e:
        return {"error": f"Error al consultar OpenAI:\n\n{str(e)}"}
