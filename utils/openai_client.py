import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_with_openai(text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Podés cambiar por "gpt-4" si querés
            messages=[
                {
                    "role": "system",
                    "content": "Sos un asistente experto en análisis de licitaciones públicas. Leé el contenido de los documentos y devolvé un informe organizado y claro con toda la información clave del pliego."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error al consultar OpenAI: {str(e)}"
