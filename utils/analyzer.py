# utils/analyzer.py
import os
from typing import Literal
from .openai_client import chat  # nuestro wrapper (Responses API)

OutType = Literal["resumen", "exhaustivo", "tabla_requisitos"]
Lang = Literal["es", "en"]

def analizar_y_generar_informe(
    texto_pliego: str,
    out_type: OutType = "exhaustivo",
    lang: Lang = "es",
) -> str:
    """
    Recibe TODO el texto consolidado del/los pliegos y devuelve un informe HTML/texto.
    La usa main.py dentro del endpoint de análisis.
    """
    if not (texto_pliego or "").strip():
        return "No se encontró texto para analizar."

    # Modelo económico por defecto (podés sobreescribir en Render)
    model = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-mini")

    if out_type == "tabla_requisitos":
        objetivo = (
            "Genera SOLO una tabla en Markdown con columnas: "
            "Ítem | Requisito | Referencia (pág./anexo) | Observaciones. "
            "Sin texto extra antes ni después."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1400"))
    elif out_type == "resumen":
        objetivo = (
            "Redacta un **resumen ejecutivo** claro (~400-600 palabras) con: "
            "1) Alcance; 2) Fechas/Plazos; 3) Documentación exigida; "
            "4) Criterios de evaluación; 5) Riesgos/alertas; 6) Próximos pasos."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1600"))
    else:  # exhaustivo
        objetivo = (
            "Elabora un **informe exhaustivo** (secciones con subtítulos) que incluya: "
            "Resumen ejecutivo; Requisitos obligatorios; Criterios y ponderaciones; "
            "Cronograma y plazos; Condiciones comerciales/legales; Exclusiones/anexos; "
            "Riesgos y no conformidades; Recomendaciones."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2600"))

    system = (
        "Sos un analista experto en pliegos. Escribe de forma clara y accionable. "
        "Usá subtítulos y bullets. Si falta info, indicalo sin inventar."
    )
    system += " Responde SIEMPRE en español." if lang == "es" else " Answer in English."

    user = (
        f"{objetivo}\n\n"
        "Texto del pliego (puede incluir varios anexos, ya concatenados):\n"
        "----------------------------------------\n"
        f"{texto_pliego}\n"
        "----------------------------------------\n"
        "No inventes datos; si algo no está en el texto, acláralo."
    )

    out = chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        model=model,
        temperature=0.2,
        max_output_tokens=max_tokens,
    )
    return out.strip()

# Alias por compatibilidad si en algún lugar llaman distinto:
def analyze_documents(texto_pliego: str, out_type: OutType = "exhaustivo", lang: Lang = "es") -> str:
    return analizar_y_generar_informe(texto_pliego, out_type, lang)
