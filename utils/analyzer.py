# utils/analyzer.py
import os
from typing import Literal, Optional
from utils.openai_client import analizar_con_openai, embed_text  # embed_text queda por compatibilidad

OutType = Literal["resumen", "exhaustivo", "tabla_requisitos"]
Lang = Literal["es", "en"]

# Límites de entrada (aprox por caracteres) para evitar desbordar el contexto
MAX_CHARS = int(os.getenv("ANALYSIS_MAX_CHARS", "160000"))     # total aprox
TAIL_CHARS = int(os.getenv("ANALYSIS_TAIL_CHARS", "35000"))    # porción final que preservamos si hay recorte


def _trim_long_text(txt: str) -> tuple[str, bool]:
    """
    Si el texto supera MAX_CHARS, preserva el comienzo y el final.
    Devuelve (texto_posiblemente_recortado, fue_recortado?)
    """
    txt = (txt or "").strip()
    if len(txt) <= MAX_CHARS:
        return txt, False

    head = MAX_CHARS - TAIL_CHARS - 1000  # un pequeño margen para separadores
    head = max(head, 0)
    trimmed = (
        txt[:head]
        + "\n\n[... CONTENIDO OMITIDO POR LARGO ...]\n\n"
        + txt[-TAIL_CHARS:]
    )
    return trimmed, True


def _build_prompt(texto_pliego: str, objetivo: str, lang: Lang, fue_truncado: bool) -> str:
    # “System” + “User” combinados en un único input (Responses API)
    sys = (
        "Sos un analista experto en licitaciones y pliegos. Escribís claro, estructurado y accionable; "
        "usás subtítulos y viñetas cuando corresponde. Si falta información en el texto, lo indicás sin inventar."
    )
    sys += " Responde SIEMPRE en español." if lang == "es" else " Answer in English."

    aviso = "\n\nNota: El texto fue truncado por tamaño; evita conjeturas sobre lo omitido." if fue_truncado else ""

    user = (
        f"{objetivo}{aviso}\n\n"
        "=== TEXTO DEL PLIEGO (puede incluir anexos ya concatenados) ===\n"
        f"{texto_pliego}\n"
        "=== FIN TEXTO DEL PLIEGO ===\n\n"
        "No inventes datos; si algo no está en el texto, acláralo."
    )

    # Un único “prompt” para Responses API
    prompt = f"[SYSTEM]\n{sys}\n\n[USER]\n{user}"
    return prompt


def analizar_y_generar_informe(
    texto_pliego: str,
    out_type: OutType = "exhaustivo",
    lang: Lang = "es",
) -> str:
    """
    Recibe TODO el texto consolidado del/los pliegos y devuelve un informe (Markdown/HTML).
    La usa main.py dentro del endpoint de análisis.
    """
    if not (texto_pliego or "").strip():
        return "No se encontró texto para analizar."

    # Modelo económico por defecto (podés sobreescribir en Render)
    model: Optional[str] = (
        os.getenv("OPENAI_MODEL_TEXT")               # recomendado
        or os.getenv("OPENAI_RESPONSES_MODEL")       # compat antigua
        or "gpt-5-mini"
    )

    # Objetivo por tipo + tope de salida
    if out_type == "tabla_requisitos":
        objetivo = (
            "Genera SOLO una tabla en Markdown con columnas EXACTAS: "
            "Ítem | Requisito | Referencia (pág./anexo) | Observaciones. "
            "No agregues texto antes o después. No uses código triple backticks."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1400"))
    elif out_type == "resumen":
        objetivo = (
            "Redacta un **resumen ejecutivo** claro (~400–600 palabras) con secciones: "
            "1) Alcance; 2) Fechas/Plazos; 3) Documentación exigida; "
            "4) Criterios de evaluación; 5) Riesgos/alertas; 6) Próximos pasos."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1600"))
    else:  # exhaustivo
        objetivo = (
            "Elabora un **informe exhaustivo** con subtítulos que incluya: "
            "Resumen ejecutivo; Requisitos obligatorios; Criterios y ponderaciones; "
            "Cronograma y plazos; Condiciones comerciales/legales; Exclusiones y anexos; "
            "Riesgos y no conformidades; Recomendaciones prácticas."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2600"))

    # Recorte seguro si el texto es muy largo
    texto_ok, fue_truncado = _trim_long_text(texto_pliego)

    # Construir prompt único para Responses API
    prompt = _build_prompt(texto_ok, objetivo, lang, fue_truncado)

    # Llamada central: SIEMPRE Responses API (definida en utils/openai_client.py)
    salida = analizar_con_openai(
        prompt=prompt,
        model=model,
        temperature=0.2,
        max_output_tokens=max_tokens,
    )

    return (salida or "").strip()


# Alias por compatibilidad si en algún lugar llaman distinto
def analyze_documents(texto_pliego: str, out_type: OutType = "exhaustivo", lang: Lang = "es") -> str:
    return analizar_y_generar_informe(texto_pliego, out_type, lang)
