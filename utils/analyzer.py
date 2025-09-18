# utils/analyzer.py
import os
from typing import Literal
from utils.openai_client import analizar_con_openai

OutType = Literal["resumen", "exhaustivo", "tabla_requisitos"]
Lang = Literal["es", "en"]

def analizar_y_generar_informe(
    texto_pliego: str,
    out_type: OutType = "exhaustivo",
    lang: Lang = "es",
) -> str:
    if not (texto_pliego or "").strip():
        return "No se encontrÃ³ texto para analizar."

    if out_type == "tabla_requisitos":
        objetivo = (
            "Genera SOLO una tabla en Markdown con columnas: "
            "Ãtem | Requisito | Referencia (pÃ¡g./anexo) | Observaciones. "
            "Sin texto extra antes ni despuÃ©s."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1400"))
    elif out_type == "resumen":
        objetivo = (
            "Redacta un **resumen ejecutivo** claro (~400-600 palabras) con: "
            "1) Alcance; 2) Fechas/Plazos; 3) DocumentaciÃ³n exigida; "
            "4) Criterios de evaluaciÃ³n; 5) Riesgos/alertas; 6) PrÃ³ximos pasos."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1600"))
    else:  # exhaustivo
        objetivo = (
            "Elabora un **informe exhaustivo** (secciones con subtÃ­tulos) que incluya: "
            "Resumen ejecutivo; Requisitos obligatorios; Criterios y ponderaciones; "
            "Cronograma y plazos; Condiciones comerciales/legales; Exclusiones/anexos; "
            "Riesgos y no conformidades; Recomendaciones."
        )
        max_tokens = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2600"))

    return analizar_con_openai(texto_pliego, objetivo, lang=lang, max_out=max_tokens)

# alias de compatibilidad
def analyze_documents(texto_pliego: str, out_type: OutType = "exhaustivo", lang: Lang = "es") -> str:
    return analizar_y_generar_informe(texto_pliego, out_type, lang)
