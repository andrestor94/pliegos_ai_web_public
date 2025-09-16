# backend/utils/openai_client.py
import os
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI

log = logging.getLogger(__name__)

# Cliente único
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Modelos por defecto (podés cambiarlos en Render -> Environment)
RESPONSES_MODEL = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-mini")
EMBED_MODEL      = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

def _responses_text(resp) -> str:
    """
    Extrae texto de la Responses API sin importar el shape.
    """
    # SDK nuevo: resp.output_text ya concatena todo
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt.strip()

    # Fallback por si cambia el shape
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "text":
                        parts.append(c.text or "")
        return "".join(parts).strip()
    except Exception:
        return ""

def embed_text(text: str, model: Optional[str] = None) -> List[float]:
    """
    Devuelve el embedding (list[float]) del texto.
    """
    m = model or EMBED_MODEL
    r = _client.embeddings.create(model=m, input=text)
    return r.data[0].embedding

def analizar_con_openai(
    prompt_or_messages: Any,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1500,
) -> str:
    """
    Llama a **Responses API**. Acepta un string o una lista de mensajes [{"role": "...", "content": "..."}].
    """
    m = model or RESPONSES_MODEL

    # normalizamos a mensajes (Responses admite directamente messages)
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages

    resp = _client.responses.create(
        model=m,
        input=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    out = _responses_text(resp)
    log.info("[OPENAI] responses.create model=%s out_tokens=%s", m, getattr(resp, "usage", None))
    return out or ""

# --------- Compatibilidad para el chat del topbar ----------
def responder_chat_openai(
    mensaje: str,
    contexto: Optional[str] = None,
    usuario: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 800,
) -> str:
    """
    Chat simple vía **Responses API** (reemplaza cualquier uso previo de Chat Completions).
    """
    m = model or RESPONSES_MODEL

    system = (
        "Sos un asistente de Suizo Argentina. Respondé corto, claro y en español. "
        "Si te preguntan por un pliego, recordá que el análisis se hace desde la página."
    )
    if usuario:
        system += f" Usuario actual: {usuario}."

    messages = [{"role": "system", "content": system}]
    if contexto:
        messages.append({"role": "user", "content": f"Contexto: {contexto}"})
    messages.append({"role": "user", "content": mensaje})

    resp = _client.responses.create(
        model=m,
        input=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    out = _responses_text(resp)
    log.info("[OPENAI] chat via responses.create model=%s", m)
    return out or "No pude generar respuesta."
