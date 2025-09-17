# utils/openai_client.py
import os
from typing import List, Dict, Any
from openai import OpenAI

# ===== DEBUG de arranque (lo vas a ver en Render) =====
print("[BOOT] utils.openai_client -> RESPONSES API activo")

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    return OpenAI(api_key=api_key, base_url=base_url)

def _model(default: str = "gpt-5-mini") -> str:
    # Permite override por ENV
    return os.getenv("OPENAI_RESPONSES_MODEL", default)

def _max_out(default: int) -> int:
    try:
        return int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", str(default)))
    except Exception:
        return default

def _responses(messages: List[Dict[str, str]], max_output_tokens: int) -> str:
    """
    Llama SIEMPRE a Responses API (nada de chat.completions).
    """
    model = _model()
    print(f"[OPENAI] Responses API model={model} max_out={max_output_tokens}")  # <- traza clara
    client = _client()
    # Flatten de mensajes a un solo prompt (system + user concatenado) — sencillo/robusto
    sys = "\n".join(m["content"] for m in messages if m["role"] == "system")
    usr = "\n".join(m["content"] for m in messages if m["role"] == "user")
    prompt = (sys + "\n\n" + usr).strip()
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
        max_output_tokens=max_output_tokens,
    )
    # salida segura
    if resp.output_text:
        return resp.output_text
    try:
        return resp.output[0].content[0].text
    except Exception:
        return ""

# ------------- API pública usada por main.py -------------
def responder_chat_openai(mensaje: str, contexto: str = "") -> str:
    messages = [
        {"role": "system", "content": (contexto or "Sos un asistente experto. Responde en español.")},
        {"role": "user", "content": mensaje},
    ]
    return _responses(messages, max_output_tokens=_max_out(800))

def analizar_con_openai(texto: str, objetivo: str, lang: str = "es", max_out: int = 2200) -> str:
    sys = (
        "Sos un analista experto en pliegos. Escribe claro, con subtítulos y bullets. "
        "Si falta info, indicalo sin inventar."
    )
    sys += " Responde SIEMPRE en español." if lang == "es" else " Answer in English."
    user = f"{objetivo}\n\n--- TEXTO DEL PLIEGO ---\n{texto}\n------------------------"
    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
    return _responses(messages, max_output_tokens=_max_out(max_out))

# (opcional) embeddings pequeño y barato
def embed_text(texto: str) -> list:
    client = _client()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    out = client.embeddings.create(model=model, input=texto)
    return out.data[0].embedding