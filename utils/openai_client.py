# utils/openai_client.py
import os
from typing import List, Dict, Any
from openai import OpenAI

print("[BOOT] utils.openai_client -> Responses API preferida (con fallback)")

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    # Usar el nombre correcto de la env var (OPENAI_API_BASE)
    base_url = os.getenv("OPENAI_API_BASE") or None
    org = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION") or None
    if base_url and org:
        return OpenAI(api_key=api_key, base_url=base_url, organization=org)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    if org:
        return OpenAI(api_key=api_key, organization=org)
    return OpenAI(api_key=api_key)

def _model(default: str = "gpt-4.1-mini") -> str:
    # Permite override por ENV
    return os.getenv("OPENAI_RESPONSES_MODEL", default)

def _max_out(default: int) -> int:
    try:
        return int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", str(default)))
    except Exception:
        return default

def _extract_responses_text(resp) -> str:
    try:
        txt = (resp.output_text or "").strip()
        if txt:
            return txt
    except Exception:
        pass
    try:
        # Recorrido defensivo del árbol
        parts = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        parts.append((getattr(c, "text", "") or "").strip())
        return "".join(parts).strip()
    except Exception:
        return ""

def _extract_chat_text(resp) -> str:
    try:
        ch = resp.choices[0]
        return (ch.message.content or "").strip()
    except Exception:
        return ""

def _responses(messages: List[Dict[str, str]], max_output_tokens: int) -> str:
    """
    Intenta Responses API con input=[{role,content}, ...].
    Si falla por compatibilidad de parámetros, reintenta sin max_output_tokens.
    Si aún así falla, cae a Chat Completions.
    """
    client = _client()
    model = _model()
    print(f"[OPENAI] Responses API model={model} max_out={max_output_tokens}")

    # Normalizamos a items role-based para Responses
    items = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

    # 1) Responses API (primer intento, con max_output_tokens si el SDK/endpoint lo permite)
    try:
        resp = client.responses.create(
            model=model,
            input=items,
            temperature=0.2,
            max_output_tokens=max_output_tokens,
        )
        txt = _extract_responses_text(resp)
        if txt:
            return txt
    except TypeError:
        # SDK más viejo que no acepta max_output_tokens -> reintentar sin ese argumento
        pass
    except Exception as e:
        # Si es un 400 por argumentos, probamos sin max_output_tokens
        s = f"{type(e).__name__}: {e}".lower()
        if "400" not in s and "bad request" not in s:
            # no parece ser por argumentos -> intentemos igual fallback
            pass

    # 2) Responses API (segundo intento, sin max_output_tokens)
    try:
        resp2 = client.responses.create(
            model=model,
            input=items,
            temperature=0.2,
        )
        txt2 = _extract_responses_text(resp2)
        if txt2:
            return txt2
    except Exception:
        pass

    # 3) Fallback: Chat Completions
    chat_msgs = []
    for m in messages:
        r = m.get("role", "user")
        if r not in ("system", "user", "assistant"):
            r = "user"
        chat_msgs.append({"role": r, "content": m.get("content", "")})

    try:
        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        resp3 = client.chat.completions.create(
            model=chat_model,
            messages=chat_msgs,
            temperature=0.2,
            max_tokens=max_output_tokens,  # si el SDK no lo acepta, ignorará
        )
        txt3 = _extract_chat_text(resp3)
        return txt3 or ""
    except Exception:
        return ""

# -------- API pública usada por el backend --------
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

def embed_text(texto: str) -> list:
    client = _client()
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    out = client.embeddings.create(model=model, input=texto)
    return out.data[0].embedding
