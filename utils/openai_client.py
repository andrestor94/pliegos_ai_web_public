# utils/openai_client.py
import os
from typing import List, Dict, Any
from openai import OpenAI

# Flag unificado: si está en 1, NO usamos /v1/responses en este módulo
FORCE_CHAT = os.getenv("OPENAI_FORCE_CHAT", "0") == "1"

print(f"[BOOT] utils.openai_client -> FORCE_CHAT={FORCE_CHAT}")

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    # Nombre correcto de la env var:
    base_url = os.getenv("OPENAI_API_BASE") or None
    org = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION") or None
    if base_url and org:
        return OpenAI(api_key=api_key, base_url=base_url, organization=org)
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    if org:
        return OpenAI(api_key=api_key, organization=org)
    return OpenAI(api_key=api_key)

def _resp_model(default: str = "gpt-4.1-mini") -> str:
    return os.getenv("OPENAI_RESPONSES_MODEL", default)

def _chat_model(default: str = "gpt-4o-mini") -> str:
    return os.getenv("OPENAI_CHAT_MODEL", default)

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
    Si FORCE_CHAT=1 -> usar SIEMPRE Chat Completions (evita 400 de /v1/responses).
    Si no, intentamos Responses y caemos a chat como fallback.
    """
    client = _client()

    # 0) ¿Forzado a chat?
    if FORCE_CHAT:
        chat_model = _chat_model()
        print(f"[OPENAI] Chat-only (FORCE_CHAT=1) model={chat_model} max_tokens={max_output_tokens}")
        try:
            resp = client.chat.completions.create(
                model=chat_model,
                messages=[{"role": m.get("role","user"), "content": m.get("content","")} for m in messages],
                temperature=0.2,
                max_tokens=max_output_tokens,
            )
            return _extract_chat_text(resp)
        except Exception as e:
            print(f"[OPENAI][chat] error: {e}")
            return ""

    # 1) Responses API (si no forzado)
    model = _resp_model()
    print(f"[OPENAI] Responses API model={model} max_out={max_output_tokens}")
    items = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]

    # 1.a) Primer intento con max_output_tokens
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
        # SDK sin ese parámetro
        pass
    except Exception as e:
        s = f"{type(e).__name__}: {e}".lower()
        print(f"[OPENAI][responses] intento1 error: {s}")

    # 1.b) Segundo intento sin max_output_tokens
    try:
        resp2 = client.responses.create(
            model=model,
            input=items,
            temperature=0.2,
        )
        txt2 = _extract_responses_text(resp2)
        if txt2:
            return txt2
    except Exception as e:
        print(f"[OPENAI][responses] intento2 error: {e}")

    # 2) Fallback chat
    chat_model = _chat_model()
    print(f"[OPENAI] Fallback -> chat.completions model={chat_model}")
    try:
        resp3 = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": m.get("role","user"), "content": m.get("content","")} for m in messages],
            temperature=0.2,
            max_tokens=max_output_tokens,
        )
        return _extract_chat_text(resp3)
    except Exception as e:
        print(f"[OPENAI][chat-fallback] error: {e}")
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
