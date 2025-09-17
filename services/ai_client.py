# backend/services/ai_client.py
import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

_singleton = None

def _get_client() -> OpenAI:
    global _singleton
    if _singleton is not None:
        return _singleton
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1") or ""
    base = os.getenv("OPENAI_API_BASE") or None
    _singleton = OpenAI(api_key=api_key, base_url=base) if base else OpenAI(api_key=api_key)
    return _singleton

def _resp_to_text(resp) -> str:
    # Responses API: 'output_text' está disponible en SDKs recientes
    try:
        txt = (resp.output_text or "").strip()
        if txt:
            return txt
    except Exception:
        pass
    # Fallback ultra defensivo por si cambia el SDK
    parts: List[str] = []
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(getattr(c, "text", ""))
    except Exception:
        pass
    return "".join(parts).strip()

def chat(
    message: str,
    contexto: Optional[str] = None,
    usuario: Optional[str] = None,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
) -> str:
    """
    Wrapper unificado sobre **Responses API**. NO usa chat.completions.
    """
    client = _get_client()
    model = model or os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-mini")
    sys = system or "Sos un asistente útil. Responde en español. Si no hay info suficiente, decilo sin inventar."

    items: List[Dict[str, Any]] = [{"role": "system", "content": sys}]
    ctx = (contexto or "").strip()
    if ctx:
        items.append({"role": "user", "content": f"Contexto (historial resumido):\n{ctx}"})
    items.append({"role": "user", "content": message})

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": items,
        "temperature": temperature,
    }
    mot = max_output_tokens or os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
    if mot:
        try:
            kwargs["max_output_tokens"] = int(mot)
        except Exception:
            pass
    # metadata/user es opcional y puede no estar en todos los SDKs
    try:
        if usuario:
            kwargs["metadata"] = {"user": usuario}
    except Exception:
        pass

    try:
        resp = client.responses.create(**kwargs)
    except TypeError:
        # SDK más viejo: sin max_output_tokens/metadata
        kwargs.pop("max_output_tokens", None)
        kwargs.pop("metadata", None)
        resp = client.responses.create(**kwargs)

    return _resp_to_text(resp)