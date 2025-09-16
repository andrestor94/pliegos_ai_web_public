# utils/openai_client.py
import os
from typing import Iterable, List, Optional, Dict, Any
from openai import OpenAI

# Usa la API key del entorno (Render / .env)
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Embeddings (por si ya lo usás en otra parte) ===
def embed(texts: Iterable[str], model: Optional[str] = None) -> List[List[float]]:
    mdl = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    resp = _client.embeddings.create(model=mdl, input=list(texts))
    return [d.embedding for d in resp.data]

# === Chat (Responses API) – reemplaza chat.completions ===
def chat(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1600,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """
    messages: [{"role":"system"|"user"|"assistant", "content":"..."}]
    Devuelve SOLO el texto de salida.
    """
    mdl = model or os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-mini")
    kwargs: Dict[str, Any] = {}
    if response_format:
        kwargs["response_format"] = response_format

    resp = _client.responses.create(
        model=mdl,
        input=messages,              # Responses API acepta la lista de mensajes
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        **kwargs,
    )
    # SDK nuevo tiene resp.output_text; dejamos fallback por compatibilidad
    try:
        return (resp.output_text or "").strip()
    except Exception:
        try:
            chunks: List[str] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    # c.text.value en SDKs recientes
                    val = getattr(getattr(c, "text", None), "value", None)
                    if val:
                        chunks.append(val)
            return "\n".join(chunks).strip()
        except Exception:
            return str(resp)
