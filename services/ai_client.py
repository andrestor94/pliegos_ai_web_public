# backend/services/ai_client.py
import os, logging
from openai import OpenAI
from openai import NotFoundError, BadRequestError, APIStatusError

log = logging.getLogger("ai")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")       # ID correcto (con guiones)
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed(texts):
    """
    texts: list[str]
    """
    r = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in r.data]

def _chat_once(model, messages, temperature=0.2, max_tokens=2000):
    return client.chat.completions.create(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens
    ).choices[0].message.content

def chat(messages, temperature=0.2, max_tokens=2000):
    """
    Llama al modelo principal y hace fallback automático si hay 400/404.
    También loguea el cuerpo del error para depurar rápido.
    """
    try:
        return _chat_once(OPENAI_CHAT_MODEL, messages, temperature, max_tokens)
    except (NotFoundError, BadRequestError, APIStatusError) as e:
        # logueamos el cuerpo exacto que devuelve la API
        try:
            body = getattr(e, "response", None)
            body = body.json() if body is not None else str(e)
        except Exception:
            body = str(e)
        log.error("OpenAI chat error with %s: %s", OPENAI_CHAT_MODEL, body)

        # fallback
        if OPENAI_FALLBACK_MODEL and OPENAI_FALLBACK_MODEL != OPENAI_CHAT_MODEL:
            log.warning("Falling back to %s", OPENAI_FALLBACK_MODEL)
            return _chat_once(OPENAI_FALLBACK_MODEL, messages, temperature, max_tokens)
        raise
