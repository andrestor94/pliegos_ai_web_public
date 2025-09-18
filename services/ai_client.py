# backend/services/ai_client.py
"""
Cliente unificado y robusto para llamadas a LLM usando el SDK oficial de OpenAI.
Prioriza la Responses API; si no está disponible en el runtime/SDK, cae a Chat Completions.
Incluye:
- Descubrimiento flexible de credenciales y base_url (OPENAI_API_KEY / OPENAI_API_KEY_1 / OPENAI_API_BASE).
- Compatibilidad opcional con Azure OpenAI si existen AZURE_OPENAI_* (sin romper si el SDK no lo trae).
- Retries exponenciales en rate limit/errores transitorios.
- Límite de tokens configurable por env (OPENAI_MAX_OUTPUT_TOKENS).
- Firma estable: `chat(message, contexto=None, usuario=None, system=None, model=None, ...) -> str`.
"""

from __future__ import annotations

import os
import time
from typing import Optional, List, Dict, Any, Callable

# SDK OpenAI (Responses API y/o Chat Completions)
from openai import OpenAI

# Intentar AzureOpenAI si el entorno lo usa (no fallar si no existe)
try:
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - import opcional
    AzureOpenAI = None  # type: ignore

# =======================
# Singleton del cliente
# =======================
_singleton = None


def _pick_api_key() -> str:
    """Elige la primera API key disponible en variables de entorno."""
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY_1")
        or os.getenv("AZURE_OPENAI_API_KEY")  # por si sólo configuraron Azure
        or ""
    )


def _mk_openai_client() -> Any:
    """
    Construye el cliente adecuado:
    - AzureOpenAI si hay endpoint/versión configurados.
    - OpenAI clásico con base_url si se indicó OPENAI_API_BASE.
    """
    # Azure (opcional)
    if AzureOpenAI and (os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_BASE_URL")):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_BASE_URL")
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or _pick_api_key()
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if endpoint and api_key:
            return AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )

    # OpenAI estándar (u otro proveedor compatible vía base_url)
    api_key = _pick_api_key()
    base = os.getenv("OPENAI_API_BASE") or None
    org = os.getenv("OPENAI_ORG") or os.getenv("OPENAI_ORGANIZATION") or None

    if base and org:
        return OpenAI(api_key=api_key, base_url=base, organization=org)
    if base:
        return OpenAI(api_key=api_key, base_url=base)
    if org:
        return OpenAI(api_key=api_key, organization=org)
    return OpenAI(api_key=api_key)


def _get_client():
    global _singleton
    if _singleton is None:
        _singleton = _mk_openai_client()
    return _singleton


# =======================
# Helpers de extracción
# =======================
def _resp_to_text(resp) -> str:
    """
    Extrae texto de la Responses API de forma tolerante a cambios del SDK.
    """
    # SDKs recientes
    try:
        txt = (resp.output_text or "").strip()
        if txt:
            return txt
    except Exception:
        pass

    # Fallback defensivo: recorrer output -> items -> content -> text
    parts: List[str] = []
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") in ("output_text", "text"):
                        parts.append((getattr(c, "text", "") or "").strip())
    except Exception:
        pass
    return "".join(parts).strip()


def _chat_to_text(resp) -> str:
    """
    Extrae texto de Chat Completions (fallback).
    """
    try:
        choice = (resp.choices or [None])[0]
        msg = getattr(choice, "message", None)
        if msg and getattr(msg, "content", None):
            return (msg.content or "").strip()
    except Exception:
        pass
    return ""


def _supports_responses_api(client) -> bool:
    """
    Detecta si el cliente soporta Responses API (y si el entorno no la bloquea).
    """
    try:
        # cliente moderno expone .responses
        return hasattr(client, "responses") and callable(getattr(client.responses, "create", None))
    except Exception:
        return False


# =======================
# Retries / Backoff
# =======================
def _is_retryable_error(err: Exception) -> bool:
    """
    Heurística simple: errores 429/5xx/timeout son reintentos.
    No dependemos de clases específicas para no romper con SDKs distintos.
    """
    s = f"{type(err).__name__}: {err}".lower()
    return (
        "rate" in s
        or "429" in s
        or "timeout" in s
        or "time out" in s
        or "temporarily" in s
        or "overloaded" in s
        or "5" in s and "server" in s
    )


def _with_retries(func: Callable[[], Any], max_retries: int = 3, base_delay: float = 0.6) -> Any:
    last = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:  # pragma: no cover - defensivo
            last = e
            if attempt >= max_retries or not _is_retryable_error(e):
                raise
            # Exponencial con jitter leve
            sleep_s = base_delay * (2 ** attempt)
            time.sleep(sleep_s)
    # Si llegamos acá, relanzamos el último error
    raise last  # type: ignore[misc]


# =======================
# API pública
# =======================
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
    Wrapper unificado. Intenta **Responses API** y si no, cae a **Chat Completions**.

    Params:
      - message: prompt del usuario.
      - contexto: texto adicional breve (historial resumido).
      - usuario: se envía como metadata si el SDK/endpoint lo permite.
      - system: mensaje de sistema (instrucciones de comportamiento).
      - model: override del modelo (si no se pasa, usa OPENAI_RESPONSES_MODEL o sensible por defecto).
      - temperature / max_output_tokens: ajustes del muestreo.

    Returns:
      str con la respuesta del modelo (sin streaming).
    """
    client = _get_client()

    # Default del modelo:
    # - Responses: OPENAI_RESPONSES_MODEL o "gpt-5-mini"
    # - Chat fallback: OPENAI_CHAT_MODEL o "gpt-4o-mini"
    model_responses = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-5-mini")
    model_chat = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    chosen_model = model or model_responses

    sys = system or (
        "Sos un asistente útil. Respondé en español. "
        "Si no hay información suficiente en el contexto, decilo sin inventar."
    )

    # Construcción de items/mensajes
    items: List[Dict[str, Any]] = [{"role": "system", "content": sys}]
    ctx = (contexto or "").strip()
    if ctx:
        items.append({"role": "user", "content": f"Contexto (historial resumido):\n{ctx}"})
    items.append({"role": "user", "content": message})

    # ===========================================================
    # 1) Responses API (preferida)
    # ===========================================================
    if _supports_responses_api(client):
        kwargs: Dict[str, Any] = {
            "model": chosen_model,
            "input": items,
            "temperature": float(temperature),
        }
        mot = max_output_tokens or os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
        if mot:
            try:
                kwargs["max_output_tokens"] = int(mot)
            except Exception:
                pass
        # metadata opcional
        if usuario:
            try:
                kwargs["metadata"] = {"user": usuario}
            except Exception:
                pass

        def _call():
            try:
                return client.responses.create(**kwargs)
            except TypeError:
                # SDK más viejo sin max_output_tokens/metadata
                kwargs.pop("max_output_tokens", None)
                kwargs.pop("metadata", None)
                return client.responses.create(**kwargs)

        try:
            resp = _with_retries(_call)
            return _resp_to_text(resp)
        except Exception as e:  # pragma: no cover - fallback a Chat
            # Si falla por razón no recuperable, intentamos Chat Completions
            pass

    # ===========================================================
    # 2) Fallback: Chat Completions
    # ===========================================================
    # Normalizamos a formato chat.completions (system + user messages)
    chat_msgs: List[Dict[str, str]] = []
    # Convertimos 'items' al formato clásico
    for m in items:
        role = "user"
        if m.get("role") == "system":
            role = "system"
        elif m.get("role") == "assistant":
            role = "assistant"
        chat_msgs.append({"role": role, "content": str(m.get("content", ""))})

    kwargs_chat: Dict[str, Any] = {
        "model": model or model_chat,
        "messages": chat_msgs,
        "temperature": float(temperature),
    }
    mot = max_output_tokens or os.getenv("OPENAI_MAX_OUTPUT_TOKENS")
    if mot:
        try:
            kwargs_chat["max_tokens"] = int(mot)
        except Exception:
            pass
    if usuario:
        try:
            kwargs_chat["user"] = usuario
        except Exception:
            pass

    def _call_chat():
        return client.chat.completions.create(**kwargs_chat)

    resp = _with_retries(_call_chat)
    text = _chat_to_text(resp)
    return text or "No pude generar una respuesta en este momento."
