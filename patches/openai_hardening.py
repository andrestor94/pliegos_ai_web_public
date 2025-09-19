# backend/patches/openai_hardening.py
"""
Parche defensivo para normalizar llamadas a Responses API y evitar 400:
- Convierte 'max_tokens' -> 'max_output_tokens'
- Convierte 'messages'  -> 'input'
- Acepta input como str o como lista de dicts {role, content}
"""

import os
import sys

# Permite desactivar el parche poniendo OPENAI_PATCH_RESPONSES=0
if os.getenv("OPENAI_PATCH_RESPONSES", "1") != "1":
    print("[OPENAI_PATCH] Parche desactivado por OPENAI_PATCH_RESPONSES=0")
else:
    try:
        from openai.resources.responses import Responses  # SDK >= 1.0
        _orig_create = Responses.create

        def _patched_create(self, *args, **kwargs):
            # Normalizaciones de parámetros comunes que disparan 400
            if "messages" in kwargs and "input" not in kwargs:
                kwargs["input"] = kwargs.pop("messages")

            # Si alguien pasó 'max_tokens' (propio de chat.completions), lo mapeamos
            if "max_tokens" in kwargs and "max_output_tokens" not in kwargs:
                kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

            # Si pasaron 'input' vacío o desconocido, forzamos string legible
            inp = kwargs.get("input", None)
            if inp is None or (isinstance(inp, str) and not inp.strip()):
                # Evitar body inválido: mete un texto mínimo para no 400 por input vacío
                kwargs["input"] = " "

            # Log cortito para diagnóstico (no invade logs)
            print("[OPENAI_PATCH] responses.create normalizado -> keys:",
                  sorted(list(kwargs.keys())))

            return _orig_create(self, *args, **kwargs)

        # Monkeypatch
        Responses.create = _patched_create  # type: ignore[attr-defined]
        print("[OPENAI_PATCH] Responses.create() parcheado OK")

    except Exception as e:
        # Si el SDK cambia estructura, no rompemos el arranque
        print(f"[OPENAI_PATCH] No se pudo parchear Responses.create(): {e}", file=sys.stderr)
