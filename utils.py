# -*- coding: utf-8 -*-
# utils.py — Parte 1/4 (Base + Ingesta KB)

"""
Módulo utilitario para:
- Extracción de texto (PDF/DOCX/Imágenes + OCR selectivo con OpenAI Vision)
- Normalizaciones y helpers varios
- Pipeline de análisis con modelos OpenAI
- Ingesta de Base de Conocimiento (KB): subir/leer archivos, trocear y embebidos

Notas:
- Se usa lazy-import para evitar ciclos (p.ej. con prompts.py o modelos SQLAlchemy).
- Wrapper de Chat Completions compatible con max_completion_tokens / max_tokens.
- Esta es la PARTE 1/4: Base + funciones de KB. Las demás partes completan el módulo.
"""

from __future__ import annotations

import io
import os
import re
import json
import time
import base64
import shutil
import hashlib
import mimetypes
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Iterator, Any
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from zoneinfo import ZoneInfo

# ========================= Carga de .env =========================
load_dotenv()

# ========================= OpenAI client =========================
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "90"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=OPENAI_TIMEOUT)

# ========================= Embeddings model ======================
EMBEDDINGS_MODEL = (
    os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-large").strip()
    or "text-embedding-3-large"
)

# ========================= Base de Conocimiento (KB) =========================
# Lazy-import de modelos para evitar registrar tablas dos veces en SQLAlchemy.
def _kb_models():
    try:
        from models import (
            KBSource as _KBSource,
            KBFile as _KBFile,
            KBChunk as _KBChunk,
            KBPriority as _KBPriority,
        )
    except Exception:
        # fallback si se ejecuta como paquete
        from .models import (  # type: ignore
            KBSource as _KBSource,
            KBFile as _KBFile,
            KBChunk as _KBChunk,
            KBPriority as _KBPriority,
        )
    return _KBSource, _KBFile, _KBChunk, _KBPriority


def _kb_clean_text(s: str) -> str:
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").split())


def _kb_chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> Iterator[str]:
    if not text:
        return iter(())
    n = len(text)
    i = 0
    while i < n:
        j = min(i + max_chars, n)
        yield text[i:j]
        if j >= n:
            break
        i = max(0, j - overlap)


def _kb_sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _kb_detect_content_type(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".json"):
        return "application/json"
    if fn.endswith((".txt", ".md", ".log", ".csv")):
        return "text/plain"
    # por defecto, texto plano para indexación básica
    return mimetypes.guess_type(filename)[0] or "text/plain"


def _kb_extract_text_from_path(path: str) -> Tuple[str, Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    meta: Dict[str, Any] = {}
    if ext in [".txt", ".md", ".csv", ".log"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(), meta
    if ext == ".json":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            obj = json.load(f)
        meta["json_keys"] = list(obj.keys()) if isinstance(obj, dict) else None
        return json.dumps(obj, ensure_ascii=False), meta
    if ext == ".pdf":
        doc = fitz.open(path)
        texts: List[str] = []
        spans: List[Dict[str, Any]] = []
        for p in range(len(doc)):
            t = (doc[p].get_text("text") or "").strip()
            texts.append(t)
            spans.append({"page": p + 1, "chars": len(t)})
        meta["pages"] = len(doc)
        meta["spans"] = spans
        return "\n".join(texts), meta
    # binario no soportado para texto: indexa sin texto
    return "", {"unsupported": True, "ext": ext}


def _kb_embed(text: str, _client: Optional[OpenAI] = None) -> List[float]:
    cli = _client or client
    resp = cli.embeddings.create(model=EMBEDDINGS_MODEL, input=text)
    return resp.data[0].embedding  # type: ignore


def kb_create_or_get_source(db, name: str, storage_path: str, scope: Dict[str, Any]):
    KBSource, _, _, _ = _kb_models()
    src = db.query(KBSource).filter_by(name=name).first()
    if src:
        changed = False
        if storage_path and src.storage_path != storage_path:
            src.storage_path = storage_path
            changed = True
        if scope and (src.scope or {}) != scope:
            src.scope = scope
            changed = True
        if changed:
            db.add(src)
            db.commit()
            db.refresh(src)
        return src
    os.makedirs(storage_path or ".", exist_ok=True)
    src = KBSource(name=name, storage_path=storage_path, scope=scope or {})
    db.add(src)
    db.commit()
    db.refresh(src)
    return src


def kb_ingest_file(
    db,
    source,
    local_path: str,
    *,
    rubric: str,
    tags: Optional[List[str]] = None,
    _client: Optional[OpenAI] = None,
    chunk_chars: int = 1500,
    overlap: int = 200,
) -> int:
    KBSource, KBFile, KBChunk, _ = _kb_models()
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"No existe el archivo: {local_path}")
    os.makedirs(source.storage_path or ".", exist_ok=True)
    sha = _kb_sha256_path(local_path)
    existing = db.query(KBFile).filter_by(hash_sha256=sha).first()
    if existing:
        return existing.id
    dest = os.path.join(source.storage_path or ".", os.path.basename(local_path))
    if os.path.abspath(dest) != os.path.abspath(local_path):
        shutil.copy2(local_path, dest)
    size = os.path.getsize(dest)
    ctype = _kb_detect_content_type(dest)
    text, meta_extra = _kb_extract_text_from_path(dest)
    kbfile = KBFile(
        source_id=source.id,
        filename=os.path.basename(dest),
        content_type=ctype,
        bytes=size,
        hash_sha256=sha,
        stored_path=dest,
        meta={"rubro": rubric, "tags": (tags or []), **(meta_extra or {})},
    )
    db.add(kbfile)
    db.flush()
    if (text or "").strip():
        text_norm = _kb_clean_text(text)
        ord_idx = 0
        for chunk in _kb_chunk_text(text_norm, max_chars=chunk_chars, overlap=overlap):
            vec = _kb_embed(chunk, _client=_client)
            db.add(
                KBChunk(
                    file_id=kbfile.id,
                    ord=ord_idx,
                    text=chunk,
                    embedding=json.dumps(vec),
                    span={},
                    meta={},
                )
            )
            ord_idx += 1
    db.commit()
    return kbfile.id


def kb_upsert_priority(db, rubric: str, label: str, details: str = "", weight: float = 1.0) -> None:
    _, _, _, KBPriority = _kb_models()
    from sqlalchemy import and_

    row = (
        db.query(KBPriority)
        .filter(and_(KBPriority.rubric == rubric, KBPriority.label == label))
        .first()
    )
    if not row:
        row = KBPriority(rubric=rubric, label=label, details=details or "", weight=weight)
        db.add(row)
    else:
        row.details = details or row.details
        row.weight = weight
    db.commit()
# -*- coding: utf-8 -*-
# utils.py — Parte 2/4 (Prompts + Extracción base)

# ========================= Prompts (lazy import) =========================
_prom = None

def _get_prom():
    """Carga prompts.py bajo demanda, evitando ciclos de import."""
    global _prom
    if _prom is None:
        try:
            import importlib
            _prom = importlib.import_module("prompts")
        except Exception:
            _prom = None
    return _prom

def _sinonimos_text() -> str:
    mod = _get_prom()
    return getattr(mod, "SINONIMOS_CANONICOS", "") if mod else ""

def _craft_prompt_notas_text() -> str:
    mod = _get_prom()
    return getattr(mod, "CRAFT_PROMPT_NOTAS",
                   "Extrae bullets técnicos y concisos con citas literales; cero invenciones.")

def prompt_andres(varios_anexos: bool) -> str:
    """
    Devuelve el prompt maestro usando PROMPT_PARAMETRIZADO + reglas_citas(varios_anexos).
    Si algo falla, vuelve a un fallback mínimo.
    """
    mod = _get_prom()
    if mod and hasattr(mod, "PROMPT_PARAMETRIZADO") and hasattr(mod, "reglas_citas"):
        try:
            return mod.PROMPT_PARAMETRIZADO.format(
                REGLAS_CITAS=mod.reglas_citas(varios_anexos),
                NO_RENGLONES_RULE=getattr(mod, "NO_RENGLONES_RULE", "")
            )
        except Exception:
            try:
                return (mod.PROMPT_PARAMETRIZADO + "\n\n" + mod.reglas_citas(varios_anexos)).strip()
            except Exception:
                pass
    # Fallback ultra-minimal
    return (
        "Elabora un informe técnico-jurídico estructurado con citas literales. "
        "No inventes. Cita como '(p. N)'. Si hay múltiples anexos, usa '(Anexo X, p. N)'."
    )

# ========================= Modelos / Heurísticas =========================
MODEL_ANALISIS   = os.getenv("OPENAI_MODEL_ANALISIS", "gpt-4o-mini")
VISION_MODEL     = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")
MODEL_NOTAS      = os.getenv("OPENAI_MODEL_NOTAS", MODEL_ANALISIS)
MODEL_SINTESIS   = os.getenv("OPENAI_MODEL_SINTESIS", MODEL_ANALISIS)
FAST_FORCE_MODEL = os.getenv("FAST_FORCE_MODEL", "").strip()  # opcional para fast
FALLBACK_MODEL_DEFAULT = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-4o-mini")

MAX_SINGLE_PASS_CHARS       = int(os.getenv("MAX_SINGLE_PASS_CHARS", "120000"))
MAX_SINGLE_PASS_CHARS_MULTI = int(os.getenv("MAX_SINGLE_PASS_CHARS_MULTI", str(MAX_SINGLE_PASS_CHARS)))

CHUNK_SIZE_BASE = int(os.getenv("CHUNK_SIZE", "24000"))
TARGET_PARTS    = int(os.getenv("TARGET_PARTS", "2"))

# Tope "blando" de tokens de salida (el wrapper puede reducir dinámicamente)
MAX_COMPLETION_TOKENS_SALIDA = int(os.getenv("MAX_COMPLETION_TOKENS_SALIDA", "3500"))
TEMPERATURE_ANALISIS         = os.getenv("TEMPERATURE_ANALISIS", "").strip()
ANALISIS_MODO                = os.getenv("ANALISIS_MODO", "").lower().strip()  # "fast" opcional

# Granularidad / anti-copia ligera
RENGLON_DESC_MAX_WORDS = int(os.getenv("RENGLON_DESC_MAX_WORDS", "24"))
ART_SNIPPET_MAX_WORDS  = int(os.getenv("ART_SNIPPET_MAX_WORDS", "18"))

# Concurrencia
ANALISIS_CONCURRENCY = int(os.getenv("ANALISIS_CONCURRENCY", "3"))
NOTAS_MAX_TOKENS     = int(os.getenv("NOTAS_MAX_TOKENS", "1400"))

# OCR
VISION_MAX_PAGES     = int(os.getenv("VISION_MAX_PAGES", "8"))
VISION_DPI           = int(os.getenv("VISION_DPI", "150"))
OCR_TEXT_MIN_CHARS   = int(os.getenv("OCR_TEXT_MIN_CHARS", "120"))
OCR_CONCURRENCY      = int(os.getenv("OCR_CONCURRENCY", "4"))

# Control de paginado en texto nativo
PAGINAR_TEXTO_NATIVO = int(os.getenv("PAGINAR_TEXTO_NATIVO", "1"))

# Calidad/recall
MULTI_FORCE_TWO_STAGE_MIN_CHARS = int(os.getenv("MULTI_FORCE_TWO_STAGE_MIN_CHARS", "45000"))
ENABLE_REGEX_HINTS              = int(os.getenv("ENABLE_REGEX_HINTS", "1"))
HINTS_MAX_CHARS                 = int(os.getenv("HINTS_MAX_CHARS", "12000"))
HINTS_PER_FIELD                 = int(os.getenv("HINTS_PER_FIELD", "8"))
ENABLE_SECOND_PASS_COMPLETION   = int(os.getenv("ENABLE_SECOND_PASS_COMPLETION", "1"))

# Ampliaciones automáticas
EXPAND_SECTIONS_213_216        = int(os.getenv("EXPAND_SECTIONS_213_216", "0"))
MAX_RENGLONES_OUT              = int(os.getenv("MAX_RENGLONES_OUT", "12"))
MAX_ARTICULOS_OUT              = int(os.getenv("MAX_ARTICULOS_OUT", "12"))
FORCE_DETERMINISTIC_213_216    = int(os.getenv("FORCE_DETERMINISTIC_213_216", "0"))

# ====== Gobernanza de longitud / orden de salida ======
STRICT_OUT             = int(os.getenv("STRICT_OUT", "1"))  # 1 = aplicar recortes y orden forzado
MAX_TOTAL_CHARS_OUT    = int(os.getenv("MAX_TOTAL_CHARS_OUT", "16000"))
MAX_LINES_PER_SECTION  = int(os.getenv("MAX_LINES_PER_SECTION", "20"))
MAX_WORDS_PER_BULLET   = int(os.getenv("MAX_WORDS_PER_BULLET", "35"))
SECTION_CHAR_LIMIT     = int(os.getenv("SECTION_CHAR_LIMIT", "2200"))
MAX_WORDS_TOTAL_GUIDE  = int(os.getenv("MAX_WORDS_TOTAL_GUIDE", "1200"))  # guía para el prompt

# ========================= Timers PERF =========================
def _t() -> float:
    return time.perf_counter()

def _log_tiempo(etiqueta: str, t0: float) -> None:
    try:
        dt = time.perf_counter() - t0
        print(f"[PERF] {etiqueta}: {dt:0.2f}s")
    except Exception:
        pass

# ==================== OCR / Raster ====================
def _rasterizar_pagina(page: fitz.Page, dpi: int = VISION_DPI) -> bytes:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")

def _chat_create_safe(**kw):
    """
    Wrapper para compatibilidad:
      - Prefiere max_completion_tokens (modelos nuevos).
      - Si falla, intenta con max_tokens (legacy).
      - Nunca envía temperature=None.
    """
    if kw.get("temperature", None) is None:
        kw.pop("temperature", None)

    tok = kw.pop("max_tokens", kw.pop("max_completion_tokens", None))
    base = dict(kw)

    intents = []
    if tok is not None:
        intents.append({**base, "max_completion_tokens": int(tok)})
        intents.append({**base, "max_tokens": int(tok)})
    else:
        intents.append(base)

    last_err = None
    for payload in intents:
        try:
            return client.chat.completions.create(**payload)
        except Exception as e:
            last_err = e
            continue

    # último intento sin temperature
    payload = dict(intents[0])
    payload.pop("temperature", None)
    return client.chat.completions.create(**payload)

def _ocr_openai_imagen_b64(b64_img: str, mime: str = "image/png") -> str:
    prompt = (
        "Extraé el TEXTO literal de esta imagen escaneada de un pliego. "
        "Conservá títulos, tablas como líneas con separadores, listas y números. No resumas ni interpretes."
    )
    try:
        resp = _chat_create_safe(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}}
                ]
            }],
            max_completion_tokens=900,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[OCR-ERROR] {e}"

def _ocr_selectivo_por_pagina(doc: fitz.Document, max_pages: int) -> str:
    """
    Muestrea páginas distribuidas: usa texto nativo si hay; si no, raster + OCR.
    """
    n = len(doc)
    if n == 0:
        return ""

    to_process = min(n, max_pages)
    if to_process >= n:
        page_idxs = list(range(n))
    else:
        page_idxs = sorted({
            int(round(i * (n - 1) / max(1, to_process - 1)))
            for i in range(to_process)
        })

    resultados_map: Dict[int, str] = {}

    def _proc_page(i: int) -> Tuple[int, str]:
        p = doc.load_page(i)
        txt_nat = (p.get_text() or "").strip()
        if len(txt_nat) >= OCR_TEXT_MIN_CHARS:
            return i, f"[PÁGINA {i+1}]\n{txt_nat}"
        png_bytes = _rasterizar_pagina(p)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        txt = _ocr_openai_imagen_b64(b64, mime="image/png")
        return i, (f"[PÁGINA {i+1}]\n{txt}" if txt else f"[PÁGINA {i+1}] (sin texto OCR)")

    t0 = _t()
    with ThreadPoolExecutor(max_workers=OCR_CONCURRENCY) as ex:
        futs = [ex.submit(_proc_page, i) for i in page_idxs]
        for fut in as_completed(futs):
            try:
                i, s = fut.result()
                resultados_map[i] = s
            except Exception:
                pass

    orden = sorted(resultados_map.keys())
    res = [resultados_map[i] for i in orden]

    if n > to_process:
        res.append(f"\n[AVISO] OCR muestreó {to_process}/{n} páginas distribuidas.")

    _log_tiempo("ocr_selectivo", t0)
    return "\n\n".join([r for r in res if r]).strip()

# ==================== Extracción por tipo de archivo ====================
def _leer_todo(file) -> bytes:
    try:
        file.file.seek(0)
        raw = file.file.read()
    except Exception:
        try:
            raw = file.read()
        except Exception:
            raw = b""
    return raw or b""

def _ext_de_archivo(file) -> str:
    nombre = getattr(file, "filename", "") or ""
    _, ext = os.path.splitext(nombre)
    return (ext or "").lower().strip()

def _mime_guess(file) -> str:
    nombre = getattr(file, "filename", "") or ""
    m, _ = mimetypes.guess_type(nombre)
    return m or ""

def _texto_nativo_etiquetado(doc: fitz.Document) -> str:
    partes: List[str] = []
    for i, p in enumerate(doc, 1):
        t = (p.get_text() or "").strip()
        if t:
            partes.append(f"[PÁGINA {i}]\n{t}")
        else:
            partes.append(f"[PÁGINA {i}] (sin texto)")
    return "\n\n".join(partes).strip()

def extraer_texto_de_pdf(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_pdf_sin_bytes", t0)
        return ""

    try:
        with fitz.open(stream=raw, filetype="pdf") as doc:
            suma = 0
            for p in doc:
                suma += len((p.get_text() or "").strip())

            if suma < 500:
                ocr_t0 = _t()
                ocr_text = _ocr_selectivo_por_pagina(doc, VISION_MAX_PAGES)
                _log_tiempo("ocr_selectivo", ocr_t0)
                _log_tiempo("extraccion_pdf_total", t0)
                return ocr_text

            out = (
                _texto_nativo_etiquetado(doc)
                if PAGINAR_TEXTO_NATIVO
                else "\n".join([(p.get_text() or "") for p in doc])
            )
            _log_tiempo("extraccion_pdf_total", t0)
            return (out or "").strip()
    except Exception:
        try:
            out = raw.decode("utf-8", errors="ignore")
            _log_tiempo("extraccion_pdf_decode", t0)
            return out
        except Exception:
            _log_tiempo("extraccion_pdf_error", t0)
            return ""

def extraer_texto_de_docx(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_docx_sin_bytes", t0)
        return ""

    try:
        import docx  # python-docx (lazy)
    except Exception:
        try:
            out = raw.decode("utf-8", errors="ignore")
            _log_tiempo("extraccion_docx_decode", t0)
            return out
        except Exception:
            _log_tiempo("extraccion_docx_error", t0)
            return ""

    try:
        document = docx.Document(io.BytesIO(raw))
        partes: List[str] = []

        for p in document.paragraphs:
            txt = (p.text or "").strip()
            if txt:
                partes.append(txt)

        for tbl in document.tables:
            for row in tbl.rows:
                celdas = [(cell.text or "").strip() for cell in row.cells]
                fila = " | ".join([c for c in celdas if c is not None])
                if fila.strip():
                    partes.append(fila)

        out = "\n".join(partes).strip()
        _log_tiempo("extraccion_docx_total", t0)
        return out
    except Exception:
        try:
            out = raw.decode("utf-8", errors="ignore")
            _log_tiempo("extraccion_docx_decode_fallback", t0)
            return out
        except Exception:
            _log_tiempo("extraccion_docx_error", t0)
            return ""

def extraer_texto_de_imagen(file) -> str:
    t0 = _t()
    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraccion_imagen_sin_bytes", t0)
        return ""

    ext = _ext_de_archivo(file)
    try:
        img_doc = fitz.open(stream=raw, filetype=ext.lstrip(".") or None)
        page = img_doc.load_page(0)
        png = page.get_pixmap(alpha=False).tobytes("png")
        b64 = base64.b64encode(png).decode("utf-8")
        mime = "image/png"
    except Exception:
        b64 = base64.b64encode(raw).decode("utf-8")
        mime = "image/png" if ext == ".png" else "image/jpeg"

    out = _ocr_openai_imagen_b64(b64, mime=mime)
    _log_tiempo("extraccion_imagen_ocr", t0)
    return out

def extraer_texto_universal(file) -> str:
    t0 = _t()
    ext = _ext_de_archivo(file)
    mime = _mime_guess(file)

    if ext == ".pdf" or (mime == "application/pdf"):
        out = extraer_texto_de_pdf(file)
        _log_tiempo("extraer_texto_universal_pdf", t0)
        return out

    if ext == ".docx" or (mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]):
        out = extraer_texto_de_docx(file)
        _log_tiempo("extraer_texto_universal_docx", t0)
        return out

    if ext in [".png", ".jpg", ".jpeg", ".webp"] or (mime.startswith("image/") if mime else False):
        out = extraer_texto_de_imagen(file)
        _log_tiempo("extraer_texto_universal_imagen", t0)
        return out

    raw = _leer_todo(file)
    if not raw:
        _log_tiempo("extraer_texto_universal_sin_bytes", t0)
        return ""

    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = ""

    if ext == ".rtf":
        text = re.sub(r"{\\rtf1.*?\\viewkind4\\uc1", "", text, flags=re.S)
        text = re.sub(r"\\[a-z]+-?\d* ?", "", text)
        text = text.replace("{", "").replace("}", "")

    out = (text or "").strip()
    _log_tiempo("extraer_texto_universal_texto_plano", t0)
    return out

# ==================== Pre-limpieza y helpers ====================
def _limpieza_basica_preanalisis(s: str) -> str:
    s = re.sub(r"\n?P[aá]gina\s+\d+\s+de\s+\d+\s*\n", "\n", s, flags=re.I)
    s = re.sub(r"\n[-_]{3,}\n", "\n", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return (s or "").strip()

_META_PATTERNS = [
    re.compile(r"(?i)\bparte\s+\d+\s+de\s+\d+"),
    re.compile(r"(?i)informe\s+basado\s+en\s+la\s+parte"),
    re.compile(r"(?i)revise\s+las\s+partes\s+restantes"),
    re.compile(r"(?i)informaci[oó]n\s+puede\s+estar\s+incompleta"),
    re.compile(r"(?i)^\s*informe\s+completo\s*$"),
    re.compile(r"(?i)^\s*informe\s+original\s*$"),
]

def _limpiar_meta(texto: str) -> str:
    lineas = []
    for ln in (texto or "").splitlines():
        if any(p.search(ln) for p in _META_PATTERNS):
            continue
        lineas.append(ln)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lineas)).strip()

def _particionar(texto: str, max_chars: int) -> List[str]:
    return [texto[i:i + max_chars] for i in range(0, len(texto or ""), max_chars)]

# Índices y utilidades para páginas y anexos
_ANEXO_RE   = re.compile(r"(?im)^===\s*ANEXO\s+(\d+)")
_PAG_TAG_RE = re.compile(r"\[PÁGINA\s+(\d+)\]")

def _contar_anexos(s: str) -> int:
    return len(_ANEXO_RE.findall(s or ""))

def _index_paginas(s: str) -> List[Tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in _PAG_TAG_RE.finditer(s or "")]

def _pagina_de_indice(indices: List[Tuple[int, int]], pos: int) -> int:
    last = 1
    for i, p in indices:
        if i <= pos:
            last = p
        else:
            break
    return last

def _index_anexos(s: str) -> List[Tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in _ANEXO_RE.finditer(s or "")]

def _anexo_en_pos(indices: List[Tuple[int, int]], pos: int) -> Optional[int]:
    last = None
    for i, a in indices:
        if i <= pos:
            last = a
        else:
            break
    return last
# -*- coding: utf-8 -*-
# utils.py — Parte 3/4 (Normalización + Hints + Secciones)

# --- Parche de alias por compatibilidad (si alguna parte usó __leer_todo) ---
try:
    __leer_todo  # type: ignore
except NameError:
    __leer_todo = _leer_todo  # alias seguro

# =============== Normalización de citas (multi vs único anexo) ===============
_CITA_ANEXO_RE = re.compile(r"\(Anexo\s+([IVXLCDM\d]+)(?:,\s*p\.\s*(\d+))?\)", re.I)

def _normalize_citas_salida(texto: str, varios_anexos: bool) -> str:
    if varios_anexos:
        return texto or ""
    # Si NO hay múltiples anexos, simplifica "(Anexo X, p. N)" => "(p. N)"
    def repl(m):
        pag = m.group(2)
        if pag:
            return f"(p. {pag})"
        return "(Fuente: documento provisto)"
    return _CITA_ANEXO_RE.sub(repl, texto or "")

# ==================== Normalización para PDF (sin markdown) ====================
_HDR_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+)$")
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```.*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_ITALIC_RE = re.compile(r"(\*\*|\*|__|_)(.*?)\1")

def _title_case(s: str) -> str:
    return " ".join(w.capitalize() if w else w for w in re.split(r"(\s+)", s or ""))

def preparar_texto_para_pdf(markdown_text: str) -> str:
    out_lines: List[str] = []
    for raw_ln in (markdown_text or "").splitlines():
        ln = raw_ln.rstrip()

        # Normaliza encabezado de Ficha si el modelo lo numeró como "0)"
        if re.match(r"^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento\b", ln, flags=re.I):
            ln = "Ficha estandarizada del procedimiento (campos estandarizados)"

        if _CODE_FENCE_RE.match(ln):
            continue
        if re.match(r"(?i)^\s*informe\s+completo\s*$", ln):
            continue
        if re.match(r"(?i)^\s*informe\s+original\s*$", ln):
            continue

        m = _HDR_RE.match(ln)
        if m:
            titulo = _title_case(m.group(2).strip(": ").strip())
            out_lines.append(titulo)
            out_lines.append("")  # espacio tras título
            continue

        if _TABLE_SEP_RE.match(ln):
            continue

        if _BULLET_RE.match(ln):
            ln = _BULLET_RE.sub("• ", ln)

        ln = _LINK_RE.sub(lambda mm: f"{mm.group(1)} ({mm.group(2)})", ln)
        ln = _BOLD_ITALIC_RE.sub(lambda mm: mm.group(2), ln)
        out_lines.append(ln)

        if ln.strip().endswith(":"):
            out_lines.append("")  # espacio extra tras línea-título

    texto = "\n".join(out_lines)
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto

# ==================== Hints regex (recall) ====================
def _buscar_candidatos(texto: str, pats: List[str], idx_pag: List[Tuple[int, int]], limit: int) -> List[str]:
    hits: List[str] = []
    for pat in pats:
        for m in re.finditer(pat, texto or "", flags=re.I):
            pos = m.start()
            p = _pagina_de_indice(idx_pag, pos)
            start = max(0, pos - 160)
            end = min(len(texto), pos + 240)
            snippet = (texto[start:end]).replace("\n", " ").strip()
            hits.append(f"- p. {p}: {snippet}")
            if len(hits) >= limit:
                return hits
    return hits[:limit]

def _build_regex_hints(texto: str, limit_per_field: Optional[int] = None, max_chars: Optional[int] = None) -> str:
    if not texto:
        return ""
    if limit_per_field is None:
        limit_per_field = HINTS_PER_FIELD
    if max_chars is None:
        max_chars = HINTS_MAX_CHARS

    idx_pag = _index_paginas(texto)
    secciones: List[str] = []
    for key, meta in DETECTABLE_FIELDS.items():
        hits = _buscar_candidatos(texto, meta["pats"], idx_pag, limit_per_field)
        if hits:
            secciones.append(f"[{meta['label']}]\n" + "\n".join(hits))
        if sum(len(s) for s in secciones) > max_chars:
            break
    return "\n\n".join(secciones[:])

DETECTABLE_FIELDS: Dict[str, Dict] = {
    "mant_oferta": {"label": "Mantenimiento de oferta", "pats": [r"mantenim[ií]ento de la oferta", r"validez de la oferta"]},
    "gar_mant":    {"label": "Garantía de mantenimiento", "pats": [r"garant[ií]a.*manten", r"\b5 ?%"]},
    "gar_cumpl":   {"label": "Garantía de cumplimiento", "pats": [r"garant[ií]a.*cumpl", r"\b10 ?%"]},
    "plazo_ent":   {"label": "Plazo de entrega", "pats": [r"plazo de entrega", r"\b\d{1,3}\s*d[ií]as"]},
    "tipo_cambio": {"label": "Tipo de cambio", "pats": [r"Banco\s+Naci[oó]n", r"tipo de cambio", r"\bBNA\b"]},
    "comision":    {"label": "Comisión de (Pre)?Adjudicación", "pats": [r"Comisi[oó]n.*(pre)?adjudicaci[oó]n"]},
    "muestras":    {"label": "Muestras", "pats": [r"\bmuestras?\b"]},
    "planilla":    {"label": "Planilla de cotización y renglones", "pats": [r"planilla.*cotizaci[oó]n", r"renglones?"]},
    "modalidad":   {"label": "Procedimiento/Modalidad", "pats": [r"licitaci[oó]n\s+(p[úu]blica|privada)", r"contrataci[oó]n\s+directa", r"compra\s+menor", r"subasta", r"modalidad"]},
    "plazo_contr": {"label": "Duración del contrato", "pats": [r"duraci[oó]n del contrato", r"plazo contractual", r"por el t[eé]rmino\s+de\s+\d+", r"\b\d{1,4}\s*d[ií]as"]},
    "prorroga":    {"label": "Prórroga/Ampliación", "pats": [r"pr[oó]rroga", r"ampliaci[oó]n", r"hasta\s+el\s+100%"]},
    "presupuesto": {"label": "Monto / Presupuesto", "pats": [r"presupuesto (estimado|oficial|referencial)", r"monto\s+estimado", r"cr[eé]dito\s+disponible", r"\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?"]},
    "expediente":  {"label": "Expediente / N° proceso", "pats": [r"\bEX-\d{4}-[A-Z0-9-]+", r"\bN[°º]\s*de\s*(proceso|procedimiento|expediente)"]},
    "fechas":      {"label": "Fechas y horas", "pats": [r"\b\d{2}/\d{2}/\d{4}\b", r"\b\d{1,2}:\d{2}\s*(?:hs|h)\b"]},
    "contacto":    {"label": "Contactos y portales", "pats": [r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", r"https?://[^\s)]+|www\.[^\s)]+"]},
    "costo_pliego":{"label": "Costo/valor del pliego", "pats": [r"(?:costo|valor)\s+del\s+pliego", r"adquisici[oó]n\s+del\s+pliego", r"\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?"]},
    "subsanacion": {"label": "Subsanación", "pats": [r"subsanaci[oó]n"]},
    "perf_modif":  {"label": "Perfeccionamiento/Modificaciones", "pats": [r"perfeccionamiento", r"modificaci[oó]n"]},
    "preferencias":{"label": "Preferencias", "pats": [r"preferencias"]},
    "criterios":   {"label": "Criterios de evaluación", "pats": [r"criterios?\s+de\s+evaluaci[oó]n"]},
    "renglones":   {"label": "Renglones y especificaciones", "pats": [r"Rengl[oó]n\s*\d+", r"Especificaciones?\s+t[ée]cnicas?"]},
    "articulos":   {"label": "Artículos citados", "pats": [r"\bArt(?:[íi]culo|\.)\s*\d+[A-Za-z]?\b"]},
    "estado":      {"label": "Estado del trámite", "pats": [r"\bestado\b", r"\bvigente\b", r"\b(adjudicado|desierto|fracasado|cerrado)\b"]},
    "consultas":   {"label": "Inicio y final de consultas", "pats": [r"\bconsultas\b", r"aclaraciones", r"preguntas"]},
    "apertura":    {"label": "Acto de apertura", "pats": [r"acto\s+de\s+apertura", r"\bapertura\b"]},
    "tipo_cotiz":  {"label": "Tipo de cotización", "pats": [r"forma\s+de\s+cotizaci[oó]n", r"tipo\s+de\s+cotizaci[oó]n", r"cotizaci[oó]n\s+por"]},
    "tipo_adj":    {"label": "Tipo de adjudicación", "pats": [r"adjudicaci[oó]n\s+por\s+(rengl[oó]n|lote|total)"]},
    "moneda":      {"label": "Moneda", "pats": [r"\bmoneda\b", r"\bARS\b", r"\bUSD\b"]},
    "obj_gasto":   {"label": "Objeto del gasto", "pats": [r"objeto\s+del\s+gasto", r"partida\s+presupuestaria", r"clasificador"]},
    "ofertas_perm":{"label": "Ofertas permitidas", "pats": [r"m[aá]s\s+de\s+una\s+oferta", r"ofertas?\s+alternativas", r"una\s+sola\s+oferta"]},
}

# ==================== Utilidades de conteo y evidencia ====================
def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text or "", flags=re.I))

_ART_HEAD_RE = re.compile(r"(?im)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.*)$")
_ART_BLOCK_RE = re.compile(
    r"(?ims)^\s*(art(?:[íi]culo|\.?)\s*\d+[a-zº°]?)\s*[-–—:]?\s*(.+?)(?=^\s*art(?:[íi]culo|\.?)\s*\d+[a-zº°]?|\Z)"
)

def _extraer_articulos_con_snippets(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    texto = texto or ""
    idx = _index_paginas(texto)
    idx_ax = _index_anexos(texto)
    res: List[Tuple[str, str, int, Optional[int]]] = []

    for m in _ART_BLOCK_RE.finditer(texto):
        start = m.start()
        p = _pagina_de_indice(idx, start)
        ax = _anexo_en_pos(idx_ax, start)
        rotulo = (m.group(1) or "").strip()
        contenido = (m.group(2) or "").strip()
        snippet = contenido[:200].replace("\n", " ").strip()
        res.append((rotulo, snippet, p, ax))

    if not res:
        for m in _ART_HEAD_RE.finditer(texto):
            start = m.start()
            p = _pagina_de_indice(idx, start)
            ax = _anexo_en_pos(idx_ax, start)
            rotulo = (m.group(1) or "").strip()
            snippet = ((m.group(2) or "").strip())[:200].replace("\n", " ")
            res.append((rotulo, snippet, p, ax))

    return res

_ROW_START_RE = re.compile(r"(?im)^(?:reng(?:l[oó]n)?\.?\s*)(\d{1,4})\b")
_CODE_RE = re.compile(r"\b[A-Z]{1,3}\d{5,8}\b")
_QTY_RE  = re.compile(r"\b\d{1,6}\b")

def _extraer_renglones_y_especificaciones(texto: str) -> List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]]:
    texto = texto or ""
    idx = _index_paginas(texto)
    idx_ax = _index_anexos(texto)
    res: List[Tuple[int, Optional[int], Optional[str], str, int, Optional[int]]] = []

    lines = texto.splitlines()
    pos = 0
    starts: List[Tuple[int, int]] = []
    for i, ln in enumerate(lines):
        m = _ROW_START_RE.match(ln)
        if m:
            starts.append((i, pos))
        pos += len(ln) + 1

    if not starts:
        return res

    starts.append((len(lines), len(texto)))

    for k in range(len(starts) - 1):
        i_line, abs_pos = starts[k]
        j_line, _abs_pos_next = starts[k + 1]
        block_lines = lines[i_line:j_line]
        block_text = " ".join([re.sub(r"\s+", " ", x).strip() for x in block_lines if x.strip()])

        mnum = _ROW_START_RE.match(lines[i_line])
        try:
            num_r = int(mnum.group(1)) if mnum else None
        except Exception:
            num_r = None

        qty = None
        if mnum:
            tail = lines[i_line][mnum.end():]
            mqty = _QTY_RE.search(tail)
            if mqty:
                try:
                    qty = int(mqty.group(0))
                except Exception:
                    qty = None

        mcode = _CODE_RE.search(block_text)
        code = mcode.group(0) if mcode else None

        desc = block_text
        if code:
            desc = re.sub(re.escape(code), "", desc)
        if qty is not None:
            desc = re.sub(rf"\b{qty}\b", "", desc)
        if num_r is not None:
            desc = re.sub(rf"^\s*{num_r}\b", "", desc)
        desc = re.sub(r"\s+", " ", desc).strip()

        p = _pagina_de_indice(idx, abs_pos)
        ax = _anexo_en_pos(idx_ax, abs_pos)

        if num_r is not None:
            res.append((num_r, qty, code, desc, p, ax))

    res.sort(key=lambda t: t[0])
    return res

# ==================== Construcción de secciones (2.13, 2.16, 2.3, 2.15) ====================
_ART_KEYS = re.compile(
    r"(objeto|tipolog|modalidad|mantenim|pr[oó]rroga|oferta|apertura|evaluaci[oó]n|empate|mejora|adjudicaci[oó]n|"
    r"garant[ií]a|entrega|plazo|pago|factura|sanci[oó]n|penalidad|rescis[ií]n|perfeccionamiento|subsanaci[oó]n)",
    re.I
)

def _truncate_words(s: str, max_words: int) -> str:
    try:
        words = re.findall(r"\S+", s or "")
        if len(words) <= max_words:
            return (s or "").strip()
        return " ".join(words[:max_words]).rstrip(",.;:") + "..."
    except Exception:
        return (s or "").strip()

def _build_section_213(texto: str, varios_anexos: bool) -> str:
    rows = _extraer_renglones_y_especificaciones(texto or "")
    if not rows:
        return ""
    rows = rows[:max(1, MAX_RENGLONES_OUT)]
    lines = ["2.13 Planilla de cotización y renglones:"]
    for (num, qty, code, desc, p, ax) in rows:
        desc_corta = _truncate_words(desc, RENGLON_DESC_MAX_WORDS)
        partes = [f"Renglón {num}"]
        if qty is not None:
            partes.append(f"Cantidad: {qty}")
        if code:
            partes.append(f"Código: {code}")
        partes.append(f"Descripción/Especificaciones: {desc_corta}")
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        lines.append(" - " + " — ".join(partes) + f" {cita}")
    return "\n".join(lines)

def _build_section_216(texto: str, varios_anexos: bool) -> str:
    arts = _extraer_articulos_con_snippets(texto or "")
    if not arts:
        return ""
    arts = [(rot, sn, p, ax) for (rot, sn, p, ax) in arts if _ART_KEYS.search(sn or "") or _ART_KEYS.search(rot or "")]
    if not arts:
        return ""
    arts = arts[:max(1, MAX_ARTICULOS_OUT)]
    lines = ["2.16 Catálogo de artículos citados:"]
    for (rot, sn, p, ax) in arts:
        rot_norm = re.sub(r"(?i)art(?:[íi]culo|\.)\s*", "Art. ", rot or "").strip()
        sn = _truncate_words(sn or "", ART_SNIPPET_MAX_WORDS)
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        lines.append(f" - {rot_norm} — {sn} {cita}")
    return "\n".join(lines)

CONTACT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CONTACT_URL_RE   = re.compile(r"(https?://[^\s)]+|www\.[^\s)]+)")

def _extraer_contactos_con_paginas(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    texto = texto or ""
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)
    res: List[Tuple[str, str, int, Optional[int]]] = []

    for m in CONTACT_EMAIL_RE.finditer(texto):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos)
        ax = _anexo_en_pos(idx_ax, pos)
        res.append(("email", m.group(0), p, ax))

    for m in CONTACT_URL_RE.finditer(texto):
        pos = m.start()
        p = _pagina_de_indice(idx_pag, pos)
        ax = _anexo_en_pos(idx_ax, pos)
        v = (m.group(0) or "").rstrip(").,;")
        res.append(("url", v, p, ax))

    seen = set()
    dedup: List[Tuple[str, str, int, Optional[int]]] = []
    for t, v, p, ax in res:
        key = (t, v.lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append((t, v, p, ax))
    return dedup

def _build_section_23(texto: str, varios_anexos: bool) -> str:
    items = _extraer_contactos_con_paginas(texto or "")
    if not items:
        return ""
    out = ["2.3 Contactos y portales:"]
    for (t, v, p, ax) in items:
        etiqueta = "Email" if t == "email" else "URL"
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {etiqueta}: {v} {cita}")
    return "\n".join(out)

NORM_TIPOS = [
    ("Ley",        r"\bLey(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    ("Decreto",    r"\bDecreto(?:\s*N[°º])?\s*([\d\.]{1,7}(?:/\d{2,4})?)\b"),
    ("Resolución", r"\bResoluci[oó]n(?:\s*(?:Ministerial|Conjunta))?\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
    ("Disposición",r"\bDisposici[oó]n\s*(?:N[°º]\s*)?(\d{1,7}(?:/\d{2,4})?)\b"),
]
NORM_PATTS = [(tipo, re.compile(patt, re.I)) for (tipo, patt) in NORM_TIPOS]

def _extraer_normativa(texto: str) -> List[Tuple[str, str, int, Optional[int]]]:
    texto = texto or ""
    idx_pag = _index_paginas(texto)
    idx_ax  = _index_anexos(texto)
    res: List[Tuple[str, str, int, Optional[int]]] = []

    for (tipo, cre) in NORM_PATTS:
        for m in cre.finditer(texto):
            pos = m.start()
            p = _pagina_de_indice(idx_pag, pos)
            ax = _anexo_en_pos(idx_ax, pos)
            numero = (m.group(1) or "").strip()
            res.append((tipo, numero, p, ax))

    seen = set()
    dedup: List[Tuple[str, str, int, Optional[int]]] = []
    for t, n, p, ax in res:
        key = (t.lower(), n)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((t, n, p, ax))
    return dedup

def _build_section_215(texto: str, varios_anexos: bool) -> str:
    normas = _extraer_normativa(texto or "")
    if not normas:
        return ""
    out = ["2.15 Normativa aplicable:"]
    for (t, n, p, ax) in normas:
        cita = f"(Anexo {ax}, p. {p})" if varios_anexos and ax else (f"(p. {p})" if p else "(Fuente: documento provisto)")
        out.append(f" - {t} {n} {cita}")
    return "\n".join(out)

# ====== Reemplazo/inyección de secciones en el informe ======
def _find_section_bounds(text: str, header_regex: str) -> Tuple[int, int]:
    text = text or ""
    m = re.search(header_regex, text, flags=re.I)
    if not m:
        return (-1, -1)
    start = m.start()
    nxt = re.search(r"(?im)^\s*2\.(1[0-9]|[1-9])\s", text[m.end():])
    if not nxt:
        return (start, len(text))
    return (start, m.end() + nxt.start())

def _replace_section(text: str, header_regex: str, replacement: str) -> str:
    s, e = _find_section_bounds(text or "", header_regex)
    if s == -1:
        return (text or "").rstrip() + "\n\n" + (replacement or "").strip() + "\n"
    return (text or "")[:s] + (replacement or "").strip() + "\n" + (text or "")[e:]

def _ampliar_secciones_especificas(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    out = informe or ""

    # Siempre normalizar 2.3 y 2.15 desde extracción determinística (evita ruido/omisiones)
    sec23 = _build_section_23(texto_fuente or "", varios_anexos)
    if sec23:
        out = _replace_section(out, r"(?im)^\s*2\.3\s+Contactos", sec23)

    sec215 = _build_section_215(texto_fuente or "", varios_anexos)
    if sec215:
        out = _replace_section(out, r"(?im)^\s*2\.15\s+Normativa", sec215)

    if not EXPAND_SECTIONS_213_216:
        return out

    sec213 = _build_section_213(texto_fuente or "", varios_anexos)
    if sec213:
        alt213 = sec213.replace("2.13 Planilla de cotización y renglones:", "9) Renglones y planilla de cotización:")
        out = _replace_section(out, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla", alt213)
        out = _replace_section(out, r"(?im)^\s*2\.13\s+Planilla", sec213)

    sec216 = _build_section_216(texto_fuente or "", varios_anexos)
    if sec216:
        out = _replace_section(out, r"(?im)^\s*2\.16\s+Cat[aá]logo\s+de\s+art", sec216)
        # remueve posibles encabezados redundantes generados por el modelo
        out = re.sub(r"(?im)^\s*(ANEXO|Anexo)\s*[-–—]?\s*Cat[aá]logo\s+de\s+art[^\n]*\n?", "", out)

    out = re.sub(r"(?im)^\s*informe\s+original\s*$", "", out)
    return out

# === Post-procesos de Ficha y secciones ===
def _reparar_ficha(informe: str, texto_fuente: str) -> str:
    try:
        total_renglones = len(_extraer_renglones_y_especificaciones(texto_fuente or ""))
    except Exception:
        total_renglones = 0

    if total_renglones:
        informe = re.sub(
            r"(?im)^(\s*•\s*(?:N[uú]mero\s+de\s+rengl[oó]n|Numero\s+de\s+renglon)\s*:\s*)[^\n]*$",
            lambda m: f"{m.group(1)}Total de renglones: {total_renglones}; ver Sección 9 para el detalle completo",
            informe or ""
        )
        informe = re.sub(
            r"(?im)\bTotal de renglones:\s*N\b",
            f"Total de renglones: {total_renglones}",
            informe or ""
        )

    informe = re.sub(
        r"(?im)^(\s*•\s*Monto:\s*)(?:\$+\s*\.{0,3}|[$…]+)\s*(\(.*?\))?\s*$",
        lambda m: f"{m.group(1)}NO ESPECIFICADO{(' ' + m.group(2) if m.group(2) else '')}",
        informe or ""
    )
    return (informe or "")

def _normalizar_encabezados_salida(informe: str) -> str:
    s = informe or ""
    s = re.sub(
        r"(?im)^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento\s*\(campos\s+estandarizados\)\s*$",
        "Ficha estandarizada del procedimiento (campos estandarizados)", s
    )
    s = re.sub(
        r"(?im)^\s*0\)\s*Ficha\s+estandarizada\s+del\s+procedimiento\s*$",
        "Ficha estandarizada del procedimiento", s
    )
    return s.strip()

def _corregir_seccion_9_si_vacia(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    s = informe or ""
    start, end = _find_section_bounds(s, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla")
    needs_fix = False

    if start == -1:
        needs_fix = True
    else:
        bloque = s[start:end]
        if (_count(r"(?im)\bRengl[oó]n\s+\d+", bloque) == 0) or \
           (re.search(r"(?i)\bNO ESPECIFICADO\b", bloque) is not None) or \
           (len(bloque.strip()) < 80):
            needs_fix = True

    if not needs_fix:
        return s

    sec213 = _build_section_213(texto_fuente or "", varios_anexos)
    if not sec213:
        return s

    sec9 = sec213.replace("2.13 Planilla de cotización y renglones:", "9) Renglones y planilla de cotización:")
    if start == -1:
        return (s.rstrip() + "\n\n" + sec9.strip() + "\n")
    else:
        return _replace_section(s, r"(?im)^\s*9\)\s*Renglones\s+y\s+planilla", sec9)

# ====== Política de salida (guía para el modelo + recorte determinístico) ======
def _output_policy_block() -> str:
    return (
        "\n\n=== POLÍTICA DE LONGITUD Y ORDEN ===\n"
        f"- El informe completo NO debe exceder ~{MAX_WORDS_TOTAL_GUIDE} palabras.\n"
        f"- Cada sección 1–12 máximo {MAX_LINES_PER_SECTION} líneas; cada bullet/ítem máximo {MAX_WORDS_PER_BULLET} palabras.\n"
        f"- Evitar repeticiones; un solo dato por línea con su cita.\n"
        "- Prohibido agregar anexos, apéndices, “hallazgos” o texto fuera de la estructura pedida.\n"
        "- Mantener exactamente el orden: Ficha, 1) … 12)."
    )

def _split_informe_por_secciones(s: str) -> List[Tuple[str, str]]:
    """
    Devuelve lista [(header, body)] incluyendo "Ficha..." como primera si existe.
    """
    s = s or ""
    s = re.sub(r"\n{3,}", "\n\n", s)

    blocks: List[Tuple[str, str]] = []
    ficha_re = re.compile(r"(?im)^\s*Ficha\s+estandarizada\s+del\s+procedimiento.*$")
    sec_re = re.compile(r"(?im)^\s*(\d{1,2})\)\s")

    m_ficha = ficha_re.search(s)
    starts: List[Tuple[int, str]] = []

    if m_ficha:
        starts.append((m_ficha.start(), m_ficha.group(0).strip()))

    for m in sec_re.finditer(s):
        line_start = s.rfind("\n", 0, m.start()) + 1
        line_end = s.find("\n", m.start())
        if line_end == -1:
            line_end = len(s)
        header_line = s[line_start:line_end].strip()
        starts.append((line_start, header_line))

    if not starts:
        return [("INFORME", s.strip())]

    starts.sort(key=lambda t: t[0])
    starts.append((len(s), ""))  # centinela

    for i in range(len(starts) - 1):
        hpos, header = starts[i]
        npos, _ = starts[i+1]
        body = s[hpos: npos].split("\n", 1)
        if len(body) == 1:
            header_line, body_content = header, ""
        else:
            header_line, body_content = body[0].strip(), body[1].strip()
        blocks.append((header_line, body_content))

    return blocks

def _recortar_por_politica(header: str, body: str) -> str:
    """
    Aplica recortes determinísticos a un bloque de sección (body), manteniendo el header.
    """
    if not STRICT_OUT:
        return f"{header}\n{body}".strip()

    body = re.sub(r"(?im)^===.*$", "", body)

    lines = [ln for ln in (body or "").splitlines()]
    recortadas: List[str] = []
    max_lines = max(1, MAX_LINES_PER_SECTION)

    for ln in lines:
        ln_clean = ln.strip()
        if not ln_clean:
            recortadas.append("")
            continue
        ln_clean = _truncate_words(ln_clean, MAX_WORDS_PER_BULLET)
        recortadas.append(ln_clean)
        if len([x for x in recortadas if x.strip()]) >= max_lines:
            break

    body_recortado = "\n".join(recortadas).strip()

    if len(body_recortado) > SECTION_CHAR_LIMIT:
        body_recortado = body_recortado[:SECTION_CHAR_LIMIT].rsplit("\n", 1)[0].rstrip() + "\n..."

    return f"{header}\n{body_recortado}".strip()

def _enforce_output_policy(informe: str) -> str:
    """
    Aplica orden y recortes por sección; limita tamaño total.
    """
    s = informe or ""
    blocks = _split_informe_por_secciones(s)

    ficha = [b for b in blocks if re.match(r"(?im)^Ficha\s+estandarizada\s+del\s+procedimiento", b[0])]
    otros = [b for b in blocks if not re.match(r"(?im)^Ficha\s+estandarizada\s+del\s+procedimiento", b[0])]

    def _sec_num(h: str) -> int:
        m = re.match(r"^\s*(\d{1,2})\)\s", h)
        return int(m.group(1)) if m else 99

    otros.sort(key=lambda b: _sec_num(b[0]))
    ordered = ficha + otros

    partes: List[str] = []
    for (h, body) in ordered:
        partes.append(_recortar_por_politica(h, body))

    out = "\n\n".join([p for p in partes if p.strip()])

    if len(out) > MAX_TOTAL_CHARS_OUT:
        out = out[:MAX_TOTAL_CHARS_OUT].rsplit("\n", 1)[0].rstrip() + "\n..."

    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out
# -*- coding: utf-8 -*-
# utils.py — Parte 4/4 (Pipeline de análisis + PDF + helpers finales)

# ==================== Mensajería al modelo ====================
def _mk_temperature() -> Optional[float]:
    try:
        return float(TEMPERATURE_ANALISIS) if TEMPERATURE_ANALISIS else None
    except Exception:
        return None

def _pick_model(final_pass: bool = False) -> str:
    """
    Selecciona el modelo según modo/flags. Para pasadas parciales puede forzar uno 'fast'.
    """
    if not final_pass and (ANALISIS_MODO == "fast") and FAST_FORCE_MODEL:
        return FAST_FORCE_MODEL
    return MODEL_ANALISIS

def _craft_system_prompt(varios_anexos: bool, texto_hints: str = "") -> str:
    base = prompt_andres(varios_anexos)
    bloques = [base]
    sinos = _sinonimos_text()
    if sinos:
        bloques.append(sinos)
    bloques.append(_output_policy_block())
    if (texto_hints or "").strip():
        bloques.append("\n=== HINTS DETECTADOS (útiles para recall) ===\n" + texto_hints.strip())
    return "\n\n".join(b for b in bloques if b).strip()

def _msg_single_block(varios_anexos: bool, texto_fuente: str, texto_hints: str = "", titulo: str = "") -> List[Dict[str, Any]]:
    sys = _craft_system_prompt(varios_anexos, texto_hints=texto_hints)
    user = []
    if titulo:
        user.append(f"TÍTULO/BLOQUE: {titulo}")
    user.append("CONTENIDO A ANALIZAR (texto literal paginado):")
    user.append((texto_fuente or "").strip())
    content = "\n\n".join(user)
    return [{"role": "system", "content": sys}, {"role": "user", "content": content}]

def _call_chat(messages: List[Dict[str, Any]], model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
    payload: Dict[str, Any] = {
        "model": model or _pick_model(final_pass=False),
        "messages": messages,
    }
    temp = _mk_temperature()
    if temp is not None:
        payload["temperature"] = temp
    if max_tokens is not None:
        payload["max_completion_tokens"] = int(max_tokens)

    resp = _chat_create_safe(**payload)
    return (resp.choices[0].message.content or "").strip()

# ==================== Análisis: single/multi-pass ====================
def _resumen_parcial(chunk_text: str, varios_anexos: bool, idx: int, total: int, texto_hints: str = "") -> str:
    """
    Produce un resumen estructurado (mini-informe) del bloque.
    """
    titulo = f"Bloque {idx}/{total}"
    msgs = _msg_single_block(varios_anexos, chunk_text, texto_hints=texto_hints, titulo=titulo)
    out = _call_chat(msgs, model=_pick_model(final_pass=False), max_tokens=NOTAS_MAX_TOKENS)
    return out

def _agregar_y_consolidar(parciales: List[str], varios_anexos: bool, texto_hints: str = "") -> str:
    """
    Funde los parciales en un informe único, aplicando la guía de salida.
    """
    sys = _craft_system_prompt(varios_anexos, texto_hints=texto_hints)
    corpus = "\n\n".join([f"=== PARCIAL {i+1} ===\n{p}" for i, p in enumerate(parciales) if (p or "").strip()])
    user = (
        "Integrá TODOS los parciales anteriores en un ÚNICO informe final, sin repetir texto, "
        "llenando las secciones que falten y citando correctamente (ver reglas). "
        "No agregues anexos ni texto fuera de la estructura pedida."
        "\n\n"
        + corpus
    )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    out = _call_chat(msgs, model=_pick_model(final_pass=True), max_tokens=MAX_COMPLETION_TOKENS_SALIDA)
    return out

def _postproceso_final(informe: str, texto_fuente: str, varios_anexos: bool) -> str:
    """
    Post-procesos determinísticos/seguros, manteniendo citas y estructura.
    """
    s = informe or ""
    s = _normalizar_encabezados_salida(s)
    s = _reparar_ficha(s, texto_fuente or "")
    s = _ampliar_secciones_especificas(s, texto_fuente or "", varios_anexos)
    s = _corregir_seccion_9_si_vacia(s, texto_fuente or "", varios_anexos)
    s = _normalize_citas_salida(s, varios_anexos)
    s = _enforce_output_policy(s)
    s = _limpiar_meta(s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def analizar_y_generar_informe(
    texto_fuente: str,
    *,
    varios_anexos: Optional[bool] = None,
    force_multi: Optional[bool] = None,
) -> str:
    """
    Pipeline principal para obtener el informe desde el texto crudo (paginado).
    - Limpia/normaliza.
    - Opcionalmente genera HINTS para recall.
    - Decide single vs multi-pass y consolida.
    - Aplica post-procesos determinísticos.
    """
    t0 = _t()

    # Limpieza previa
    raw = (texto_fuente or "").strip()
    raw = _limpieza_basica_preanalisis(raw)
    raw = _limpiar_meta(raw)

    # Heurística de anexos si no se especifica
    if varios_anexos is None:
        varios_anexos = (_contar_anexos(raw) > 1)

    # Hints regex (opcionales)
    hints = _build_regex_hints(raw) if ENABLE_REGEX_HINTS else ""

    # Elección single vs multi
    multi = bool(force_multi) or (len(raw) > MAX_SINGLE_PASS_CHARS)

    if not multi:
        msgs = _msg_single_block(varios_anexos, raw, texto_hints=hints)
        borrador = _call_chat(msgs, model=_pick_model(final_pass=True), max_tokens=MAX_COMPLETION_TOKENS_SALIDA)
        final = _postproceso_final(borrador, raw, varios_anexos)
        _log_tiempo("pipeline_single_pass", t0)
        return final

    # Multi-pass (parciales en paralelo)
    partes = _particionar(raw, max_chars=CHUNK_SIZE_BASE)
    parciales: List[str] = []

    def _work(i_chunk: int, total: int, texto: str) -> str:
        return _resumen_parcial(texto, varios_anexos, i_chunk, total, texto_hints=hints)

    t1 = _t()
    with ThreadPoolExecutor(max_workers=max(1, ANALISIS_CONCURRENCY)) as ex:
        futs = [ex.submit(_work, i+1, len(partes), partes[i]) for i in range(len(partes))]
        for fut in as_completed(futs):
            try:
                parciales.append(fut.result())
            except Exception as e:
                parciales.append(f"[ERROR parcial] {e}")

    _log_tiempo("parciales_multi_pass", t1)

    # Consolidación
    t2 = _t()
    borrador = _agregar_y_consolidar(parciales, varios_anexos, texto_hints=hints)
    final = _postproceso_final(borrador, raw, varios_anexos)
    _log_tiempo("consolidacion_multi_pass", t2)
    _log_tiempo("pipeline_multi_pass_total", t0)
    return final

# ==================== Exportar a PDF (ReportLab) ====================
def _wrap_lines(s: str, max_chars: int = 110) -> List[str]:
    """
    Wrap simple por caracteres (word-wrap), suficiente para A4 con márgenes y 10pt.
    """
    out: List[str] = []
    for ln in (s or "").splitlines():
        w = ln.strip("\r")
        if not w:
            out.append("")
            continue
        parts: List[str] = []
        buf: List[str] = []
        for tok in re.findall(r"\S+|\s+", w):
            if tok.isspace():
                if sum(len(x) for x in buf) + len(tok) > max_chars:
                    parts.append("".join(buf).rstrip())
                    buf = []
                else:
                    buf.append(tok)
            else:
                if sum(len(x) for x in buf) + len(tok) > max_chars:
                    if buf:
                        parts.append("".join(buf).rstrip())
                        buf = [tok]
                    else:
                        parts.append(tok[:max_chars])
                        buf = [tok[max_chars:]]
                else:
                    buf.append(tok)
        if buf:
            parts.append("".join(buf).rstrip())
        out.extend(parts if parts else [""])
    return out

def generar_pdf_informe(texto_markdown: str, out_path: Optional[str] = None) -> str:
    """
    Crea un PDF simple y legible desde el texto del informe.
    - Convierte markdown light -> texto plano con bullets.
    - Usa márgenes y salto de página autom.
    Devuelve la ruta del PDF generado.
    """
    # Normaliza a texto plano para PDF
    contenido = preparar_texto_para_pdf(texto_markdown or "")

    if not out_path:
        try:
            tz = ZoneInfo("America/Argentina/Buenos_Aires")
        except Exception:
            tz = None  # usa timezone local por defecto
        ts = datetime.now(tz=tz).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.abspath(f"informe_{ts}.pdf")

    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    left = 20 * mm
    right = 15 * mm
    top = 18 * mm
    bottom = 18 * mm

    usable_w = width - left - right
    x = left
    y = height - top

    c.setTitle("Informe generado")
    c.setAuthor("Pliegos AI")
    c.setSubject("Informe técnico-jurídico")

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(HexColor("#222222"))
    c.drawString(x, y, "Informe técnico-jurídico")
    y -= 8 * mm

    c.setFont("Helvetica", 10)
    c.setFillColor(HexColor("#000000"))

    # Línea separadora
    c.line(x, y, x + usable_w, y)
    y -= 6 * mm

    # Cuerpo
    lines = _wrap_lines(contenido, max_chars=110)

    for ln in lines:
        if y < bottom + 15 * mm:
            c.showPage()
            y = height - top
            c.setFont("Helvetica", 10)
            c.setFillColor(HexColor("#000000"))
        c.drawString(x, y, ln)
        y -= 5 * mm

    c.showPage()
    c.save()
    return out_path

# ==================== Helpers de alto nivel ====================
def generar_informe_y_pdf(
    texto_fuente: str,
    *,
    varios_anexos: Optional[bool] = None,
    force_multi: Optional[bool] = None,
    export_pdf: bool = True,
    ruta_pdf: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Atajo: corre el pipeline de análisis y, opcionalmente, exporta a PDF.
    Devuelve (informe_texto, ruta_pdf | None)
    """
    informe = analizar_y_generar_informe(
        texto_fuente,
        varios_anexos=varios_anexos,
        force_multi=force_multi,
    )
    pdf_path = generar_pdf_informe(informe, out_path=ruta_pdf) if export_pdf else None
    return informe, pdf_path

# ==================== Compatibilidad y alias retro ====================
def responder_chat_openai(
    prompt_or_messages,
    *,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    system: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
    **kwargs,
) -> str:
    """
    Wrapper legacy para chat simple.
    Acepta string (prompt) o lista de mensajes estilo OpenAI.
    Ignora kwargs extra y usa _chat_create_safe para compatibilidad de tokens.
    """
    # Normaliza mensajes
    if isinstance(prompt_or_messages, str):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt_or_messages})
    else:
        messages = list(prompt_or_messages or [])
        if system:
            # Insertar system al principio si no está
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system}] + messages

    # Modelo y temperatura
    use_model = (model or _pick_model(final_pass=False))
    temp = temperature if (temperature is not None) else _mk_temperature()

    payload: Dict[str, Any] = {"model": use_model, "messages": messages}
    if temp is not None:
        payload["temperature"] = float(temp)
    if max_tokens is not None:
        payload["max_completion_tokens"] = int(max_tokens)
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    resp = _chat_create_safe(**payload)
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

def analizar_con_openai(
    texto_fuente: str,
    *,
    varios_anexos: Optional[bool] = None,
    force_multi: Optional[bool] = None,
    **kwargs,
) -> str:
    """
    Alias legacy hacia analizar_y_generar_informe.
    Ignora kwargs extra (p.ej. de APIs viejas).
    """
    return analizar_y_generar_informe(
        texto_fuente,
        varios_anexos=varios_anexos,
        force_multi=force_multi,
    )

def generar_pdf(informe_texto: str, ruta_pdf: Optional[str] = None) -> str:
    """Alias histórico para exportar a PDF."""
    return generar_pdf_informe(informe_texto, out_path=ruta_pdf)

def analizar_y_pdf(
    texto_fuente: str,
    *,
    varios_anexos: Optional[bool] = None,
    force_multi: Optional[bool] = None,
    ruta_pdf: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Alias cómodo equivalente a generar_informe_y_pdf."""
    return generar_informe_y_pdf(
        texto_fuente,
        varios_anexos=varios_anexos,
        force_multi=force_multi,
        export_pdf=True,
        ruta_pdf=ruta_pdf,
    )

# ==================== Compat extra (plantillas) ====================
def generar_pdf_con_plantilla(
    informe_texto: str,
    *,
    plantilla: Optional[str] = None,
    salida: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Compatibilidad con versiones anteriores.
    Ignora 'plantilla' y usa el generador simple de ReportLab.
    - informe_texto: contenido en markdown-light
    - plantilla: nombre/slug de plantilla (no usado aquí)
    - salida: ruta opcional del PDF a escribir
    """
    return generar_pdf_informe(informe_texto, out_path=salida)

# ==================== __all__ ====================
__all__ = [
    # nuevas
    "analizar_y_generar_informe", "generar_informe_y_pdf", "generar_pdf_informe",
    # compat
    "analizar_con_openai", "analizar_y_pdf", "generar_pdf", "generar_pdf_con_plantilla",
    "responder_chat_openai",
    # helpers útiles
    "extraer_texto_universal", "preparar_texto_para_pdf",
]
# ==================== Compat: analizar_anexos (multi-archivo) ====================

class _FauxUpload:
    """Wrapper simple para paths/bytes -> objeto tipo UploadFile (lo mínimo que usamos)."""
    def __init__(self, *, filename: str, data: bytes):
        self.filename = filename
        self._data = data
        self.file = io.BytesIO(data)
    def read(self) -> bytes:
        return self._data

def _coerce_uploadlike(x) -> Any:
    """
    Devuelve un objeto con .filename y .file/.read para poder pasarlo a extraer_texto_universal.
    Admite:
      - Starlette UploadFile / file-like con .filename
      - str path a archivo
      - dict {"filename":..., "bytes":...} o {"path":...}
      - bytes (se nombra como 'anexo.bin')
    """
    # Tiene .filename y .file/.read -> ya sirve
    if hasattr(x, "filename") and (hasattr(x, "file") or hasattr(x, "read")):
        return x

    # Path en str
    if isinstance(x, str) and os.path.isfile(x):
        with open(x, "rb") as f:
            data = f.read()
        return _FauxUpload(filename=os.path.basename(x), data=data)

    # Dict con path
    if isinstance(x, dict) and "path" in x and os.path.isfile(str(x["path"])):
        p = str(x["path"])
        with open(p, "rb") as f:
            data = f.read()
        return _FauxUpload(filename=os.path.basename(p), data=data)

    # Dict con bytes
    if isinstance(x, dict) and "bytes" in x:
        fn = x.get("filename") or "anexo.bin"
        data = x["bytes"] if isinstance(x["bytes"], (bytes, bytearray)) else bytes(x["bytes"])
        return _FauxUpload(filename=str(fn), data=data)

    # Bytes sueltos
    if isinstance(x, (bytes, bytearray)):
        return _FauxUpload(filename="anexo.bin", data=bytes(x))

    # Último recurso: string no existente -> vacío
    return _FauxUpload(filename="anexo_desconocido", data=b"")

def analizar_anexos(
    anexos: List[Any],
    *,
    varios_anexos: Optional[bool] = None,
    force_multi: Optional[bool] = None,
    **kwargs,
) -> str:
    """
    Toma múltiples archivos (PDF/DOCX/imagen/texto), extrae su texto y genera UN informe consolidado.
    Compatible con firmas antiguas que usaban `analizar_anexos`.
    """
    anexos = anexos or []
    partes: List[str] = []
    for i, raw in enumerate(anexos, start=1):
        f = _coerce_uploadlike(raw)
        try:
            texto = extraer_texto_universal(f)
        except Exception as e:
            texto = f"[ERROR al extraer Anexo {i}: {e}]"
        nombre = getattr(f, "filename", f"Anexo_{i}")
        nombre = (nombre or f"Anexo_{i}").strip()
        partes.append(f"=== ANEXO {i} — {nombre}\n{texto}\n")

    corpus = "\n\n".join(partes).strip()
    # Fuerza el modo multi-anexo para que el prompt cite como (Anexo X, p. N)
    return analizar_y_generar_informe(
        corpus,
        varios_anexos=True if varios_anexos is None else bool(varios_anexos),
        force_multi=force_multi,
    )

def analizar_anexos_y_pdf(
    anexos: List[Any],
    *,
    varios_anexos: Optional[bool] = None,
    force_multi: Optional[bool] = None,
    ruta_pdf: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Igual que analizar_anexos pero además exporta a PDF.
    """
    texto = analizar_anexos(
        anexos,
        varios_anexos=varios_anexos,
        force_multi=force_multi,
    )
    pdf_path = generar_pdf_informe(texto, out_path=ruta_pdf)
    return texto, pdf_path

# Exportar símbolos para imports antiguos
try:
    __all__.extend(["analizar_anexos", "analizar_anexos_y_pdf"])  # type: ignore
except Exception:
    pass
# ========= RAG — ORM + sesión KB + OpenAI helpers =========
import os, json, math, hashlib, time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, List, Tuple

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# Usa la misma BD que el resto del sistema
try:
    from database import DB_PATH as _USERS_DB_PATH  # e.g. "usuarios.db"
    _KB_DB_URL = f"sqlite:///{_USERS_DB_PATH}"
except Exception:
    _KB_DB_URL = "sqlite:///usuarios.db"

# ORM local de KB (tablas propias, sin colisionar con otras)
KBBase = declarative_base()

class KBSource(KBBase):
    __tablename__ = "kb_sources"
    id = Column(Integer, primary_key=True)
    slug = Column(String(200), unique=True, index=True, nullable=False)
    name = Column(String(250), nullable=False)
    meta_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    files = relationship("KBFile", back_populates="source")

class KBFile(KBBase):
    __tablename__ = "kb_files"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("kb_sources.id"), nullable=False, index=True)
    path = Column(Text, nullable=False)
    size = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source = relationship("KBSource", back_populates="files")
    chunks = relationship("KBChunk", back_populates="file")

class KBChunk(KBBase):
    __tablename__ = "kb_chunks"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, index=True, nullable=False)
    file_id = Column(Integer, ForeignKey("kb_files.id"), index=True, nullable=False)
    ordinal = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    # guardamos embedding como JSON para portabilidad
    embedding_json = Column(Text, nullable=False)
    n_chars = Column(Integer, default=0)
    md5 = Column(String(64), index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    file = relationship("KBFile", back_populates="chunks")

Index("idx_kbchunk_src_ord", KBChunk.source_id, KBChunk.ordinal)

class KBPriority(KBBase):
    __tablename__ = "kb_priorities"
    id = Column(Integer, primary_key=True)
    term = Column(String(300), index=True, nullable=False)
    weight = Column(Integer, default=1)
    source = Column(String(200), nullable=True)  # slug o None (global)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Engine & Session (mismo archivo SQLite que el sistema)
_kb_engine = create_engine(_KB_DB_URL, future=True)
KBBase.metadata.create_all(bind=_kb_engine)
KBSess = sessionmaker(bind=_kb_engine, autoflush=False, autocommit=False)

@contextmanager
def kb_session():
    """Sesión de KB (usada por main también)."""
    s = KBSess()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

# ---------- OpenAI helpers ----------
_OPENAI_MODEL_CHAT = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
_OPENAI_MODEL_EMB = os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small")

try:
    # SDK nuevo
    from openai import OpenAI
    _oa_client = OpenAI()
except Exception:
    _oa_client = None

def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeddings con OpenAI (lista de vectores)."""
    if not _oa_client:
        raise RuntimeError("OpenAI client no disponible")
    resp = _oa_client.embeddings.create(model=_OPENAI_MODEL_EMB, input=texts)
    return [d.embedding for d in resp.data]

def _chat_completion(messages: List[dict], temperature: float = 0.2, max_tokens: int = 1200) -> str:
    if not _oa_client:
        raise RuntimeError("OpenAI client no disponible")
    r = _oa_client.chat.completions.create(
        model=_OPENAI_MODEL_CHAT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (r.choices[0].message.content or "").strip()
# ========= RAG — Ingesta & Prioridades =========

def _slugify(name: str) -> str:
    import re
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s.strip("-") or "kb"

def kb_create_or_get_source(db, name: str, meta: dict = None):
    """Crea o recupera un rubro/fuente por slug."""
    slug = _slugify(name)
    row = db.query(KBSource).filter(KBSource.slug == slug).first()
    if row:
        return {"id": row.id, "slug": row.slug, "name": row.name}
    row = KBSource(slug=slug, name=name, meta_json=json.dumps(meta or {}))
    db.add(row)
    db.flush()
    return {"id": row.id, "slug": row.slug, "name": row.name}

# ---- extracción de texto ----
def _read_text_from_file(path: str) -> str:
    """Usa tu extractor si existe; si no, hace fallback simples."""
    try:
        # si tenés una función fuerte:
        if 'extraer_texto_universal' in globals() and callable(globals()['extraer_texto_universal']):
            return (globals()['extraer_texto_universal'](path) or "").strip()
    except Exception:
        pass
    # Fallbacks mínimos
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".txt", ".md", ".json", ".yaml", ".yml"):
            return open(path, "r", encoding="utf-8", errors="ignore").read()
        if ext == ".pdf" and 'extraer_texto_de_pdf' in globals():
            return (extraer_texto_de_pdf(path) or "").strip()
    except Exception:
        return ""
    return ""

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
        if i >= n:
            break
    return chunks

def kb_ingest_file(db, source_ref, path: str, meta: dict = None):
    """
    Corta en chunks, genera embeddings y guarda en kb_files/kb_chunks.
    source_ref puede ser dict {'id','slug'} o el slug directamente.
    """
    if isinstance(source_ref, dict):
        src_id = source_ref.get("id")
        src_slug = source_ref.get("slug")
    else:
        src_slug = _slugify(str(source_ref))
        src_row = db.query(KBSource).filter(KBSource.slug == src_slug).first()
        if not src_row:
            src_row = KBSource(slug=src_slug, name=src_slug)
            db.add(src_row)
            db.flush()
        src_id = src_row.id

    text = _read_text_from_file(path)
    if not text:
        return {"ok": False, "reason": "sin_texto"}

    size = 0
    try:
        size = os.path.getsize(path)
    except Exception:
        pass

    f = KBFile(source_id=src_id, path=path, size=size)
    db.add(f)
    db.flush()

    parts = _chunk_text(text)
    if not parts:
        return {"ok": False, "reason": "sin_chunks"}

    vecs = _embed_texts(parts)
    for k, (chunk, emb) in enumerate(zip(parts, vecs), start=1):
        md5 = hashlib.md5(chunk.encode("utf-8", errors="ignore")).hexdigest()
        db.add(
            KBChunk(
                source_id=src_id,
                file_id=f.id,
                ordinal=k,
                text=chunk,
                embedding_json=json.dumps(emb),
                n_chars=len(chunk),
                md5=md5,
            )
        )
    return {"ok": True, "n_chunks": len(parts), "file_id": f.id, "source_id": src_id}

# ---- prioridades ----
def kb_upsert_priority(db, term: str, weight: int = 1, source: Optional[str] = None):
    term = (term or "").strip()
    if not term:
        raise ValueError("term requerido")
    weight = int(weight or 1)
    source_slug = _slugify(source) if source else None

    row = (
        db.query(KBPriority)
        .filter(KBPriority.term == term)
        .filter(KBPriority.source == source_slug)
        .first()
    )
    if row:
        row.weight = weight
        return {"ok": True, "updated": True}
    db.add(KBPriority(term=term, weight=weight, source=source_slug))
    return {"ok": True, "updated": False}

def kb_list_sources():
    with kb_session() as db:
        rows = db.query(KBSource).order_by(KBSource.slug.asc()).all()
        return [{"id": r.id, "slug": r.slug, "name": r.name} for r in rows]

def kb_list_priorities():
    with kb_session() as db:
        rows = db.query(KBPriority).order_by(KBPriority.weight.desc(), KBPriority.term.asc()).all()
        return [{"id": r.id, "term": r.term, "weight": r.weight, "source": r.source} for r in rows]
# ========= RAG — Recuperación + Uso en Chat y Análisis =========
import numpy as _np

def _cosine(a: List[float], b: List[float]) -> float:
    va = _np.array(a, dtype=_np.float32)
    vb = _np.array(b, dtype=_np.float32)
    na = max(1e-8, _np.linalg.norm(va))
    nb = max(1e-8, _np.linalg.norm(vb))
    return float(_np.dot(va, vb) / (na * nb))

def _load_priorities(db, source_slug: Optional[str]):
    q = db.query(KBPriority)
    rows = q.all()
    global_terms = {r.term: r.weight for r in rows if not r.source}
    scoped_terms = {r.term: r.weight for r in rows if r.source == (source_slug or None)}
    return global_terms, scoped_terms

def rag_retrieve(query: str, top_k: int = 8, source: Optional[str] = None) -> List[dict]:
    """Busca los chunks más similares (aplica bonus por prioridades)."""
    if not query or not query.strip():
        return []

    with kb_session() as db:
        q_emb = _embed_texts([query])[0]
        # Para performance básica: limitar a últimos N chunks
        N = int(os.getenv("KB_MAX_CHUNKS_SEARCH", "4000"))
        base = db.query(KBChunk).order_by(KBChunk.id.desc()).limit(N)
        if source:
            base = base.filter(KBChunk.source_id == db.query(KBSource.id).filter(KBSource.slug == _slugify(source)).scalar_subquery())
        rows = list(reversed(base.all()))  # ascendente

        gprio, sprio = _load_priorities(db, _slugify(source) if source else None)

        scored = []
        for r in rows:
            emb = json.loads(r.embedding_json)
            score = _cosine(q_emb, emb)

            # bonus por prioridades si el término aparece en el texto
            txt_lower = (r.text or "").lower()
            bonus = 0.0
            for t, w in gprio.items():
                if t.lower() in txt_lower:
                    bonus += 0.03 * max(1, int(w))
            for t, w in sprio.items():
                if t.lower() in txt_lower:
                    bonus += 0.05 * max(1, int(w))
            score = score + bonus

            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, r in scored[:top_k]:
            src = db.query(KBSource).filter(KBSource.id == r.source_id).first()
            fil = db.query(KBFile).filter(KBFile.id == r.file_id).first()
            out.append({
                "score": round(float(s), 4),
                "source": src.slug if src else "",
                "file": fil.path if fil else "",
                "ordinal": r.ordinal,
                "text": r.text,
            })
        return out

def build_kb_context(query: str, source: Optional[str] = None, top_k: int = 8) -> str:
    """Arma un bloque de contexto listo para el prompt."""
    hits = rag_retrieve(query=query, top_k=top_k, source=source)
    if not hits:
        return "(KB: sin coincidencias relevantes)"
    lines = ["[KB] Extractos relevantes (priorizar estos datos):"]
    for h in hits:
        head = f"Fuente: {h['source'] or '-'} | Archivo: {os.path.basename(h['file'] or '')} | Chunk #{h['ordinal']} | score={h['score']}"
        body = (h["text"] or "").strip()
        lines.append(f"--- {head}\n{body}\n")
    return "\n".join(lines)

# --------- Chat KB-aware ---------
def responder_chat_openai(mensaje: str, contexto_historial: str, usuario_actual: str) -> str:
    """
    Reemplazo KB-aware:
    1) Busca contexto en KB con el mensaje del usuario.
    2) Arma prompt que prioriza KB.
    3) Responde.
    """
    kb_ctx = build_kb_context(query=mensaje, source=None, top_k=8)
    sys = (
        "Sos un asistente técnico. Prioriza SIEMPRE la información de [KB] para responder. "
        "Si algo no está en [KB], podés razonar con lo demás, pero marcá la incertidumbre."
    )
    usr = (
        f"{kb_ctx}\n\n"
        f"[Historial resumido]\n{contexto_historial}\n\n"
        f"[Consulta del usuario]\n{mensaje}\n\n"
        "Instrucciones: contesta citando pasajes de [KB] cuando corresponda (paráfrasis), "
        "y si hay conflicto entre KB y otras señales, gana la KB."
    )
    return _chat_completion(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=0.1,
        max_tokens=900,
    )

# --------- Análisis de pliego KB-aware ---------
def analizar_y_generar_informe(corpus: str, varios_anexos: bool = False) -> str:
    """
    Toma el texto extraído de los anexos y busca soporte en la KB antes de redactar.
    """
    # consulta para RAG: usamos un resumen del corpus (primeros 1500 chars)
    query = (corpus or "").strip()[:1500]
    kb_ctx = build_kb_context(query=query, source=None, top_k=10)

    sys = (
        "Sos un analista de licitaciones. Tu tarea es crear un informe claro y accionable.\n"
        "PRIORIDAD: Corroborar con [KB] y destacar coincidencias/discrepancias."
    )
    usr = (
        f"{kb_ctx}\n\n"
        f"[Anexos recibidos]\n{corpus}\n\n"
        "Redactá el informe con esta estructura:\n"
        "1) Resumen ejecutivo\n"
        "2) Requisitos clave (con referencias a [KB] si aplica)\n"
        "3) Plazos/garantías/forma de presentación\n"
        "4) Riesgos y dudas\n"
        "5) Recomendaciones\n"
        "Usá viñetas breves, títulos y lenguaje simple. Señalá explícitamente si algo no está respaldado por la KB."
    )
    return _chat_completion(
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
        temperature=0.2,
        max_tokens=1400,
    )

