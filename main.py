# =========================
# main.py — PARTE 1 / 6
# (imports, helpers base, app init, WS, utils, + KB bootstrap)
# =========================

import os
import sqlite3
import uuid
import asyncio
import re
import json
import unicodedata  # PATCH: para normalizar en _email_safe
import logging  # PATCH: logging básico para mejor trazabilidad
from functools import partial  # PATCH: lo usaremos con run_in_threadpool (Parte 2)
from importlib import import_module  # import robusto para utils/Kb
from math import ceil
from typing import List, Optional, Dict, Set, Iterable, Tuple
from contextlib import contextmanager, nullcontext  # kb_session fallback
from pathlib import Path  # PATCH: paths robustos
import shutil  # PATCH: copy2 para reubicar PDFs

from fastapi import (
    FastAPI,
    Request,
    Form,
    UploadFile,
    File,
    HTTPException,
    Body,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    status,
    Query,
)
from fastapi.responses import (
    HTMLResponse,
    RedirectResponse,
    FileResponse,
    JSONResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.concurrency import run_in_threadpool

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import or_
from pydantic import BaseModel, EmailStr
from jinja2 import ChoiceLoader, FileSystemLoader  # loader sin cache

# --- Import base de utils (las funciones “no KB”) ---
from utils import (
    extraer_texto_de_pdf,
    analizar_con_openai,
    generar_pdf_con_plantilla,
    responder_chat_openai,
)

# Import del módulo utils para llamadas dinámicas (KB, extractores, etc.)
import utils as U

# ? Fallback robusto para evitar ImportError: utils.analizar_anexos
try:
    from utils import analizar_anexos as _analizar_anexos  # type: ignore
except Exception:
    _analizar_anexos = None

# ==== Ident / email safe ======================================================
# Intentamos importar desde utils si existe; si falla, usamos fallback local.
try:
    from utils.ident import email_safe as _email_safe  # type: ignore
except Exception:
    def _email_safe(email: Optional[str]) -> str:
        """
        Devuelve una versión segura para IDs/rutas a partir de un email.
        - Normaliza acentos
        - Lowercase
        - Reemplaza '@' y '+' por tokens
        - Deja solo [a-z0-9_-], todo lo demás -> '_'
        - Corta a 120 chars por seguridad
        """
        if not email:
            return "anon"
        s = str(email).strip().lower()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        s = s.replace("@", "_at_").replace("+", "_plus_")
        s = re.sub(r"[^a-z0-9_-]+", "_", s)
        return s[:120]

# ==== Helpers de archivos / seguridad =========================================
def _safe_basename(name: str) -> str:
    """Nombre base sin ruta ni extensión, saneado para usar en archivos."""
    if not name:
        return "archivo"
    base = os.path.basename(name)
    base = os.path.splitext(base)[0]
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._-")
    return base or "archivo"

def _get_allowed_ext() -> Set[str]:
    """
    Junta las extensiones permitidas declaradas más abajo (CHAT/INCID/KB).
    Si aún no están definidas (orden de carga), usa un set razonable por defecto.
    """
    acc: Set[str] = set()
    for key in ("CHAT_ALLOWED_EXT", "INCID_ALLOWED_EXT", "KB_ALLOWED_EXT"):
        val = globals().get(key)
        if isinstance(val, (set, list, tuple)):
            acc |= set(val)
    if not acc:
        acc = {
            ".pdf", ".png", ".jpg", ".jpeg", ".webp",
            ".txt", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".pptx",
            ".md", ".json", ".yaml", ".yml"
        }
    return {e.lower() for e in acc}

def _validate_ext(filename: str):
    ext = os.path.splitext(filename or "")[1].lower()
    if not ext:
        raise HTTPException(status_code=400, detail="Archivo sin extensión")
    allowed = _get_allowed_ext()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Extensión no permitida: {ext}")

async def _save_upload_stream(fup: UploadFile, dst_path: str, chunk_size: int = 1024 * 1024) -> int:
    """
    Guarda un UploadFile a disco por streams (asíncrono). Devuelve bytes escritos.
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    written = 0
    with open(dst_path, "wb") as out:
        while True:
            chunk = await fup.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            written += len(chunk)
    return written

# ==== Helpers de actor/req =====================================================
def _actor_info(request: Request) -> Tuple[Optional[int], Optional[str]]:
    """
    Devuelve (actor_user_id, ip) para auditoría.
    actor_user_id viene de la tabla usuarios (columna id).
    """
    email = request.session.get("usuario")
    ip_hdr = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    ip = ip_hdr or (request.client.host if request.client else None)
    try:
        row = obtener_usuario_por_email(email) if email else None
        uid = int(row[0]) if (isinstance(row, (list, tuple)) and len(row) >= 1 and row[0] is not None) else None
    except Exception:
        uid = None
    return uid, ip

# ==== Anexos (puente) ==========================================================
def analizar_anexos(archivos: List[UploadFile]) -> str:
    """
    Síncrona (para run_in_threadpool). Si utils.analizar_anexos existe, delega.
    Si no, extrae texto con utils.extraer_texto_universal y llama al pipeline
    utils.analizar_y_generar_informe, preservando '=== ANEXO N' para citas.
    """
    if callable(_analizar_anexos):
        return _analizar_anexos(archivos)

    textos = []
    for i, a in enumerate(archivos, start=1):
        if not a or not getattr(a, "filename", None):
            continue
        try:
            t = U.extraer_texto_universal(a)
        except Exception as e:
            t = f"[ERROR leyendo {getattr(a, 'filename', '')}: {e}]"
        t = (t or "").strip()
        if t:
            textos.append(f"=== ANEXO {i}\n{t}")

    corpus = "\n\n".join(textos).strip()
    if not corpus:
        return "No se pudo extraer texto de los anexos."

    varios = len(textos) > 1
    try:
        return U.analizar_y_generar_informe(corpus, varios_anexos=varios)
    except Exception as e:
        return f"[Error de análisis] {e}"

from database import (
    DB_PATH,
    inicializar_bd,
    obtener_usuario_por_email,
    agregar_usuario,
    listar_usuarios,
    actualizar_password,
    cambiar_estado_usuario,
    borrar_usuario,
    cambiar_rol,
    buscar_usuarios,
    guardar_en_historial,
    obtener_historial,
    eliminar_del_historial,
    obtener_historial_completo,
    crear_ticket,
    obtener_todos_los_tickets,
    obtener_tickets_por_usuario,
    actualizar_estado_ticket,
    eliminar_ticket,
    obtener_auditoria,
    enviar_mensaje,
    obtener_hilos_para,
    obtener_mensajes_entre,
    marcar_mensajes_leidos,
    contar_no_leidos,
    ocultar_hilo,
    restaurar_hilo,
    guardar_adjunto,
    es_admin,
    iniciar_analisis_historial,
    marcar_valoracion_historial,
    tiene_valoracion_pendiente,
    # ?? importamos para usarlo en la sección Admin (PARTE 4+)
    crear_o_restaurar_usuario,
)
# ORM (audit_logs)
from db_orm import inicializar_bd_orm, SessionLocal, AuditLog

# ---------- KB: init ORM si hay models.py ----------
def _kb_init_orm():
    """
    Garantiza que las tablas de la KB existan si está el módulo models.py
    con declarativos (KBSource/KBFile/KBChunk/KBPriority).
    No detiene la app si no existe.
    """
    try:
        import models as KBM  # debe exponer Base + clases KB*
        # obtener engine desde una sesión viva
        with SessionLocal() as s:
            engine = s.get_bind()
        if engine is not None and hasattr(KBM, "Base"):
            KBM.Base.metadata.create_all(bind=engine)
            print("✓ KB: tablas verificadas/creadas")
    except Exception as e:
        # si no está models.py o falla algo, no frenamos la app
        print("· KB init omitido:", repr(e))

# ================== TZ & helpers ==================
TZ_AR = ZoneInfo("America/Argentina/Buenos_Aires")

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def now_stamp_ar() -> str:
    return datetime.now(TZ_AR).strftime("%Y%m%d%H%M%S")

def iso_utc_to_ar_str(iso_utc: str, fmt: str = "%d/%m/%Y %H:%M") -> str:
    if not iso_utc:
        return ""
    s = str(iso_utc).strip()
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)  # puede ser naive
    except ValueError:
        # fallback "YYYY-MM-DD HH:MM:SS"
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return iso_utc
    # clave: si es naive, asumimos Buenos Aires (NO UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_AR)  # y mostramos en AR
    return dt.astimezone(TZ_AR).strftime(fmt)

# --- Normalizador robusto de datetimes a UTC aware (para comparaciones seguras) ---
def _parse_dt_utc(value) -> Optional[datetime]:
    """
    Acepta datetime o str (con o sin 'Z') y devuelve datetime con tz UTC.
    Si es naive, asumimos que estaba en AR local y la convertimos a UTC.
    """
    if not value:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

    if dt.tzinfo is None:
        # naive → asumimos AR local
        dt = dt.replace(tzinfo=TZ_AR)
    # devolvemos en UTC para comparaciones/orden
    return dt.astimezone(timezone.utc)

# ================== App & Middlewares ==================
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-in-prod")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "session")
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "0") == "1"  # https_only

if SESSION_SECRET == "change-this-in-prod":
    print("⚠️  SESSION_SECRET por defecto: configurá SESSION_SECRET en producción.")

_middlewares: List[Middleware] = [
    # Cookie de sesión más robusta y persistente
    Middleware(
        SessionMiddleware,
        secret_key=SESSION_SECRET,
        same_site="lax",
        max_age=60 * 60 * 24 * 30,  # 30 días
        https_only=SESSION_COOKIE_SECURE,
        session_cookie=SESSION_COOKIE_NAME,
    )
]

# (Opcional) compresión de respuestas
if os.getenv("ENABLE_GZIP", "1") == "1":
    try:
        from starlette.middleware.gzip import GZipMiddleware
        _middlewares.append(Middleware(GZipMiddleware, minimum_size=1024))
    except Exception as _egzip:
        print("· GZip no disponible:", repr(_egzip))

# (Opcional) CORS para frontends externos
if os.getenv("ENABLE_CORS", "0") == "1":
    try:
        from fastapi.middleware.cors import CORSMiddleware
        _origins_env = os.getenv("CORS_ORIGINS", "*")
        _origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
        _middlewares.append(
            Middleware(
                CORSMiddleware,
                allow_origins=_origins if _origins else ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        )
    except Exception as _ecors:
        print("· CORS no disponible:", repr(_ecors))

# (Opcional) Trusted Host para evitar Host header attacks
_trusted = os.getenv("TRUSTED_HOSTS", "").strip()
if _trusted:
    try:
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        _hosts = [h.strip() for h in _trusted.split(",") if h.strip()]
        if _hosts:
            _middlewares.append(Middleware(TrustedHostMiddleware, allowed_hosts=_hosts))
    except Exception as _ehost:
        print("· TrustedHost no disponible:", repr(_ehost))

app = FastAPI(middleware=_middlewares)

# PATCH: Configuración de logging (nivel via env LOG_LEVEL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# ---------- Garantizar tablas de chat si faltan (fix 'no such table: mensajes') ----------
def ensure_chat_tables():
    """Crea tablas de chat si no existen en usuarios.db (robustez en Render)."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            # mensajes
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mensajes(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    de_email TEXT NOT NULL,
                    para_email TEXT NOT NULL,
                    texto TEXT,
                    leido INTEGER NOT NULL DEFAULT 0,
                    fecha TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mensajes_para_leido ON mensajes(para_email, leido)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mensajes_de_para ON mensajes(de_email, para_email)"
            )
            # hilos ocultos
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hilos_ocultos(
                    owner_email TEXT NOT NULL,
                    otro_email TEXT NOT NULL,
                    hidden_at TEXT NOT NULL,
                    PRIMARY KEY(owner_email, otro_email)
                )
                """
            )
            # adjuntos
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mensajes_adjuntos(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mensaje_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    original TEXT,
                    mime TEXT,
                    size INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
    except Exception as e:
        print("· ensure_chat_tables() no pudo crear tablas:", repr(e))
    finally:
        try:
            conn.close()
        except Exception:
            pass

# Inicializa BD SQLite (usuarios, historial, tickets, mensajes, hilos_ocultos, adjuntos)
inicializar_bd()
# Asegurar explícitamente las tablas de chat (por si el módulo de DB venía sin creadoras)
ensure_chat_tables()
# Inicializa ORM (audit_logs)
inicializar_bd_orm()
# Inicializa (si existe) el esquema de la KB en el mismo engine
_kb_init_orm()

# ---------- Bootstrap de admin si DB está vacía ----------
def ensure_default_admin():
    """
    Si no hay usuarios en la DB, crea un admin inicial.
    Variables de entorno:
    - DEFAULT_ADMIN_EMAIL
    - DEFAULT_ADMIN_NAME
    - DEFAULT_ADMIN_PASSWORD (fallback a DEFAULT_NEW_USER_PASSWORD o '1234')
    """
    try:
        usuarios = listar_usuarios()
    except Exception:
        usuarios = []

    if usuarios:
        return

    default_email = (os.getenv("DEFAULT_ADMIN_EMAIL", "admin@suizo.com") or "").lower()
    default_name = os.getenv("DEFAULT_ADMIN_NAME", "Admin")
    default_pwd = os.getenv(
        "DEFAULT_ADMIN_PASSWORD", os.getenv("DEFAULT_NEW_USER_PASSWORD", "1234")
    )

    try:
        if not obtener_usuario_por_email(default_email):
            agregar_usuario(
                nombre=default_name,
                email=default_email,
                password=default_pwd,
                rol="admin",
                actor_user_id=None,
                ip=None,
            )
            print(f"✓ Admin inicial creado: {default_email}")
    except Exception as e:
        print("· ensure_default_admin() error:", repr(e))

ensure_default_admin()
# ---------- fin bootstrap ----------

# ================== Static & PDFs (UNIFICACIÓN DE RUTAS) ==================
os.makedirs("static", exist_ok=True)

# Canon: servimos SIEMPRE desde /opt/render/project/src/generated_pdfs (raíz src)
# Puedes overridear con env: PDF_DIR=/ruta/absoluta
APP_DIR = Path(__file__).resolve().parent                  # .../src/backend
ROOT_DIR = APP_DIR.parent                                  # .../src
PDF_SERVE_DIR = Path(os.getenv("PDF_DIR", ROOT_DIR / "generated_pdfs")).resolve()

# Otros lugares donde podría estar escribiendo utils.generar_pdf_con_plantilla
PDF_CANDIDATE_DIRS = [
    PDF_SERVE_DIR,
    ROOT_DIR / "generated_pdfs",
    APP_DIR / "generated_pdfs",
    Path.cwd() / "generated_pdfs",
    ROOT_DIR / "backend" / "generated_pdfs",
]
for _d in PDF_CANDIDATE_DIRS:
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _pdf_candidates(filename: str) -> List[Path]:
    fn = os.path.basename(filename)
    return [d / fn for d in PDF_CANDIDATE_DIRS]

def _ensure_pdf_in_serve_dir(filename: str) -> Optional[str]:
    """
    Garantiza que <filename> esté en PDF_SERVE_DIR.
    - Si ya está, devuelve su ruta.
    - Si aparece en otro candidato, lo copia a PDF_SERVE_DIR y devuelve ruta destino.
    - Si no existe en ningún lado, devuelve None.
    """
    target = PDF_SERVE_DIR / os.path.basename(filename)
    if target.exists() and target.is_file():
        return str(target)
    for cand in _pdf_candidates(filename):
        if cand.exists() and cand.is_file():
            try:
                shutil.copy2(str(cand), str(target))
            except Exception:
                # fallback simple si copy2 falla
                with open(cand, "rb") as src, open(target, "wb") as dst:
                    dst.write(src.read())
            return str(target)
    return None

app.mount("/static", StaticFiles(directory="static"), name="static")
# Montamos /generated_pdfs apuntando al directorio CANÓNICO
app.mount("/generated_pdfs", StaticFiles(directory=str(PDF_SERVE_DIR)), name="generated_pdfs")

templates = Jinja2Templates(directory="templates")
templates.env.globals["os"] = os

# desactivar cache de Jinja y forzar FileSystemLoader (evita plantillas viejas en Render/CDN)
try:
    templates.env.loader = ChoiceLoader([FileSystemLoader("templates")])
    templates.env.auto_reload = True
    templates.env.cache = {}
except Exception:
    pass

# Filtro Jinja para mostrar UTC como hora local AR
def ar_time(value: str) -> str:
    try:
        return iso_utc_to_ar_str(value)
    except Exception:
        return value

templates.env.filters["ar_time"] = ar_time

# No-cache para HTML (evita que el browser/CDN te muestre UI vieja)
@app.middleware("http")
async def _no_cache_html(request, call_next):
    resp = await call_next(request)
    ctype = (resp.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

# Headers de seguridad básicos
@app.middleware("http")
async def _security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "no-referrer-when-downgrade")
    return resp

# ================== Guardas/Dependencias de auth/roles ==================
def require_auth(request: Request):
    if not request.session.get("usuario"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado"
        )

def require_admin(request: Request):
    email = request.session.get("usuario")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado"
        )
    rol = request.session.get("rol")
    if rol == "admin":
        return
    try:
        if es_admin(email):
            request.session["rol"] = "admin"
            return
    except Exception:
        pass
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Solo admins")

# ================== Preferencias de respuesta (HTML/JSON) ==================
def wants_json(request: Request) -> bool:
    """
    True si el cliente espera JSON (útil para que /incidencias/cerrar|eliminar respondan {ok:true}
    cuando vienen por fetch).
    """
    acc = (request.headers.get("accept") or "").lower()
    xrw = (request.headers.get("x-requested-with") or "").lower()
    return ("application/json" in acc) or (xrw in ("fetch", "xmlhttprequest")) or (
        request.query_params.get("_json") == "1"
    )

def _wants_html(req: Request) -> bool:
    acc = (req.headers.get("accept") or "").lower()
    return "text/html" in acc and "application/json" not in acc

# ================== Alert/WS manager ==================
class ConnectionManager:
    def __init__(self):
        self._by_user: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, email: str):
        await websocket.accept()
        email = (email or "").strip() or "anon"
        self._by_user.setdefault(email, set()).add(websocket)

    def disconnect(self, websocket: WebSocket, email: str):
        try:
            if email in self._by_user and websocket in self._by_user[email]:
                self._by_user[email].remove(websocket)
            if email in self._by_user and not self._by_user[email]:
                del self._by_user[email]
        except Exception:
            pass

    async def send_to_user(self, email: str, payload: dict):
        if not email:
            return
        conns = list(self._by_user.get(email, []))
        dead = []
        for ws in conns:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._by_user.get(email, set()).discard(ws)

    async def broadcast(self, payload: dict):
        for email in list(self._by_user.keys()):
            await self.send_to_user(email, payload)

manager = ConnectionManager()

def _get_ws_email(websocket: WebSocket) -> str:
    email = None
    try:
        email = websocket.scope.get("session", {}).get("usuario")
    except Exception:
        email = None
    if not email:
        email = websocket.query_params.get("email")
    return email or "anon"

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    email = _get_ws_email(websocket)
    await manager.connect(websocket, email)
    try:
        while True:
            _ = await websocket.receive_text()
            try:
                await websocket.send_json({"event": "ws:pong", "ts": now_iso_utc()})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, email)
    except Exception:
        manager.disconnect(websocket, email)

async def emit_alert(email: str, title: str, body: str = "", extra: dict = None):
    payload = {"event": "alert:new", "title": title, "body": body, "ts": now_iso_utc()}
    if extra:
        payload["extra"] = extra
    await manager.send_to_user(email, payload)

async def emit_chat_new_message(para_email: str, de_email: str, msg_id: int, preview: str = ""):
    payload = {
        "event": "chat:new_message",
        "from": de_email,
        "id": msg_id,
        "preview": preview[:120],
        "ts": now_iso_utc(),
    }
    await manager.send_to_user(para_email, payload)

# ================== Archivos de chat (adjuntos) ==================
CHAT_ATTACH_DIR = os.path.join("static", "chat_adjuntos")
os.makedirs(CHAT_ATTACH_DIR, exist_ok=True)

CHAT_ALLOWED_EXT = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".txt",
    ".csv",
    ".xlsx",
    ".xls",
    ".docx",
    ".doc",
    ".pptx",
}
CHAT_MAX_FILES = 10
CHAT_MAX_TOTAL_MB = 50

# ================== Adjuntos de incidencias ==================
INCID_ATTACH_DIR = os.path.join("static", "incid_adjuntos")
os.makedirs(INCID_ATTACH_DIR, exist_ok=True)
INCID_ALLOWED_EXT = CHAT_ALLOWED_EXT
INCID_MAX_FILES = 10
INCID_MAX_TOTAL_MB = 25

# ================== Avatares (perfil) ==================
AVATAR_DIR = os.path.join("static", "avatars")
os.makedirs(AVATAR_DIR, exist_ok=True)
AVATAR_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AVATAR_MAX_MB = 2  # MB

# ================== Knowledge Base (KB) bootstrap ==================
# Carpeta base para almacenar originales de la KB (ingesta por rubro)
KB_STORAGE_DIR = os.path.join("storage", "kb")
os.makedirs(KB_STORAGE_DIR, exist_ok=True)

# Extensiones permitidas para KB (reusa y amplía)
KB_ALLOWED_EXT = set(CHAT_ALLOWED_EXT) | {".md", ".json", ".yaml", ".yml"}

def _kb_slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s.strip("-") or "rubro"

def _import_utils_module():
    """
    Importa utils como *paquete* para que funcionen imports relativos internos
    (evita 'attempted relative import with no known parent package').
    Prueba varios nombres comunes de raíz.
    """
    candidates = ["utils", "app.utils", "backend.utils", "server.utils", "src.utils", "sa.utils"]
    last_err = None
    for name in candidates:
        try:
            return import_module(name)
        except Exception as e:
            last_err = e
    # último intento: devolver el ya importado U (si existe) o propagar
    try:
        return U
    except Exception:
        print("· KB utils import error:", repr(last_err))
        return None

def _kb_funcs():
    """Descubre funciones KB en utils.* sin romper si no están todavía."""
    mod = _import_utils_module()
    get = (lambda m, n: getattr(m, n, None)) if mod else (lambda *_: None)
    return {
        "create_or_get_source": get(mod, "kb_create_or_get_source"),
        "ingest_file": get(mod, "kb_ingest_file"),
        "upsert_priority": get(mod, "kb_upsert_priority"),
        "list_sources": get(mod, "kb_list_sources"),          # opcional
        "list_priorities": get(mod, "kb_list_priorities"),    # opcional
        "session": get(mod, "kb_session"),                    # si existe, la usamos abajo
    }

def _kb_enabled() -> bool:
    if os.getenv("KB_ENABLED", "1") != "1":
        return False
    f = _kb_funcs()
    return bool(f["create_or_get_source"] and f["ingest_file"])

# ---- kb_session unificada y compatible ----
@contextmanager
def kb_session():
    """
    Usa utils.kb_session() si existe; si no, devuelve una sesión SQLAlchemy local;
    y si tampoco, un contexto nulo.
    """
    f = _kb_funcs().get("session")

    # 1) Si utils.kb_session existe, intentar usarlo
    if callable(f):
        try:
            # Caso A: f() devuelve un context manager
            with f() as db:
                yield db
                return
        except TypeError:
            # Caso B: f() devuelve una sesión simple (no context manager)
            try:
                db = f()
                try:
                    yield db
                finally:
                    try:
                        close = getattr(db, "close", None)
                        if callable(close):
                            close()
                    except Exception:
                        pass
                return
            except Exception as e:
                print("· kb_session(): utils.kb_session() no usable:", repr(e))
        except Exception as e:
            # Cualquier otro error: seguir con fallbacks
            print("· kb_session(): error usando utils.kb_session():", repr(e))

    # 2) Fallback: usar SessionLocal() si existe
    try:
        with SessionLocal() as db:
            yield db
            return
    except Exception as e:
        print("· kb_session(): fallback SessionLocal() no disponible:", repr(e))

    # 3) Último recurso: devolver None en un contexto nulo
    with nullcontext() as _:
        yield None

# =========================
# Helpers de historial/paginación (usados en Partes 2/3)
# =========================
def _paginate(items: List[dict], page: int, per_page: int) -> Tuple[List[dict], int, int, int, int]:
    total_items = len(items)
    total_pages = max(1, ceil(total_items / per_page)) if per_page else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], page, per_page, total_pages, total_items

def _extraer_ts_de_nombre(nombre: str) -> Optional[str]:
    """
    Busca patrones tipo 'resumen_YYYYMMDDHHMMSS.pdf' y devuelve el timestamp.
    """
    m = re.search(r"resumen_(\d{14})\.pdf$", (nombre or "").strip(), flags=re.I)
    return m.group(1) if m else None

def _historial_para_home(email: str, rol: str, q: str = "") -> List[dict]:
    """
    Agrega una capa de filtrado/búsqueda para la vista/endpoint de historial.
    Si es admin, puede ver todo; si no, solo su propio historial.
    """
    q = (q or "").strip().lower()
    try:
        if (rol or "").lower().startswith("admin"):
            data = obtener_historial_completo() or []
        else:
            data = obtener_historial(email) or []
    except Exception:
        data = []

    out = []
    for h in data:
        # normalizamos claves mínimas esperadas
        item = {
            "id": h.get("id") if isinstance(h, dict) else (h[0] if isinstance(h, (list, tuple)) and len(h) > 0 else None),
            "historial_id": h.get("historial_id") if isinstance(h, dict) else None,
            "usuario": h.get("usuario") if isinstance(h, dict) else (h[1] if isinstance(h, (list, tuple)) and len(h) > 1 else email),
            "nombre_archivo": h.get("nombre_archivo") if isinstance(h, dict) else (h[2] if isinstance(h, (list, tuple)) and len(h) > 2 else ""),
            "ruta_pdf": h.get("ruta_pdf") if isinstance(h, dict) else (h[3] if isinstance(h, (list, tuple)) and len(h) > 3 else ""),
            "resumen": h.get("resumen") if isinstance(h, dict) else (h[4] if isinstance(h, (list, tuple)) and len(h) > 4 else ""),
            "fecha": h.get("fecha") if isinstance(h, dict) else (h[5] if isinstance(h, (list, tuple)) and len(h) > 5 else ""),
        }
        text = f"{item['usuario']} {item['nombre_archivo']} {item['resumen']}".lower()
        if q and q not in text:
            continue
        out.append(item)

    # orden nuevo→viejo usando fecha si existe
    out.sort(key=lambda x: _parse_dt_utc(x.get("fecha")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return out

def _buscar_historial_usuario(user: str, timestamp: Optional[str] = None, nombre_pdf: Optional[str] = None) -> Optional[dict]:
    """
    Localiza el análisis del usuario por timestamp y/o nombre de PDF.
    Sirve para asociar la valoración cuando el front no envía el id.
    """
    try:
        data = obtener_historial(user) or []
    except Exception:
        data = []
    if not data:
        return None

    # normalizamos a dict
    norm: List[dict] = []
    for h in data:
        if isinstance(h, dict):
            norm.append(h)
        elif isinstance(h, (list, tuple)):
            norm.append({
                "id": h[0] if len(h) > 0 else None,
                "usuario": h[1] if len(h) > 1 else user,
                "nombre_archivo": h[2] if len(h) > 2 else "",
                "ruta_pdf": h[3] if len(h) > 3 else "",
                "resumen": h[4] if len(h) > 4 else "",
                "fecha": h[5] if len(h) > 5 else "",
            })

    if nombre_pdf:
        for h in norm:
            if (h.get("nombre_archivo") or "").strip().lower() == nombre_pdf.strip().lower():
                return h

    if timestamp:
        for h in norm:
            ts = _extraer_ts_de_nombre(h.get("nombre_archivo") or "")
            if ts and ts == timestamp:
                return h

    # fallback: el más reciente
    norm.sort(key=lambda x: _parse_dt_utc(x.get("fecha")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return norm[0] if norm else None
# =========================
# main.py — PARTE 2 / 6
# (login/logout, cambiar password, rating, analizar pliego)
# =========================

# =====================================================================
# ========================== CALENDARIO (DB utilitaria) ===============
# =====================================================================

CAL_DB = "calendar.sqlite3"


def cal_conn():
    conn = sqlite3.connect(CAL_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_calendar_db():
    with cal_conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS eventos(
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                start TEXT NOT NULL,
                end TEXT,
                all_day INTEGER NOT NULL DEFAULT 0,
                color TEXT,
                created_by TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS notificaciones(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                titulo TEXT NOT NULL,
                cuerpo TEXT,
                created_at TEXT NOT NULL,
                leida INTEGER NOT NULL DEFAULT 0
            )
            """
        )


init_calendar_db()


def _now_iso():
    return now_iso_utc()


def _event_row_to_dict(r: sqlite3.Row):
    return {
        "id": r["id"],
        "title": r["title"],
        "start": r["start"],
        "end": r["end"],
        "allDay": bool(r["all_day"]),
        "description": r["description"] or "",
        "color": r["color"] or "#0ea5e9",
    }


def _notify(user: str, titulo: str, cuerpo: str = ""):
    with cal_conn() as c:
        c.execute(
            "INSERT INTO notificaciones(user, titulo, cuerpo, created_at, leida) VALUES(?,?,?,?,0)",
            (user or "Desconocido", titulo, cuerpo, _now_iso()),
        )


async def notify_async(user: str, titulo: str, cuerpo: str = ""):
    _notify(user, titulo, cuerpo)
    await emit_alert(user, titulo, cuerpo)


# ====== NUEVO: rating pendiente liviano (sidecar) ======

def init_rating_pending_db():
    with cal_conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_ratings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                historial_id TEXT,
                timestamp TEXT,
                nombre_pdf TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_ratings_user ON pending_ratings(user)"
        )


init_rating_pending_db()


def _pr_add(user: str, historial_id: Optional[str], timestamp: str, nombre_pdf: str):
    with cal_conn() as c:
        c.execute("DELETE FROM pending_ratings WHERE user=?", (user,))
        c.execute(
            "INSERT INTO pending_ratings(user, historial_id, timestamp, nombre_pdf, created_at) VALUES(?,?,?,?,?)",
            (
                user,
                str(historial_id) if historial_id is not None else None,
                timestamp,
                nombre_pdf,
                _now_iso(),
            ),
        )


def _pr_get(user: str):
    with cal_conn() as c:
        r = c.execute(
            "SELECT historial_id, timestamp, nombre_pdf FROM pending_ratings WHERE user=? ORDER BY id DESC LIMIT 1",
            (user,),
        ).fetchone()
        if r:
            return {
                "historial_id": r["historial_id"],
                "timestamp": r["timestamp"],
                "nombre_pdf": r["nombre_pdf"],
            }
    return None


def _pr_clear(user: str):
    with cal_conn() as c:
        c.execute("DELETE FROM pending_ratings WHERE user=?", (user,))


# ================== Login/Logout ==================

@app.post("/login")
async def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    remember: Optional[str] = Form(default=None)  # "on" si tildan Recordarme
):
    """
    Normalización:
    - recorta espacios y pasa a lower
    - si no trae '@', agrega dominio por defecto (LOGIN_DEFAULT_DOMAIN, por defecto suizo.com)
    """
    raw = (email or "").strip().lower()
    if "@" not in raw and " " not in raw:
        domain = (os.getenv("LOGIN_DEFAULT_DOMAIN", "suizo.com") or "").strip().lower()
        email = f"{raw}@{domain}" if domain else raw
    else:
        email = raw

    usuario = obtener_usuario_por_email(email)  # (id, nombre, email, password, rol, activo)
    is_active = True
    if isinstance(usuario, (list, tuple)) and len(usuario) >= 6:
        is_active = bool(usuario[5])

    # Limpiar la sesión previa para evitar fijación de sesión
    try:
        request.session.clear()
    except Exception:
        pass

    if usuario and str(usuario[3]) == str(password) and is_active:
        request.session["usuario"] = usuario[2]
        request.session["email"] = usuario[2]
        request.session["rol"] = usuario[4]
        request.session["nombre"] = usuario[1] or usuario[2]
        request.session["remember"] = bool(remember)

        # Registrar sesión en tabla local (calendar.sqlite3)
        sid = uuid.uuid4().hex
        request.session["sid"] = sid
        nombre_s = request.session.get("nombre") or usuario[1] or usuario[2]
        # IP robusta (X-Forwarded-For o client.host)
        ip_hdr = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        ip_s = ip_hdr or (request.client.host if request.client else None)
        ua_s = request.headers.get("user-agent", "")
        now_iso = now_iso_utc()

        with cal_conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions(
                    id TEXT PRIMARY KEY,
                    user TEXT NOT NULL,
                    nombre TEXT,
                    ip TEXT,
                    ua TEXT,
                    login_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    logout_at TEXT,
                    closed_reason TEXT
                )
                """
            )
            c.execute(
                """
                INSERT INTO sessions(id, user, nombre, ip, ua, login_at, last_seen, logout_at, closed_reason)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (sid, request.session["usuario"], nombre_s, ip_s, ua_s, now_iso, now_iso, None, None),
            )

        return RedirectResponse("/", status_code=303)

    # Mensajes de error más claros
    if usuario and not is_active:
        err = "Tu usuario está desactivado. Consultá con un administrador."
    else:
        err = "Credenciales incorrectas"

    # Mantener la UX del login (mensajes + status)
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": err, "mensaje": None}, status_code=401
    )


@app.post("/logout")
async def logout_post(request: Request):
    sid = request.session.get("sid")
    now_iso = now_iso_utc()
    if sid:
        with cal_conn() as c:
            c.execute("UPDATE sessions SET logout_at=?, closed_reason=? WHERE id=?", (now_iso, "logout", sid))
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/logout")
async def logout_get(request: Request):
    sid = request.session.get("sid")
    now_iso = now_iso_utc()
    if sid:
        with cal_conn() as c:
            c.execute("UPDATE sessions SET logout_at=?, closed_reason=? WHERE id=?", (now_iso, "logout", sid))
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ================== Cambiar contraseña (vista + alias + post) ==================

@app.get("/cambiar-password", response_class=HTMLResponse)
async def cambiar_password_view(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    ok = request.query_params.get("ok")
    return templates.TemplateResponse("cambiar_password.html", {"request": request, "error": None, "ok": ok})


# Alias con guion_bajo -> redirige a la ruta canónica con guion (GET)
@app.get("/cambiar_password", response_class=HTMLResponse)
async def cambiar_password_alias():
    return RedirectResponse("/cambiar-password", status_code=307)


# NUEVO: aceptar barra final en GET
@app.get("/cambiar-password/")
async def cambiar_password_trailing_get():
    return RedirectResponse("/cambiar-password", status_code=307)


# POST canónico
@app.post("/cambiar-password")
async def cambiar_password_post(
    request: Request,
    actual: str = Form(...),
    nueva: str = Form(...),
    confirmar: str = Form(...),
):
    email = request.session.get("usuario")
    if not email:
        return RedirectResponse("/login", status_code=303)

    if (not nueva) or (nueva != confirmar):
        return templates.TemplateResponse(
            "cambiar_password.html",
            {"request": request, "error": "Las contraseñas no coinciden."},
            status_code=400,
        )

    row = obtener_usuario_por_email(email)
    if not row:
        return templates.TemplateResponse(
            "cambiar_password.html", {"request": request, "error": "Usuario no encontrado."}, status_code=404
        )

    # Validación de contraseña actual (según tu esquema plano)
    if str(row[3]) != str(actual):
        return templates.TemplateResponse(
            "cambiar_password.html", {"request": request, "error": "La contraseña actual es incorrecta."}, status_code=400
        )

    actor_user_id, ip = _actor_info(request)
    try:
        actualizar_password(email.lower(), nueva, actor_user_id=actor_user_id, ip=ip)
    except Exception as e:
        print("· cambiar_password_post:", repr(e))
        return templates.TemplateResponse(
            "cambiar_password.html",
            {"request": request, "error": "No se pudo actualizar la contraseña."},
            status_code=500,
        )

    try:
        await notify_async(email, "Contraseña actualizada", "Tu contraseña fue actualizada correctamente.")
    except Exception:
        pass

    return RedirectResponse("/cambiar-password?ok=1", status_code=303)


# NUEVO: aceptar barra final en POST (reusa la lógica canónica)
@app.post("/cambiar-password/")
async def cambiar_password_trailing_post(
    request: Request,
    actual: str = Form(...),
    nueva: str = Form(...),
    confirmar: str = Form(...),
):
    return await cambiar_password_post(request, actual=actual, nueva=nueva, confirmar=confirmar)


# NUEVO: aceptar guion_bajo en POST (compatibilidad)
@app.post("/cambiar_password")
async def cambiar_password_underscore_post(
    request: Request,
    actual: str = Form(...),
    nueva: str = Form(...),
    confirmar: str = Form(...),
):
    return await cambiar_password_post(request, actual=actual, nueva=nueva, confirmar=confirmar)


# NUEVO: aceptar guion_bajo + barra final en POST
@app.post("/cambiar_password/")
async def cambiar_password_underscore_post_slash(
    request: Request,
    actual: str = Form(...),
    nueva: str = Form(...),
    confirmar: str = Form(...),
):
    return await cambiar_password_post(request, actual=actual, nueva=nueva, confirmar=confirmar)


# ================== Rating/Análisis ==================

class RatingIn(BaseModel):
    historial_id: Optional[int] = None
    rating: Optional[int] = None
    timestamp: Optional[str] = None
    nombre_pdf: Optional[str] = None
    estrellas: Optional[int] = None
    comentario: Optional[str] = None


@app.get("/api/rating/pending")
async def rating_pending(request: Request):
    user = request.session.get("usuario", "")
    if not user:
        return {"pending": False}

    pr = _pr_get(user)
    pend_flag = False
    try:
        pend_flag = bool(tiene_valoracion_pendiente(user))
    except Exception:
        pass

    if pr:
        last = pr
        try:
            last["historial_id"] = int(last.get("historial_id")) if last.get("historial_id") else None
        except Exception:
            last["historial_id"] = None
        return {"pending": True, "last": last}

    if pend_flag:
        h = _buscar_historial_usuario(user)
        last = None
        if h:
            nombre = (h.get("nombre_archivo") or "")
            ts = h.get("timestamp") or _extraer_ts_de_nombre(nombre)
            hid = h.get("historial_id") or h.get("id")
            last = {
                "timestamp": ts or "",
                "nombre_pdf": nombre or "",
                "historial_id": hid if isinstance(hid, int) else None,
            }
        return {"pending": True, "last": last}

    return {"pending": False, "last": None}


@app.get("/api/rating/pendiente")
async def rating_pendiente_alias(request: Request):
    data = await rating_pending(request)
    return {"pendiente": data.get("pending", False), "last": data.get("last")}


@app.post("/api/rating")
async def enviar_rating(request: Request, payload: RatingIn):
    user = request.session.get("usuario")
    if not user:
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    rating = None
    if isinstance(payload.estrellas, int):
        rating = payload.estrellas
    elif isinstance(payload.rating, int):
        rating = payload.rating

    if not rating or rating < 1 or rating > 5:
        return JSONResponse({"error": "Rating inválido. Use un entero 1..5."}, status_code=400)

    historial_id = payload.historial_id
    if not historial_id:
        h = _buscar_historial_usuario(user, timestamp=payload.timestamp, nombre_pdf=payload.nombre_pdf)
        if h:
            hid = h.get("historial_id") or h.get("id")
            if isinstance(hid, int):
                historial_id = hid

    if not historial_id:
        return JSONResponse({"error": "No pude identificar el análisis a valorar."}, status_code=400)

    actor_user_id, ip = _actor_info(request)
    try:
        marcar_valoracion_historial(historial_id, rating, actor_user_id=actor_user_id, ip=ip)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        print("· Error enviar_rating:", repr(e))
        return JSONResponse({"error": "No se pudo registrar la valoración"}, status_code=500)

    try:
        _pr_clear(user)
    except Exception:
        pass

    try:
        if payload.comentario:
            await notify_async(user, "¡Gracias por tu valoración!", f"Dejaste {rating}/5: {payload.comentario[:140]}")
        else:
            await notify_async(user, "¡Gracias por tu valoración!", f"Calificación {rating}/5 registrada.")
    except Exception:
        pass

    return {"ok": True, "message": "Valoración registrada"}


# *** AJUSTADO ***: unificación de ruta de PDFs tras la generación
@app.post("/analizar-pliego")
async def analizar_pliego(request: Request, archivos: List[UploadFile] = File(...)):
    usuario = request.session.get("usuario", "Anónimo")

    # Bloqueo si falta config de IA (evita colgarse adentro de utils)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1") or os.getenv("OPENAI_API_BASE")):
        return JSONResponse(
            {"error": "Falta configurar el proveedor de IA (OPENAI_API_KEY / OPENAI_API_BASE)."},
            status_code=503
        )

    if not archivos:
        return JSONResponse({"error": "Subí al menos un archivo"}, status_code=400)

    for a in archivos:
        if not a or not a.filename:
            continue
        _validate_ext(a.filename)

    # --- TIMEOUTS CONFIGURABLES ---
    ANALYZE_TIMEOUT = float(os.getenv("ANALYZE_TIMEOUT", "180"))  # 3 min
    PDF_TIMEOUT = float(os.getenv("PDF_TIMEOUT", "60"))           # 1 min

    # 1) Analizar anexos con timeout
    try:
        resumen = await asyncio.wait_for(
            run_in_threadpool(analizar_anexos, archivos),
            timeout=ANALYZE_TIMEOUT
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            {"error": f"El análisis tardó más de {int(ANALYZE_TIMEOUT)}s y fue cancelado (timeout). "
                      "Probá de nuevo o reducí el tamaño del archivo."},
            status_code=504
        )
    except Exception as e:
        logger.exception("Error en /analizar-pliego -> analizar_anexos")
        return JSONResponse({"error": f"Fallo en el análisis: {e}"}, status_code=500)

    # 2) Generar PDF con timeout (y luego asegurar que quede en el directorio servido)
    timestamp = now_stamp_ar()
    nombre_archivo_pdf = f"resumen_{timestamp}.pdf"

    try:
        # algunos run_in_threadpool no aceptan kwargs; probamos y caemos al partial
        try:
            await asyncio.wait_for(
                run_in_threadpool(generar_pdf_con_plantilla, resumen, nombre_archivo_pdf=nombre_archivo_pdf),
                timeout=PDF_TIMEOUT
            )
        except TypeError:
            await asyncio.wait_for(
                run_in_threadpool(partial(generar_pdf_con_plantilla, resumen, nombre_archivo_pdf=nombre_archivo_pdf)),
                timeout=PDF_TIMEOUT
            )
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Timeout generando el PDF"}, status_code=504)
    except Exception as e:
        logger.exception("Error en /analizar-pliego -> generar_pdf_con_plantilla")
        return JSONResponse({"error": f"Fallo al generar PDF: {e}", "resumen": resumen}, status_code=500)

    # === NUEVO: asegurar que el PDF quede accesible en /generated_pdfs ===
    try:
        pdf_abs = _ensure_pdf_in_serve_dir(nombre_archivo_pdf)  # usa PDF_SERVE_DIR (definido en Parte 1)
        logger.info("[PDF] ensure -> %s | serve_dir=%s", pdf_abs, str(PDF_SERVE_DIR))
        if not pdf_abs:
            logger.error("[PDF] No pude localizar '%s' en candidatos; botón de descarga fallaría.", nombre_archivo_pdf)
            return JSONResponse({"error": "El PDF se generó pero no se pudo ubicar para descarga."}, status_code=500)
    except Exception as e:
        logger.exception("[PDF] Error asegurando PDF en carpeta servida")
        return JSONResponse({"error": f"No se pudo preparar el PDF para descarga: {e}"}, status_code=500)

    # 3) Guardar en historial (guardar SOLO el basename; la ruta se reconstruye al descargar)
    analisis_id = uuid.uuid4().hex
    try:
        historial_id = iniciar_analisis_historial(
            usuario=usuario,
            nombre_archivo=nombre_archivo_pdf,
            ruta_pdf=nombre_archivo_pdf,
            analisis_id=analisis_id,
            resumen_texto=resumen,
        )
    except Exception as e:
        print("· Error iniciar_analisis_historial:", repr(e))
        try:
            guardar_en_historial(timestamp, usuario, nombre_archivo_pdf, nombre_archivo_pdf, resumen)
        except Exception:
            pass
        historial_id = None

    # 4) KB: guardar originales e intentar ingesta si hay helpers disponibles
    saved_paths: List[str] = []
    try:
        # Guardar originales en storage/kb/<usuario>/<timestamp>/
        if os.getenv("KB_SAVE_ORIGINALS", "1") == "1":
            user_dir = os.path.join(KB_STORAGE_DIR, _email_safe(usuario), timestamp)
            os.makedirs(user_dir, exist_ok=True)
            for i, a in enumerate(archivos, start=1):
                if not a or not a.filename:
                    continue
                base = _safe_basename(a.filename)
                ext = os.path.splitext(a.filename)[1].lower()
                dst = os.path.join(user_dir, f"{i:02d}_{base}{ext}")
                try:
                    await a.seek(0)
                    await _save_upload_stream(a, dst)
                    saved_paths.append(dst)
                except Exception as e:
                    print("· No se pudo guardar original KB:", a.filename, repr(e))

        # Intentar ingesta silenciosa si existen funciones en utils.*
        if saved_paths and _kb_enabled():
            fns = _kb_funcs()
            src_name = f"default:{_email_safe(usuario)}"
            try:
                with kb_session() as db:
                    # create_or_get_source: tolerante a firmas
                    try:
                        source_ref = fns["create_or_get_source"](db, src_name)
                    except TypeError:
                        try:
                            source_ref = fns["create_or_get_source"](db, src_name, {"owner": usuario})
                        except Exception:
                            source_ref = src_name  # fallback: pasar nombre
                    # Ingestar cada archivo con firmas flexibles
                    for p in saved_paths:
                        try:
                            try:
                                fns["ingest_file"](db, source_ref, p, {"uploaded_by": usuario, "timestamp": timestamp})
                            except TypeError:
                                fns["ingest_file"](db, source_ref, p)
                        except Exception as ie:
                            print("· Ingest fallida para", p, repr(ie))
            except Exception as e:
                print("· KB ingest omitida:", repr(e))
    except Exception as e:
        print("· KB save/ingest error:", repr(e))

    # 5) Registrar rating pendiente
    try:
        _pr_add(usuario, historial_id, timestamp, nombre_archivo_pdf)
    except Exception as e:
        print("· No se pudo registrar pending_ratings:", repr(e))

    return {
        "resumen": resumen,
        "pdf": nombre_archivo_pdf,  # basename (el endpoint construye la ruta)
        "timestamp": timestamp,
        "historial_id": historial_id,
        "analisis_id": analisis_id,
    }
# =========================
# main.py — PARTE 3 / 6
# (historial, usuario/avatares, diagnóstico)
# =========================

# ---- Servido de PDFs: carpeta y helpers robustos ----
PDF_SERVE_DIR = os.getenv("PDF_SERVE_DIR", "generated_pdfs")
os.makedirs(PDF_SERVE_DIR, exist_ok=True)

# Lugares candidatos donde el generador pudo dejar el PDF (por si utils escribe en otro lado)
_PDF_CANDIDATE_DIRS = [
    PDF_SERVE_DIR,
    ".",                      # cwd
    "static",
    "storage",
    "/tmp",
]

def _find_pdf_path(filename: str) -> Optional[str]:
    """Devuelve ruta absoluta si encuentra el PDF en candidatos; None si no."""
    name = os.path.basename(filename or "")
    if not name.lower().endswith(".pdf"):
        return None
    for d in _PDF_CANDIDATE_DIRS:
        p = os.path.abspath(os.path.join(d, name))
        if os.path.isfile(p):
            return p
    return None

def _ensure_pdf_in_serve_dir(filename: str) -> Optional[str]:
    """
    Garantiza que <PDF_SERVE_DIR>/<filename> exista.
    Si el archivo está en otro directorio candidato, lo copia acá.
    Devuelve ruta absoluta final o None si no se pudo ubicar/copiar.
    """
    name = os.path.basename(filename or "")
    src = _find_pdf_path(name)
    if not src:
        return None
    dst = os.path.abspath(os.path.join(PDF_SERVE_DIR, name))
    if os.path.abspath(src) == dst:
        return dst
    try:
        os.makedirs(PDF_SERVE_DIR, exist_ok=True)
        # copia segura sin dependencias externas
        with open(src, "rb") as f_in, open(dst, "wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
        return dst
    except Exception:
        try:
            import shutil
            shutil.copyfile(src, dst)
            return dst
        except Exception:
            return None

# ===== Historial (usa helpers definidos en PARTE 1) =====

@app.get("/historial")
async def ver_historial(
    request: Request,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    q: str = Query(default=""),
):
    """
    Devuelve el historial paginado en JSON (filtrado por usuario/rol).
    Útil para futuras mejoras de UI (carga perezosa, búsqueda, etc.).
    """
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    email = request.session.get("usuario")
    rol = request.session.get("rol", "usuario")

    data = _historial_para_home(email=email, rol=rol, q=q)
    items, page, per_page, total_pages, total_items = _paginate(data, page, per_page)

    return {
        "items": items,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_items": total_items,
        "q": q,
    }


@app.get("/historia")
async def alias_historia():
    return RedirectResponse("/?goto=historial", status_code=307)


@app.get("/analisis")
@app.get("/analisis/nuevo")
@app.get("/report")
async def alias_analisis():
    return RedirectResponse("/?goto=analisis", status_code=307)


# ---- Descarga de PDFs (robusta) ----
@app.get("/descargar/{archivo}")
async def descargar_pdf(archivo: str):
    """
    Descarga un PDF por nombre de archivo (basename).
    Busca en varios directorios y, si es necesario, lo copia a PDF_SERVE_DIR.
    """
    name = os.path.basename(archivo or "")
    if not name or not name.lower().endswith(".pdf"):
        return JSONResponse({"error": "Nombre de archivo inválido"}, status_code=400)

    # Garantizar que exista en la carpeta servida
    final_abs = _ensure_pdf_in_serve_dir(name)
    if not final_abs or not os.path.isfile(final_abs):
        return JSONResponse({"error": "Archivo no encontrado"}, status_code=404)

    return FileResponse(final_abs, media_type="application/pdf", filename=name)


# Nuevo: Descargar el último PDF del usuario logueado
@app.get("/descargar/ultimo")
async def descargar_ultimo(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    user = request.session.get("usuario")
    # Tomamos del historial del usuario el más reciente
    h = _buscar_historial_usuario(user)
    if not h:
        return JSONResponse({"error": "No hay informes recientes para descargar"}, status_code=404)

    filename = (h.get("nombre_archivo") or "").strip()
    if not filename:
        return JSONResponse({"error": "No pude determinar el nombre del PDF"}, status_code=404)

    # Reutilizamos el endpoint principal
    return await descargar_pdf(filename)

# Alias por compatibilidad con frontends previos
@app.get("/descargar-ultimo")
async def descargar_ultimo_alias(request: Request):
    return await descargar_ultimo(request)


@app.delete("/eliminar/{timestamp}")
async def eliminar_archivo(timestamp: str):
    eliminar_del_historial(timestamp)
    ruta = os.path.join(PDF_SERVE_DIR, f"resumen_{os.path.basename(timestamp)}.pdf")
    if os.path.exists(ruta):
        try:
            os.remove(ruta)
        except Exception:
            pass
    return {"mensaje": "Eliminado correctamente"}


# ================== Usuario actual ==================
@app.get("/usuario-actual")
async def usuario_actual(request: Request):
    """
    Devuelve info del usuario logueado para el topbar:
    email, rol, nombre legible y url de avatar (si existe).
    """
    email = request.session.get("usuario", "")
    rol = request.session.get("rol", "usuario")
    row = obtener_usuario_por_email(email) if email else None
    # nombre preferente: DB.nombre -> session['nombre'] -> email -> 'Desconocido'
    nombre = (row[1] if (row and len(row) > 1) else None) or request.session.get("nombre") or (email or "Desconocido")

    # Buscar avatar si existe
    avatar_url = ""
    if email:
        prefix = _email_safe(email)
        for ext in (".webp", ".png", ".jpg", ".jpeg"):
            p = os.path.join(AVATAR_DIR, prefix + ext)
            if os.path.isfile(p):
                avatar_url = f"/{p.replace(os.sep, '/')}"
                break

    return {
        "usuario": email or "Desconocido",
        "rol": rol,
        "nombre": nombre,
        "avatar_url": avatar_url,
    }


# ===== Subir/actualizar avatar =====
@app.post("/perfil/avatar")
async def subir_avatar(request: Request, avatar: UploadFile = File(...)):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    orig = avatar.filename or ""
    ext = os.path.splitext(orig)[1].lower()
    if ext not in AVATAR_ALLOWED_EXT:
        return JSONResponse({"error": f"Formato no permitido: {ext}"}, status_code=400)

    data = await avatar.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > AVATAR_MAX_MB:
        return JSONResponse({"error": f"Máximo {AVATAR_MAX_MB} MB"}, status_code=400)

    email = request.session.get("usuario")
    prefix = _email_safe(email)
    dst = os.path.join(AVATAR_DIR, prefix + ext)

    # elimina variantes anteriores (si existían)
    for e in (".webp", ".png", ".jpg", ".jpeg"):
        p = os.path.join(AVATAR_DIR, prefix + e)
        if os.path.isfile(p) and p != dst:
            try:
                os.remove(p)
            except Exception:
                pass

    with open(dst, "wb") as f:
        f.write(data)

    url = f"/{dst.replace(os.sep, '/')}"

    try:
        await emit_alert(email, "Perfil actualizado", "Tu avatar se actualizó correctamente")
    except Exception:
        pass

    return {"ok": True, "avatar_url": url}


# ================== Diagnóstico (controlado por env) ==================
@app.get("/__diag/auth")
async def diag_auth(request: Request):
    """
    Habilitar con ENABLE_DIAG=1 (no expone secretos).
    Útil para verificar si la cookie de sesión se está guardando.
    """
    if (os.getenv("ENABLE_DIAG", "0") != "1"):
        raise HTTPException(status_code=404, detail="Not found")

    sess = request.session or {}
    headers = {
        "user_agent": request.headers.get("user-agent", ""),
        "accept": request.headers.get("accept", ""),
    }
    return {
        "logged_in": bool(sess.get("usuario")),
        "session_keys": sorted(list(sess.keys())),
        "session_preview": {
            "usuario": sess.get("usuario"),
            "rol": sess.get("rol"),
            "nombre": sess.get("nombre"),
            "sid_present": bool(sess.get("sid")),
        },
        "cookie_present": ("session" in (request.cookies or {})),
        "route": str(request.url),
        "headers": headers,
    }


# ? Diagnóstico de templates/loader
@app.get("/__diag/templates")
async def _diag_templates():
    if (os.getenv("ENABLE_DIAG", "0") != "1"):
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "loader": str(getattr(templates.env, "loader", "")),
        "auto_reload": bool(getattr(templates.env, "auto_reload", False)),
    }


# Diagnóstico rápido siempre disponible (no revela info sensible)
@app.get("/debug/whoami")
async def debug_whoami(request: Request):
    sess = request.session or {}
    return {
        "logged_in": bool(sess.get("usuario")),
        "session_keys": sorted(list(sess.keys())),
        "session_preview": {
            "usuario": sess.get("usuario"),
            "rol": sess.get("rol"),
            "nombre": sess.get("nombre"),
            "sid_present": bool(sess.get("sid")),
        },
        "cookie_present": ("session" in (request.cookies or {})),
        "route": str(request.url),
    }
# =========================
# main.py — PARTE 4 / 6
# (chat OpenAI, chat interno)
# =========================

# ================== Helpers de contexto para Chat OpenAI ==================
def _build_chat_context(historial: List[dict], usuario_actual: str, max_items: int = 8, max_chars: int = 1500) -> str:
    """
    Construye un contexto compacto usando el último análisis del usuario + extractos del historial.
    Limita cantidad de items y longitud para evitar prompts gigantes.
    """
    if not historial:
        return "(Sin historial todavía.)"

    # último análisis del usuario
    mine = [h for h in historial if h.get("usuario") == usuario_actual and h.get("resumen")]
    mine.sort(key=lambda h: _parse_dt_utc(h.get("fecha")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    ultimo = mine[0] if mine else None
    if ultimo:
        ultimo_resumen = (
            f"\n ?? Último análisis del usuario actual:\n"
            f" - Fecha: {ultimo.get('fecha')}\n"
            f" - Archivo: {ultimo.get('nombre_archivo')}\n"
            f" - Resumen: {ultimo.get('resumen')}\n"
        )
    else:
        ultimo_resumen = "(El usuario aún no tiene análisis registrados.)"

    # resto del historial (global), ordenado nuevo→viejo
    others = [h for h in historial if h.get("resumen")]
    others.sort(key=lambda h: _parse_dt_utc(h.get("fecha")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    lines = []
    for h in others[:max_items]:
        resumen = str(h.get("resumen") or "").strip()
        if len(resumen) > max_chars:
            resumen = resumen[:max_chars] + "…"
        lines.append(f"- [{h.get('fecha')}] {h.get('usuario')} analizó '{h.get('nombre_archivo')}' y obtuvo:\n{resumen}\n")

    contexto_general = "\n".join(lines)
    return f"{ultimo_resumen}\n\n?? Historial breve:\n{contexto_general}"


async def _call_chat_llm(mensaje: str, usuario_actual: str) -> str:
    try:
        historial = obtener_historial_completo() or []
    except Exception:
        historial = []
    contexto = _build_chat_context(historial, usuario_actual)

    try:
        CHAT_LLM_TIMEOUT = float(os.getenv("CHAT_LLM_TIMEOUT", "60"))
    except Exception:
        CHAT_LLM_TIMEOUT = 60.0

    def _bridge():
        # 1) (msg, ctx, user)
        try:
            return responder_chat_openai(mensaje, contexto, usuario_actual)
        except TypeError:
            pass
        # 2) con kwargs
        try:
            return responder_chat_openai(mensaje, contexto=contexto, usuario=usuario_actual)
        except TypeError:
            pass
        # 3) (msg, ctx)
        try:
            return responder_chat_openai(mensaje, contexto)
        except TypeError:
            pass
        # 4) (msg, user)
        try:
            return responder_chat_openai(mensaje, usuario_actual)
        except TypeError:
            pass
        # 5) (msg) — la más común
        return responder_chat_openai(mensaje)

    try:
        logger.info("Chat LLM: usuario=%s, len(mensaje)=%d, timeout=%.1fs",
                    usuario_actual, len(mensaje or ""), CHAT_LLM_TIMEOUT)
        return await asyncio.wait_for(run_in_threadpool(_bridge), timeout=CHAT_LLM_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Chat LLM timeout (%.1fs) para usuario=%s", CHAT_LLM_TIMEOUT, usuario_actual)
        return "Estoy tardando más de lo normal en responder. Probá de nuevo en un momento."
    except Exception as e:
        logger.exception("Error en _call_chat_llm")
        return f"[Error de chat] {e}"


# ================== API puente (Chat OpenAI) ==================
@app.post("/chat-openai")
async def chat_openai(request: Request):
    data = await request.json()
    mensaje = (data.get("mensaje") or "").strip()
    usuario_actual = request.session.get("usuario", "Desconocido")

    if not mensaje:
        return JSONResponse({"respuesta": "Decime qué necesitás revisar del pliego ??"})

    # Chequeo rápido de clave (evita silencio si falta)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1") or os.getenv("OPENAI_API_BASE")):
        logger.error("OPENAI_API_KEY no está configurada en el servidor")
        return JSONResponse({"respuesta": "No puedo responder porque falta la configuración del proveedor de IA (OPENAI_API_KEY). Avisá al admin."}, status_code=503)

    respuesta = await _call_chat_llm(mensaje, usuario_actual)
    return JSONResponse({"respuesta": respuesta})


@app.post("/api/chat-openai")
async def api_chat_openai(request: Request, payload: dict = Body(...)):
    mensaje = (payload or {}).get("message", "").strip()
    usuario_actual = request.session.get("usuario", "Desconocido")

    if not mensaje:
        return JSONResponse({"reply": "Decime qué necesitás revisar del pliego ??"})

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1") or os.getenv("OPENAI_API_BASE")):
        logger.error("OPENAI_API_KEY no está configurada en el servidor (API)")
        return JSONResponse({"reply": "No puedo responder porque falta la configuración del proveedor de IA (OPENAI_API_KEY)."}, status_code=503)

    respuesta = await _call_chat_llm(mensaje, usuario_actual)
    return JSONResponse({"reply": respuesta})


# ===== Mini vista embebida para el widget del topbar/FAB =====
@app.get("/chat_openai_embed", response_class=HTMLResponse)
async def chat_openai_embed(request: Request):
    if not request.session.get("usuario"):
        return HTMLResponse("<div style='padding:12px'>Iniciá sesión para usar el chat.</div>")

    html = """<!doctype html><html><head>
 <meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
 <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
 <style>#t{ resize:none; min-height:42px; max-height:150px; }</style>
 </head><body class="p-2" style="background:transparent">
 <div id="log" class="mb-2" style="height:410px; overflow:auto; background:#f6f8fb; border-radius:12px; padding:8px;"></div>
 <form id="f" class="d-flex gap-2">
   <textarea id="t" class="form-control" placeholder="Escribe tu mensaje..." autocomplete="off" autofocus></textarea>
   <button id="send" type="button" class="btn btn-primary">Enviar</button>
 </form>
 <script>
  const log = document.getElementById('log');
  const ta = document.getElementById('t');
  const btn = document.getElementById('send');
  function esc(s){ return (s||'').replaceAll('<','&lt;').replaceAll('>','&gt;'); }
  function add(b){ const p=document.createElement('div'); p.innerHTML=b; log.appendChild(p); log.scrollTop=log.scrollHeight; }
  function autosize(){ ta.style.height='auto'; ta.style.height = Math.min(ta.scrollHeight, 150) + 'px'; }
  ta.addEventListener('input', autosize); autosize();
  let busy = false;
  async function send(){
    if(busy) return;
    const v = ta.value.trim();
    if(!v) return;
    busy = true; btn.disabled = true;
    add('<div><b>Tú:</b> '+esc(v)+'</div>');
    ta.value=''; autosize();
    try{
      const r = await fetch('/chat-openai', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({mensaje:v})
      });
      const j = await r.json().catch(()=>({}));
      add('<div class="mt-1"><b>IA:</b> '+(j.respuesta||'')+'</div>');
    }catch(_){
      add('<div class="text-danger mt-1"><b>Error:</b> No se pudo enviar.</div>');
    } finally{
      busy=false; btn.disabled=false; ta.focus();
    }
  }
  ta.addEventListener('keydown', (e)=>{
    if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }
  });
  btn.addEventListener('click', (e)=>{ e.preventDefault(); send(); });
 </script>
 </body></html>"""
    return HTMLResponse(html)


# ================== Chat interno (UI) ==================
@app.get("/chat", response_class=HTMLResponse)
async def chat_view(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("chat.html", {"request": request})


# ================== Chat interno (API) ==================
def _is_no_table_error(e: Exception) -> bool:
    return isinstance(e, sqlite3.OperationalError) and "no such table" in str(e).lower()


# ---------- NORMALIZADOR DE USUARIOS (tupla/dict) ----------
def _norm_rol(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("admin"):
        return "admin"
    if s.startswith("usuar"):
        return "usuario"
    if s == "borrado":
        return "borrado"
    return "usuario"


def _user_row_to_dict(u):
    """Acepta dicts o tuplas y devuelve {'id','nombre','email','rol','activo'}."""
    if isinstance(u, dict):
        return {
            "id": u.get("id"),
            "nombre": u.get("nombre") or "",
            "email": (u.get("email") or "").lower(),
            "rol": _norm_rol(u.get("rol") or u.get("role") or "usuario"),
            "activo": int(u.get("activo")) if u.get("activo") is not None else 1,
        }
    if isinstance(u, (list, tuple)):
        # (id, nombre, email, password, rol, activo)
        if len(u) >= 6:
            return {
                "id": u[0],
                "nombre": u[1] or "",
                "email": (u[2] or "").lower(),
                "rol": _norm_rol(u[4] or "usuario"),
                "activo": int(u[5]) if u[5] is not None else 1,
            }
        # (id, nombre, email, rol, activo)
        if len(u) == 5:
            return {
                "id": u[0],
                "nombre": u[1] or "",
                "email": (u[2] or "").lower(),
                "rol": _norm_rol(u[3] or "usuario"),
                "activo": int(u[4]) if u[4] is not None else 1,
            }
    return {"id": None, "nombre": "", "email": "", "rol": "usuario", "activo": 0}


@app.get("/api/usuarios")
async def api_buscar_usuarios(request: Request, term: str = "", limit: int = 8):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    term = (term or "").strip()
    if not term:
        return {"items": []}

    try:
        raw = buscar_usuarios(term, limit=limit) or []
        norm = [_user_row_to_dict(u) for u in raw]
        return {"items": [{"id": u["id"], "nombre": u["nombre"], "email": u["email"]} for u in norm]}
    except Exception as e:
        print("? Error api_buscar_usuarios:", repr(e))
        return JSONResponse({"error": "No se pudo completar la búsqueda"}, status_code=500)


@app.post("/chat/enviar")
async def chat_enviar(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    para = data.get("para")
    texto = data.get("texto", "").strip()
    if not para or not texto:
        return JSONResponse({"error": "Faltan campos: para, texto"}, status_code=400)

    de = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)

    try:
        msg_id = enviar_mensaje(
            de_email=de,
            para_email=para,
            texto=texto,
            actor_user_id=actor_user_id,
            ip=ip,
        )
        await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=texto)
        return JSONResponse({"ok": True, "id": msg_id})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": False, "error": "Inicialicé las tablas de chat, intentá de nuevo."}, status_code=503)
        print("? Error chat_enviar:", repr(e))
        return JSONResponse({"error": "No se pudo enviar el mensaje"}, status_code=500)


@app.post("/chat/enviar-archivos")
async def chat_enviar_archivos(
    request: Request,
    para: str = Form(...),
    texto: str = Form(default=""),
    archivos: List[UploadFile] = File(default=[]),
):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    de = request.session.get("usuario")
    files = [a for a in archivos if a and a.filename]
    if len(files) > CHAT_MAX_FILES:
        return JSONResponse({"error": f"Máximo {CHAT_MAX_FILES} archivos por mensaje"}, status_code=400)

    actor_user_id, ip = _actor_info(request)
    try:
        msg_id = enviar_mensaje(
            de_email=de,
            para_email=para,
            texto=texto or "",
            actor_user_id=actor_user_id,
            ip=ip,
        )
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": False, "error": "Inicialicé las tablas de chat, intentá de nuevo."}, status_code=503)
        print("? Error creando mensaje:", repr(e))
        return JSONResponse({"error": "No se pudo crear el mensaje"}, status_code=500)

    ts = now_stamp_ar()
    total_bytes = 0

    for i, archivo in enumerate(files, start=1):
        orig = archivo.filename
        _validate_ext(orig)
        ext = os.path.splitext(orig)[1].lower()
        base = _safe_basename(orig)
        safe_name = f"{ts}_{de.replace('@','_at_')}_{i:02d}_{base}{ext}"
        path = os.path.join(CHAT_ATTACH_DIR, safe_name)

        written = await _save_upload_stream(archivo, path)
        total_bytes += written

        # Límite correcto del chat
        if (total_bytes / (1024 * 1024)) > CHAT_MAX_TOTAL_MB:
            try:
                os.remove(path)
            except Exception:
                pass
            return JSONResponse({"error": f"Tamaño total supera {CHAT_MAX_TOTAL_MB} MB"}, status_code=400)

        try:
            guardar_adjunto(
                mensaje_id=msg_id,
                filename=safe_name,
                original=orig,
                mime=archivo.content_type or "",
                size=written,
            )
        except Exception as e:
            print("? Error guardar_adjunto:", repr(e))

    await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=(texto or "[Adjuntos]"))
    return JSONResponse({"ok": True, "id": msg_id})


@app.post("/chat/enviar-archivo")
async def chat_enviar_archivo(
    request: Request,
    para: str = Form(...),
    texto: str = Form(default=""),
    archivo: UploadFile = File(...),
):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    archivos = [archivo] if (archivo and getattr(archivo, "filename", None)) else []
    return await chat_enviar_archivos(request, para=para, texto=texto, archivos=archivos)


@app.get("/chat/adjunto/{filename}")
async def chat_adjunto(filename: str):
    filename = os.path.basename(filename)
    path = os.path.join(CHAT_ATTACH_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "No encontrado"}, status_code=404)
    return FileResponse(path)


@app.get("/chat/hilos")
async def chat_hilos(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    yo = request.session.get("usuario")
    try:
        hilos = obtener_hilos_para(yo)
        return JSONResponse({"hilos": hilos})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"hilos": []})
        print("? Error chat_hilos:", repr(e))
        return JSONResponse({"error": "No se pudieron obtener los hilos"}, status_code=500)


@app.get("/chat/mensajes")
async def chat_mensajes(request: Request, con: str, limit: int = 100):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    yo = request.session.get("usuario")
    if not con:
        return JSONResponse({"error": "Falta parámetro 'con' (email del contacto)"}, status_code=400)

    limit = max(1, min(int(limit or 100), 500))

    try:
        mensajes = obtener_mensajes_entre(yo, con, limit=limit)
        return JSONResponse({"entre": [yo, con], "mensajes": mensajes})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"entre": [yo, con], "mensajes": []})
        print("? Error chat_mensajes:", repr(e))
        return JSONResponse({"error": "No se pudieron obtener los mensajes"}, status_code=500)


@app.post("/chat/marcar-leidos")
async def chat_marcar_leidos(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    de = data.get("de")
    yo = request.session.get("usuario")

    if not de:
        return JSONResponse({"error": "Falta 'de' (email del contacto)"}, status_code=400)

    try:
        marcar_mensajes_leidos(de_email=de, para_email=yo)
        return JSONResponse({"ok": True})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": True})
        print("? Error chat_marcar_leidos:", repr(e))
        return JSONResponse({"error": "No se pudo marcar como leídos"}, status_code=500)


@app.get("/chat/no-leidos")
async def chat_no_leidos(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    yo = request.session.get("usuario")
    try:
        total = contar_no_leidos(yo)
        return JSONResponse({"no_leidos": total})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"no_leidos": 0})
        print("? Error chat_no_leidos:", repr(e))
        return JSONResponse({"error": "No se pudo obtener el conteo"}, status_code=500)


@app.post("/chat/ocultar")
async def chat_ocultar(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    con = (data or {}).get("con")
    if not con:
        return JSONResponse({"error": "Falta 'con' (email del contacto)"}, status_code=400)

    yo = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)

    try:
        ocultar_hilo(owner_email=yo, otro_email=con, actor_user_id=actor_user_id, ip=ip)
        return JSONResponse({"ok": True})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": True})
        print("? Error chat_ocultar:", repr(e))
        return JSONResponse({"error": "No se pudo ocultar el hilo"}, status_code=500)


@app.post("/chat/restaurar")
async def chat_restaurar(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    con = (data or {}).get("con")
    if not con:
        return JSONResponse({"error": "Falta 'con' (email del contacto)"}, status_code=400)

    yo = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)

    try:
        restaurar_hilo(owner_email=yo, otro_email=con, actor_user_id=actor_user_id, ip=ip)
        return JSONResponse({"ok": True})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": True})
        print("? Error chat_restaurar:", repr(e))
        return JSONResponse({"error": "No se pudo restaurar el hilo"}, status_code=500)


@app.post("/chat/abrir")
async def chat_abrir(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    con = (data or {}).get("con")
    if not con:
        return JSONResponse({"error": "Falta 'con' (email del contacto)"}, status_code=400)

    yo = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)

    try:
        restaurar_hilo(owner_email=yo, otro_email=con, actor_user_id=actor_user_id, ip=ip)
        return JSONResponse({"ok": True})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": True})
        print("? Error chat_abrir:", repr(e))
        return JSONResponse({"error": "No se pudo abrir el hilo"}, status_code=500)
# =========================
# main.py — PARTE 5 / 6
# (Auditoría, Admin, endpoints legacy + **Incidencias (vista GET)**)
# =========================

# ---------- Vista mínima de Incidencias (evita 404 al hacer clic en el botón) ----------
@app.get("/incidencias", response_class=HTMLResponse)
async def incidencias_view(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")

    # Intentamos renderizar templates/incidencias.html si existe; si no, devolvemos un HTML simple
    try:
        # forzamos carga para verificar existencia; si no existe, get_template lanza excepción
        templates.env.get_template("incidencias.html")
        return templates.TemplateResponse("incidencias.html", {"request": request})
    except Exception:
        html = """<!doctype html><html><head>
        <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Incidencias</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
        </head><body class="p-3">
          <div class="container">
            <h1 class="h4 mb-3">Incidencias</h1>
            <div class="alert alert-info">
              La vista <code>templates/incidencias.html</code> no existe aún. <br>
              Creala para personalizar el módulo. Mientras tanto, esta vista placeholder evita el 404.
            </div>
            <a class="btn btn-secondary" href="/">Volver al inicio</a>
          </div>
        </body></html>"""
        return HTMLResponse(html)

# Aceptar barra final y algunos alias comunes
@app.get("/incidencias/")
async def incidencias_trailing():
    return RedirectResponse("/incidencias", status_code=307)

# ================== Auditoría (vista audit_logs) ==================
@app.get("/auditoria", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def ver_auditoria(request: Request):
    logs = obtener_auditoria()
    return templates.TemplateResponse("auditoria.html", {"request": request, "logs": logs})


@app.post("/auditoria/eliminar", dependencies=[Depends(require_admin)])
async def auditoria_eliminar_disabled(request: Request):
    return JSONResponse({"error": "Operación no permitida: la auditoría es inmutable"}, status_code=405)


@app.post("/auditoria/eliminar-masivo", dependencies=[Depends(require_admin)])
async def auditoria_eliminar_masivo_disabled(request: Request):
    return JSONResponse({"error": "Operación no permitida: la auditoría es inmutable"}, status_code=405)


@app.post("/auditoria/purgar", dependencies=[Depends(require_admin)])
async def auditoria_purgar_disabled(request: Request):
    return JSONResponse({"error": "Operación no permitida: la auditoría es inmutable"}, status_code=405)


# ========= Panel de Administración =========
@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def admin_panel(request: Request):
    # Renderiza el panel de administración (botón del topbar apunta aquí)
    return templates.TemplateResponse("admin.html", {"request": request})


# ========= Admin: API de usuarios (para admin.html) =========
DEFAULT_NEW_USER_PASSWORD = os.getenv("DEFAULT_NEW_USER_PASSWORD", "1234")


class AdminUserCreate(BaseModel):
    nombre: str
    email: EmailStr
    rol: str = "usuario"  # acepta "Administrador"/"Usuario"


class AdminPasswordIn(BaseModel):
    email: EmailStr
    password: str


class AdminToggleIn(BaseModel):
    email: EmailStr
    activo: bool


class AdminRoleIn(BaseModel):
    email: EmailStr
    rol: str  # "admin" | "usuario" | "Administrador" | "Usuario" | "borrado"


@app.get("/api/admin/users")
async def admin_users_list(request: Request, q: str = "", limit: int = 500):
    require_admin(request)
    try:
        raw = listar_usuarios() or []
    except Exception as e:
        print("? admin_users_list/listar_usuarios:", repr(e))
        return {"items": []}

    items = [_user_row_to_dict(u) for u in raw]
    q = (q or "").strip().lower()
    if q:
        items = [u for u in items if q in (u["email"] + " " + (u["nombre"] or "").lower())]
    return {"items": items[:limit]}


# alias de compatibilidad
@app.get("/api/usuarios/list")
async def admin_users_list_alias(request: Request, q: str = "", limit: int = 500):
    return await admin_users_list(request, q=q, limit=limit)


@app.post("/api/admin/users")
async def admin_users_create(request: Request, payload: AdminUserCreate):
    require_admin(request)
    actor_user_id, ip = _actor_info(request)

    email = payload.email.lower()
    # Bloquea sólo si EXISTE y está ACTIVO. Si existe inactivo, se permitirá re-crear (restaurar).
    row = obtener_usuario_por_email(email)  # (id, nombre, email, password, rol, activo)
    if row and bool(row[5]):  # activo = 1
        return JSONResponse({"error": "El email ya existe"}, status_code=409)

    try:
        user_id = agregar_usuario(
            nombre=payload.nombre.strip(),
            email=email,
            password=DEFAULT_NEW_USER_PASSWORD,
            rol=_norm_rol(payload.rol),
            actor_user_id=actor_user_id,
            ip=ip,
        )
        if user_id:
            restored = bool(row and not bool(row[5]))
            return {"ok": True, "restaurado": restored}
        return JSONResponse({"error": "No se pudo crear/restaurar el usuario"}, status_code=500)
    except Exception as e:
        print("? admin_users_create:", repr(e))
        return JSONResponse({"error": "No se pudo crear/restaurar el usuario"}, status_code=500)


# alias de compatibilidad
@app.post("/api/usuarios/crear")
async def admin_users_create_alias(request: Request, payload: AdminUserCreate):
    return await admin_users_create(request, payload)


@app.post("/api/admin/users/password")
async def admin_users_password(request: Request, payload: AdminPasswordIn):
    require_admin(request)
    actor_user_id, ip = _actor_info(request)
    try:
        actualizar_password(payload.email.lower(), payload.password, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("? admin_users_password:", repr(e))
        return JSONResponse({"error": "No se pudo actualizar la contraseña"}, status_code=500)


@app.post("/api/admin/users/toggle")
async def admin_users_toggle(request: Request, payload: AdminToggleIn):
    require_admin(request)
    actor_user_id, ip = _actor_info(request)
    try:
        cambiar_estado_usuario(payload.email.lower(), 1 if payload.activo else 0, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("? admin_users_toggle:", repr(e))
        return JSONResponse({"error": "No se pudo cambiar el estado"}, status_code=500)


@app.post("/api/admin/users/role")
async def admin_users_role(request: Request, payload: AdminRoleIn):
    require_admin(request)
    actor_user_id, ip = _actor_info(request)
    try:
        ok = cambiar_rol(payload.email.lower(), _norm_rol(payload.rol), actor_user_id=actor_user_id, ip=ip)
        if not ok:
            return JSONResponse({"error": "Usuario no encontrado"}, status_code=404)
        return {"ok": True}
    except Exception as e:
        print("? admin_users_role:", repr(e))
        return JSONResponse({"error": "No se pudo cambiar el rol"}, status_code=500)


@app.delete("/api/admin/users/{email:path}")
async def admin_users_delete(request: Request, email: str, hard: bool = Query(default=False)):
    require_admin(request)
    actor_user_id, ip = _actor_info(request)
    try:
        borrar_usuario(email.lower(), actor_user_id=actor_user_id, ip=ip, soft=(not hard))
        return {"ok": True}
    except Exception as e:
        print("? admin_users_delete:", repr(e))
        return JSONResponse({"error": "No se pudo eliminar el usuario"}, status_code=500)


# ======= LEGACY/COMPAT: endpoints antiguos que usa admin.html =======
async def _json_or_form(request: Request) -> dict:
    """Acepta JSON o form-data/x-www-form-urlencoded."""
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        try:
            return await request.json()
        except Exception:
            return {}
    try:
        form = await request.form()
        return {k: (str(v) if v is not None else "") for k, v in form.items()}
    except Exception:
        return {}


@app.get("/admin/usuarios")
async def legacy_admin_users(request: Request, q: str = "", limit: int = 500):
    # Alias de /api/admin/users
    return await admin_users_list(request, q=q, limit=limit)


@app.post("/admin/crear-usuario")
async def legacy_admin_create_user(request: Request):
    # Alias de /api/admin/users (POST)
    data = await _json_or_form(request)
    payload = AdminUserCreate(
        nombre=(data.get("nombre") or "").strip(),
        email=(data.get("email") or "").strip(),
        rol=(data.get("rol") or "usuario").strip(),
    )
    return await admin_users_create(request, payload)


@app.post("/admin/usuarios/password")
@app.post("/admin/blanquear-password")
async def legacy_admin_password(request: Request):
    data = await _json_or_form(request)
    # Si no llega contraseña, usar DEFAULT_NEW_USER_PASSWORD
    new_pwd = (data.get("password") or data.get("nueva") or data.get("new") or "").strip() or DEFAULT_NEW_USER_PASSWORD
    payload = AdminPasswordIn(
        email=(data.get("email") or "").strip(),
        password=new_pwd,
    )
    return await admin_users_password(request, payload)


@app.post("/admin/usuarios/toggle")
async def legacy_admin_toggle(request: Request):
    data = await _json_or_form(request)
    activo_raw = str(data.get("activo", "")).lower()
    activo = activo_raw in ("1", "true", "t", "yes", "on", "si", "sí")
    payload = AdminToggleIn(
        email=(data.get("email") or "").strip(),
        activo=activo,
    )
    return await admin_users_toggle(request, payload)


@app.post("/admin/usuarios/rol")
async def legacy_admin_role(request: Request):
    data = await _json_or_form(request)
    payload = AdminRoleIn(
        email=(data.get("email") or "").strip(),
        rol=(data.get("rol") or "").strip(),
    )
    return await admin_users_role(request, payload)


@app.delete("/admin/usuarios/{email:path}")
async def legacy_admin_delete(request: Request, email: str, hard: bool = Query(default=False)):
    # Alias directo de /api/admin/users/{email}
    return await admin_users_delete(request, email=email, hard=hard)


@app.post("/admin/eliminar-usuario")
async def legacy_admin_delete_post(request: Request):
    # Variante POST por si el front la usa con form-data
    data = await _json_or_form(request)
    email = (data.get("email") or "").strip()
    hard_raw = str(data.get("hard", "")).lower()
    hard = hard_raw in ("1", "true", "t", "yes", "on", "si", "sí")
    return await admin_users_delete(request, email=email, hard=hard)


@app.post("/admin/desactivar-usuario", dependencies=[Depends(require_admin)])
async def legacy_admin_disable_user(request: Request):
    data = await _json_or_form(request)
    email = (data.get("email") or "").strip().lower()
    actor_user_id, ip = _actor_info(request)
    try:
        cambiar_estado_usuario(email, 0, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("? legacy_admin_disable_user:", repr(e))
        return JSONResponse({"error": "No se pudo desactivar"}, status_code=500)


@app.post("/admin/activar-usuario", dependencies=[Depends(require_admin)])
async def legacy_admin_enable_user(request: Request):
    data = await _json_or_form(request)
    email = (data.get("email") or "").strip().lower()
    actor_user_id, ip = _actor_info(request)
    try:
        cambiar_estado_usuario(email, 1, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("? legacy_admin_enable_user:", repr(e))
        return JSONResponse({"error": "No se pudo activar"}, status_code=500)


@app.post("/admin/reset-sesion", dependencies=[Depends(require_admin)])
async def legacy_admin_reset_session(request: Request):
    data = await _json_or_form(request)
    email = (data.get("email") or "").strip().lower()
    now = now_iso_utc()
    try:
        with cal_conn() as c:
            c.execute(
                "UPDATE sessions SET logout_at=?, closed_reason=? WHERE user=? AND logout_at IS NULL",
                (now, "admin-reset", email),
            )
        return {"ok": True}
    except Exception as e:
        print("? legacy_admin_reset_session:", repr(e))
        return JSONResponse({"error": "No se pudo reiniciar la sesión"}, status_code=500)
# =========================
# main.py — PARTE 6 / 6
# (Calendario, Notificaciones, Presencia/Online, Auditoría de actividad + CSV,
#  KB UI/APIs, diag rutas y fallbacks raíz/login/health)
# =========================

# --- Compat: define logger y partial si no estaban definidos en partes previas ---
try:
    logger  # type: ignore[name-defined]
except NameError:
    import logging
    logger = logging.getLogger("app")
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(_h)
    logger.setLevel(logging.INFO)

try:
    partial  # type: ignore[name-defined]
except NameError:
    from functools import partial  # usado en PARTE 2

# =====================================================================
# ========================== CALENDARIO (endpoints) ===================
# =====================================================================

@app.get("/calendario", response_class=HTMLResponse)
async def calendario_view(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("calendario.html", {"request": request})


@app.get("/calendario/eventos")
async def cal_list():
    with cal_conn() as c:
        cur = c.execute("SELECT * FROM eventos ORDER BY start ASC")
        rows = [_event_row_to_dict(r) for r in cur.fetchall()]
        return rows


@app.get("/api/calendar/events")
async def cal_list_alias():
    items = await cal_list()
    return {"events": items}


@app.post("/calendario/eventos")
async def cal_create(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()
    title = (data.get("title") or "").strip()
    start = data.get("start")
    end = data.get("end")
    all_day = 1 if data.get("AllDay") or data.get("allDay") else 0
    desc = (data.get("description") or "").strip()
    color = (data.get("color") or "#0ea5e9").strip()

    if not title or not start:
        return JSONResponse({"error": "Faltan campos: title, start"}, status_code=400)

    evt_id = uuid.uuid4().hex
    now = _now_iso()
    created_by = request.session.get("usuario", "Desconocido")

    with cal_conn() as c:
        c.execute(
            """
            INSERT INTO eventos(id,title,description,start,end,all_day,color,created_by,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (evt_id, title, desc, start, end, all_day, color, created_by, now, now),
        )

    await notify_async(created_by, "Evento creado", f"{title} • {start}{(' – '+end) if end else ''}")

    return {
        "id": evt_id,
        "title": title,
        "description": desc,
        "start": start,
        "end": end,
        "allDay": bool(all_day),
        "color": color,
    }


@app.patch("/calendario/eventos/{evt_id}")
async def cal_update(evt_id: str, request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    data = await request.json()

    def to_iso(v):
        if v is None:
            return None
        return v if isinstance(v, str) else str(v)

    title = data.get("title")
    desc = data.get("description")
    color = data.get("color")
    start = to_iso(data.get("start"))
    end = to_iso(data.get("end"))
    all_day = data.get("AllDay") if "AllDay" in data else data.get("allDay")

    sets, vals = [], []
    if title is not None:
        sets.append("title=?")
        vals.append(title)
    if desc is not None:
        sets.append("description=?")
        vals.append(desc)
    if color is not None:
        sets.append("color=?")
        vals.append(color)
    if start is not None:
        sets.append("start=?")
        vals.append(start)
    if end is not None:
        sets.append("end=?")
        vals.append(end)
    if all_day is not None:
        sets.append("all_day=?")
        vals.append(1 if all_day else 0)

    sets.append("updated_at=?")
    vals.append(_now_iso())
    vals.append(evt_id)

    if len(sets) == 1:
        return JSONResponse({"error": "Nada para actualizar"}, status_code=400)

    with cal_conn() as c:
        cur = c.execute(f"UPDATE eventos SET {', '.join(sets)} WHERE id=?", vals)
        if cur.rowcount == 0:
            return JSONResponse({"error": "Evento no encontrado"}, status_code=404)

    await notify_async(request.session.get("usuario", "Desconocido"), "Evento actualizado", f"ID: {evt_id}")
    return {"ok": True}


@app.delete("/calendario/eventos/{evt_id}")
async def cal_delete(evt_id: str, request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    with cal_conn() as c:
        cur = c.execute("DELETE FROM eventos WHERE id=?", (evt_id,))
        if cur.rowcount == 0:
            return JSONResponse({"error": "Evento no encontrado"}, status_code=404)

    await notify_async(request.session.get("usuario", "Desconocido"), "Evento eliminado", f"ID: {evt_id}")
    return {"ok": True}


# =====================================================================
# ========================== NOTIFICACIONES ============================
# =====================================================================

@app.get("/notificaciones")
async def notificaciones(
    request: Request,
    q: Optional[str] = Query(default=None),
    only_unread: Optional[bool] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    Si el request espera HTML -> renderiza notificaciones.html
    Caso contrario -> devuelve JSON con filtros (q, only_unread, limit, offset)
    """
    user = request.session.get("usuario", "Desconocido")
    if _wants_html(request) and offset == 0 and q is None and only_unread is None:
        return templates.TemplateResponse("notificaciones.html", {"request": request})

    q_like = f"%{(q or '').strip()}%"
    where = ["user=?"]
    args: List[object] = [user]

    if q and q.strip():
        where.append("(LOWER(titulo) LIKE LOWER(?) OR LOWER(cuerpo) LIKE LOWER(?))")
        args += [q_like, q_like]
    if only_unread:
        where.append("leida=0")

    where_sql = " AND ".join(where)

    with cal_conn() as c:
        total_unread = c.execute(
            "SELECT COUNT(1) FROM notificaciones WHERE user=? AND leida=0",
            (user,),
        ).fetchone()[0]

        sql = f"""
        SELECT id, titulo, cuerpo, created_at, leida
        FROM notificaciones
        WHERE {where_sql}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """
        args_sql = args + [limit, offset]
        cur = c.execute(sql, tuple(args_sql))
        items = [
            {
                "id": r["id"],
                "titulo": r["titulo"],
                "cuerpo": r["cuerpo"],
                "fecha_legible": iso_utc_to_ar_str(r["created_at"]),
                "leida": bool(r["leida"]),
            }
            for r in cur.fetchall()
        ]

    return {"total_unread": total_unread, "items": items}


@app.get("/notificaciones/vista", response_class=HTMLResponse)
async def notificaciones_vista(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("notificaciones.html", {"request": request})


# Aliases/redirects para que la campana y "Ver todas" SIEMPRE abran la vista HTML
@app.get("/notificaciones/panel")
@app.get("/notificaciones/todas")
@app.get("/notificaciones/")
@app.get("/notifications")
def notificaciones_redirect():
    return RedirectResponse("/notificaciones/vista", status_code=307)


# Alias directo que a veces usan los enlaces del front
@app.get("/notificaciones/ui", response_class=HTMLResponse)
async def notificaciones_ui_alias(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("notificaciones.html", {"request": request})


@app.post("/notificaciones/marcar-leidas")
async def mark_read(request: Request):
    user = request.session.get("usuario", "Desconocido")
    try:
        data = await request.json()
    except Exception:
        data = {}
    ids = data.get("ids")

    with cal_conn() as c:
        if isinstance(ids, list) and ids:
            placeholders = ",".join("?" for _ in ids)
            c.execute(
                f"UPDATE notificaciones SET leida=1 WHERE user=? AND id IN ({placeholders})",
                (user, *ids),
            )
        else:
            c.execute("UPDATE notificaciones SET leida=1 WHERE user=?", (user,))

    return {"ok": True}


@app.post("/notificaciones/eliminar")
async def notif_delete(request: Request):
    user = request.session.get("usuario", "Desconocido")
    data = await request.json()
    notif_id = int(data.get("id", 0))
    if not notif_id:
        return JSONResponse({"error": "Falta id"}, status_code=400)

    with cal_conn() as c:
        c.execute("DELETE FROM notificaciones WHERE id=? AND user=?", (notif_id, user))
    return {"ok": True}


# =====================================================================
# ========================== PRESENCIA / ONLINE =======================
# =====================================================================

SESSION_TIMEOUT_MIN = 10


def init_presence_db():
    with cal_conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS presence(
                user TEXT PRIMARY KEY,
                nombre TEXT,
                last_seen TEXT NOT NULL,
                ip TEXT,
                ua TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions(
                id TEXT PRIMARY KEY,
                user TEXT NOT NULL,
                nombre TEXT,
                ip TEXT,
                ua TEXT,
                login_at TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                logout_at TEXT,
                closed_reason TEXT
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sessions_dates ON sessions(login_at, last_seen, logout_at)")


init_presence_db()


@app.post("/presence/ping")
async def presence_ping(request: Request):
    email = request.session.get("usuario")
    if not email:
        return JSONResponse({"ok": False, "error": "No autenticado"}, status_code=401)

    row = obtener_usuario_por_email(email)
    nombre = row[1] if row else email
    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")
    now = now_iso_utc()
    sid = request.session.get("sid")

    with cal_conn() as c:
        c.execute(
            """
            INSERT INTO presence(user, nombre, last_seen, ip, ua)
            VALUES(?,?,?,?,?)
            ON CONFLICT(user) DO UPDATE SET
                nombre=excluded.nombre,
                last_seen=excluded.last_seen,
                ip=excluded.ip,
                ua=excluded.ua
            """,
            (email, nombre, now, ip, ua),
        )
        if sid:
            c.execute("UPDATE sessions SET last_seen=? WHERE id=?", (now, sid))

    return {"ok": True}


@app.get("/presence/online")
async def presence_online(minutes: int = 5):
    threshold_ts = datetime.now(timezone.utc).timestamp() - (minutes * 60)
    items = []
    with cal_conn() as c:
        cur = c.execute("SELECT user, nombre, last_seen, ip, ua FROM presence ORDER BY last_seen DESC")
        for r in cur.fetchall():
            try:
                dt = datetime.strptime(r["last_seen"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
            except Exception:
                ts = 0
            if ts >= threshold_ts:
                items.append(
                    {
                        "email": r["user"],
                        "nombre": r["nombre"] or r["user"],
                        "last_seen": r["last_seen"],
                        "ip": r["ip"] or "",
                        "ua": r["ua"] or "",
                    }
                )
    return {"items": items}


@app.get("/usuarios-activos", response_class=HTMLResponse)
async def usuarios_activos(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    data = await presence_online(minutes=5)
    return templates.TemplateResponse("usuarios_activos.html", {"request": request, "items": data.get("items", [])})


# =====================================================================
# ===================== AUDITORÍA DE ACTIVIDAD (admins) ===============
# =====================================================================

def _parse_iso(ts: Optional[str]):
    if not ts:
        return None
    try:
        if len(ts) == 10:
            return datetime.strptime(ts, "%Y-%m-%d").replace(tzinfo=TZ_AR)
        if ts.endswith("Z"):
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _to_dt(s: Optional[str]):
    if not s:
        return None
    try:
        if s.endswith("Z"):
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(s)
    except Exception:
        return None


@app.get("/auditoria/actividad/vista", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def auditoria_actividad_view(request: Request):
    return templates.TemplateResponse("auditoria_actividad.html", {"request": request})


@app.get("/auditoria-actividad", dependencies=[Depends(require_admin)])
async def auditoria_actividad_legacy():
    return RedirectResponse("/auditoria/actividad/vista", status_code=307)


@app.get("/auditoria/actividad", dependencies=[Depends(require_admin)])
async def auditoria_actividad(
    request: Request,
    usuario: Optional[str] = Query(default=None, description="email exacto o parte"),
    desde: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    hasta: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    limit: int = Query(default=500, ge=1, le=5000),
):
    now = datetime.now(timezone.utc)
    rows_out = []

    q = "SELECT id, user, nombre, ip, ua, login_at, last_seen, logout_at, closed_reason FROM sessions"
    conds, args = [], []

    if usuario:
        conds.append("user LIKE ?")
        args.append(f"%{usuario}%")

    d = _parse_iso(desde)
    h = _parse_iso(hasta)
    if d:
        conds.append("login_at >= ?")
        args.append(d.strftime("%Y-%m-%dT00:00:00Z"))
    if h:
        conds.append("login_at <= ?")
        args.append(h.strftime("%Y-%m-%dT23:59:59Z"))

    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY login_at DESC LIMIT ?"
    args.append(limit)

    with cal_conn() as c:
        cur = c.execute(q, tuple(args))
        for r in cur.fetchall():
            login_dt = _to_dt(r["login_at"])
            last_dt = _to_dt(r["last_seen"])
            logout_dt = _to_dt(r["logout_at"])

            if logout_dt:
                estado = "cerrada"
                ref_end = logout_dt
            else:
                if last_dt and (now - last_dt).total_seconds() <= SESSION_TIMEOUT_MIN * 60:
                    estado = "activa"
                else:
                    estado = "expirada"
                ref_end = last_dt or now

            dur_sec = None
            if login_dt and ref_end:
                dur_sec = int(max(0, (ref_end - login_dt).total_seconds()))

            rows_out.append(
                {
                    "id": r["id"],
                    "usuario": r["user"],
                    "nombre": r["nombre"] or r["user"],
                    "ip": r["ip"] or "",
                    "ua": r["ua"] or "",
                    "login_at": r["login_at"],
                    "last_seen": r["last_seen"],
                    "logout_at": r["logout_at"],
                    "estado": estado,
                    "closed_reason": r["closed_reason"] or "",
                    "duracion_seg": dur_sec,
                }
            )

    return {"items": rows_out, "timeout_min": SESSION_TIMEOUT_MIN, "now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}


@app.get("/auditoria/actividad.csv", dependencies=[Depends(require_admin)])
async def auditoria_actividad_csv(
    request: Request,
    usuario: Optional[str] = Query(default=None),
    desde: Optional[str] = Query(default=None),
    hasta: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
):
    data = await auditoria_actividad(request, usuario=usuario, desde=desde, hasta=hasta, limit=limit)
    items = data.get("items", [])

    headers = ["estado", "usuario", "nombre", "login_at", "last_seen", "logout_at", "duracion_seg", "ip", "ua", "sid", "closed_reason"]
    lines = [",".join(headers)]

    for it in items:
        row = [
            it.get("estado", ""),
            it.get("usuario", ""),
            it.get("nombre", ""),
            it.get("login_at", ""),
            it.get("last_seen", ""),
            it.get("logout_at", "") or "",
            str(it.get("duracion_seg") or 0),
            it.get("ip", ""),
            (it.get("ua", "") or "").replace(",", " "),
            it.get("id", ""),
            it.get("closed_reason", "").replace(",", " "),
        ]
        lines.append(",".join(row))

    csv_body = "\n".join(lines)
    filename = "auditoria_actividad.csv"
    return Response(
        content=csv_body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

# =========================
# KB — UI mínima + APIs
# =========================

# Vista HTML (solo admins)
@app.get("/kb", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def kb_view(request: Request):
    return templates.TemplateResponse(
        "kb.html",
        {
            "request": request,
            "kb_enabled": _kb_enabled(),
            "allowed_ext": sorted([e.lstrip(".").lower() for e in KB_ALLOWED_EXT]),
        },
    )


# ---- APIs simples ----

@app.get("/api/kb/sources", dependencies=[Depends(require_admin)])
async def kb_sources():
    """
    Lista fuentes/rubros disponibles. Si utils.kb_list_sources existe, la usa.
    Caso contrario, escanea carpetas en storage/kb.
    """
    f = _kb_funcs()
    items = []
    if callable(f["list_sources"]):
        try:
            items = f["list_sources"]() or []
        except Exception as e:
            print("kb_list_sources error:", repr(e))

    if not items:
        base = KB_STORAGE_DIR
        try:
            sources = []
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name)
                if os.path.isdir(p):
                    sources.append({"name": name, "slug": name, "path": p})
            items = sources
        except Exception as e:
            print("kb fs scan error:", repr(e))
            items = []

    return {"items": items}


@app.post("/api/kb/source", dependencies=[Depends(require_admin)])
async def kb_source_create(name: str = Form(...)):
    """
    Crea (o garantiza) la carpeta para un rubro/fuente de KB y llama a utils.kb_create_or_get_source si existe.
    Maneja múltiples firmas posibles SIN kwargs problemáticos.
    """
    f = _kb_funcs()
    name = (name or "").strip()
    if not name:
        return JSONResponse({"error": "Nombre requerido"}, status_code=400)
    slug = _kb_slugify(name)

    dst_dir = os.path.join(KB_STORAGE_DIR, slug)
    os.makedirs(dst_dir, exist_ok=True)

    cog = f["create_or_get_source"]
    if callable(cog):
        try:
            with kb_session() as db:
                called = False
                if db is not None:
                    # (db, name, slug) -> (db, name) -> (db, slug) -> (db, name, meta)
                    try:
                        cog(db, name, slug)
                        called = True
                    except TypeError:
                        try:
                            cog(db, name)
                            called = True
                        except TypeError:
                            try:
                                cog(db, slug)
                                called = True
                            except TypeError:
                                try:
                                    cog(db, name, {"slug": slug})
                                    called = True
                                except TypeError:
                                    pass
                if not called:
                    # (name, slug) -> (name) -> (slug) -> (name, meta)
                    try:
                        cog(name, slug)
                        called = True
                    except TypeError:
                        try:
                            cog(name)
                            called = True
                        except TypeError:
                            try:
                                cog(slug)
                                called = True
                            except TypeError:
                                try:
                                    cog(name, {"slug": slug})
                                    called = True
                                except TypeError:
                                    pass
        except Exception as e:
            print("kb_create_or_get_source error:", repr(e))

    return {"ok": True, "slug": slug}


@app.post("/api/kb/upload", dependencies=[Depends(require_admin)])
async def kb_upload(
    source: str = Form(...),
    files: List[UploadFile] = File(...),
):
    slug = _kb_slugify(source or "")
    if not slug:
        return JSONResponse({"error": "Falta 'source' (rubro)"}, status_code=400)
    if not files:
        return JSONResponse({"error": "Subí al menos un archivo"}, status_code=400)

    dst_dir = os.path.join(KB_STORAGE_DIR, slug)
    os.makedirs(dst_dir, exist_ok=True)

    saved = []
    funcs = _kb_funcs()
    for fup in files:
        if not fup or not fup.filename:
            continue
        orig = fup.filename
        ext = os.path.splitext(orig)[1].lower()
        if ext not in KB_ALLOWED_EXT:
            return JSONResponse({"error": f"Extensión no permitida: {ext}"}, status_code=400)

        safe = _safe_basename(orig) + ext
        path = os.path.join(dst_dir, safe)
        written = await _save_upload_stream(fup, path)
        saved.append({"file": safe, "bytes": written})

        # ---- Ingesta con firma flexible (sin kwargs polémicos)
        ingester = funcs["ingest_file"]
        if callable(ingester):
            try:
                with kb_session() as db:
                    called = False
                    if db is not None:
                        try:
                            ingester(db, slug, path, {"original_name": orig})
                            called = True
                        except TypeError:
                            try:
                                ingester(db, slug, path)
                                called = True
                            except TypeError:
                                pass
                    if not called:
                        try:
                            ingester(slug, path, {"original_name": orig})
                            called = True
                        except TypeError:
                            try:
                                ingester(slug, path)
                                called = True
                            except TypeError:
                                pass
            except Exception as e:
                print("kb_ingest_file error:", repr(e))

    return {"ok": True, "saved": saved, "source": slug}


@app.get("/api/kb/priorities", dependencies=[Depends(require_admin)])
async def kb_priorities():
    """
    Lista de prioridades (si utils.kb_list_priorities existe).
    """
    f = _kb_funcs()
    if callable(f["list_priorities"]):
        try:
            return {"items": f["list_priorities"]() or []}
        except Exception as e:
            print("kb_list_priorities error:", repr(e))
            return {"items": []}
    return {"items": []}


class KBPriorityIn(BaseModel):
    term: str
    weight: int = 1
    source: Optional[str] = None

@app.post("/api/kb/priorities", dependencies=[Depends(require_admin)])
async def kb_priorities_upsert(payload: KBPriorityIn):
    f = _kb_funcs()
    up = f.get("upsert_priority")

    term = (payload.term or "").strip()
    if not term:
        return JSONResponse({"error": "Término requerido"}, status_code=400)
    weight = int(payload.weight or 1)
    source = (payload.source or "").strip() or None

    # 1) Intentos vía utils (sin y con DB)
    if callable(up):
        try:
            try:
                up(term, weight, source)   # (term, weight, source)
                return {"ok": True}
            except TypeError:
                pass
            try:
                up(term, weight)           # (term, weight)
                return {"ok": True}
            except TypeError:
                pass

            db = None
            try:
                db = SessionLocal()
                try:
                    up(db, term, weight, source)  # (db, term, weight, source)
                    db.commit()
                    return {"ok": True}
                except TypeError:
                    pass
                try:
                    up(db, term, weight)          # (db, term, weight)
                    db.commit()
                    return {"ok": True}
                except TypeError:
                    pass
            finally:
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

        except Exception as e:
            # Error típico: "attempted relative import with no known parent package"
            msg = str(e)
            if "attempted relative import" not in msg and "No module named" not in msg:
                # otro error real: lo exponemos
                return JSONResponse({"error": f"No se pudo guardar: {e}"}, status_code=500)
            # si fue error de import relativo, caemos al fallback local
            print("· kb_upsert_priority via utils falló por import relativo; usando fallback:", repr(e))

    # 2) Fallback local contra models.KBPriority
    ok, err = _fallback_kb_upsert_priority(term, weight, source)
    if ok:
        return {"ok": True}
    return JSONResponse({"error": f"No se pudo guardar: {err}"}, status_code=500)

# (Opcional) diagnóstico de rutas
@app.get("/__diag/routes")
def _diag_routes():
    return {"routes": sorted({getattr(r, "path", "") for r in app.routes})}

def _fallback_kb_upsert_priority(term: str, weight: int, source: Optional[str] = None):
    """
    Upsert sin importar `models.py`: refleja las tablas y busca una que tenga
    columnas (term|pattern|patron|keyword|texto) y (weight|peso|score|priority)
    y opcionalmente (source|fuente|rubro|category).
    """
    try:
        from sqlalchemy import inspect as sa_inspect, MetaData, Table, select, update, insert, and_, null
        with SessionLocal() as s:
            engine = s.get_bind()
            insp = sa_inspect(engine)

            all_tables = insp.get_table_names()
            # Preferencias de nombre, pero si no están, probamos cualquier tabla compatible
            preferred = ["kb_priorities", "kbpriority", "kb_priority", "priorities", "priority"] + all_tables

            term_cands   = ["term", "pattern", "patron", "keyword", "texto"]
            weight_cands = ["weight", "peso", "score", "priority"]
            source_cands = ["source", "fuente", "rubro", "category"]

            chosen = None
            chosen_cols = None
            for name in preferred:
                if name not in all_tables:
                    continue
                cols = [c["name"] for c in insp.get_columns(name)]
                if any(c in cols for c in term_cands) and any(c in cols for c in weight_cands):
                    chosen, chosen_cols = name, cols
                    break

            if not chosen:
                return False, f"No encontré tabla compatible (disponibles: {all_tables})."

            md = MetaData()
            T = Table(chosen, md, autoload_with=engine)

            def pick(cands):
                for c in cands:
                    if c in T.c:
                        return c
                return None

            term_col   = pick(term_cands)
            weight_col = pick(weight_cands)
            source_col = pick(source_cands)

            if not term_col or not weight_col:
                return False, f"Faltan columnas requeridas en '{chosen}'. Tiene: {list(T.c.keys())}"

            where = [T.c[term_col] == term]
            if source_col:
                if source is None:
                    where.append(T.c[source_col].is_(None))
                else:
                    where.append(T.c[source_col] == source)

            with engine.begin() as conn:
                row = conn.execute(select(T).where(and_(*where))).first()
                if row:
                    vals = {weight_col: int(weight)}
                    if source_col:
                        vals[source_col] = source
                    conn.execute(update(T).where(and_(*where)).values(**vals))
                else:
                    vals = {term_col: term, weight_col: int(weight)}
                    if source_col:
                        vals[source_col] = source
                    conn.execute(insert(T).values(**vals))

        return True, None
    except Exception as e:
        return False, str(e)


# === Fallbacks raíz/login/health para Render (pegar al final del archivo) ===
from starlette.routing import Route

def _route_exists(path: str, method: str = "GET") -> bool:
    try:
        for r in app.routes:
            if isinstance(r, Route) and r.path == path and method.upper() in (r.methods or {"GET"}):
                return True
    except Exception:
        pass
    return False

# Healthcheck de Render: HEAD /
@app.head("/", include_in_schema=False)
def _head_root_ok():
    return Response(status_code=200)

# Raíz de la app: si no existe, crea un fallback que redirige a /login o muestra index.html
if not _route_exists("/", "GET"):
    @app.get("/", include_in_schema=False)
    async def _root_fallback(request: Request):
        if not request.session.get("usuario"):
            return RedirectResponse("/login", status_code=307)
        # Intentar renderizar un index.html si existe; si no, mostrar un HTML simple
        try:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    # valores seguros por si el template los espera
                    "historial_items": [],
                    "page": 1, "per_page": 10, "total_pages": 1, "total_items": 0, "q": "",
                },
            )
        except Exception:
            return HTMLResponse("<h3>Inicio</h3><p>Sistema online.</p>", status_code=200)

# Vista de login por GET: necesaria para las redirecciones a /login
if not _route_exists("/login", "GET"):
    @app.get("/login", include_in_schema=False)
    async def _login_view(request: Request):
        # Si tenés templates/login.html, lo usa; si no, muestra un form mínimo
        try:
            return templates.TemplateResponse("login.html", {"request": request, "error": None, "mensaje": None})
        except Exception:
            html = """<!doctype html><meta charset="utf-8">
            <title>Login</title>
            <form method="post" action="/login" style="max-width:320px;margin:48px auto;font-family:sans-serif">
                <h3>Ingresar</h3>
                <div><input name="email" placeholder="email" style="width:100%;padding:8px;margin:6px 0"></div>
                <div><input name="password" type="password" placeholder="contraseña" style="width:100%;padding:8px;margin:6px 0"></div>
                <label><input type="checkbox" name="remember"> Recordarme</label>
                <div><button style="padding:8px 12px;margin-top:8px">Entrar</button></div>
            </form>"""
            return HTMLResponse(html, status_code=200)

# Alias con barra final
@app.get("/login/", include_in_schema=False)
def _login_trailing():
    return RedirectResponse("/login", status_code=307)

# Endpoint simple de health (útil para pruebas manuales)
@app.get("/healthz", include_in_schema=False)
def _healthz():
    return {"ok": True, "ts": now_iso_utc()}
