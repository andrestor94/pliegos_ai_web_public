# =========================
# main.py — PARTE 1 / 5
# (imports, helpers base, app init, WS, utils)
# =========================

import os
import sqlite3
import uuid
import asyncio
import re
import json
from typing import List, Optional, Dict, Set
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Body, WebSocket, WebSocketDisconnect, Depends, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.concurrency import run_in_threadpool
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import or_
from pydantic import BaseModel, EmailStr
from jinja2 import ChoiceLoader, FileSystemLoader  # ⭐ nuevo

from utils import (
    extraer_texto_de_pdf,
    analizar_con_openai,
    generar_pdf_con_plantilla,
    responder_chat_openai,
    analizar_anexos
)

from database import (
    DB_PATH,
    inicializar_bd,
    obtener_usuario_por_email, agregar_usuario, listar_usuarios,
    actualizar_password, cambiar_estado_usuario, borrar_usuario, cambiar_rol,
    buscar_usuarios,
    guardar_en_historial, obtener_historial, eliminar_del_historial,
    obtener_historial_completo,
    crear_ticket, obtener_todos_los_tickets, obtener_tickets_por_usuario,
    actualizar_estado_ticket, eliminar_ticket,
    obtener_auditoria,
    enviar_mensaje, obtener_hilos_para, obtener_mensajes_entre,
    marcar_mensajes_leidos, contar_no_leidos,
    ocultar_hilo, restaurar_hilo,
    guardar_adjunto,
    es_admin,
    iniciar_analisis_historial, marcar_valoracion_historial, tiene_valoracion_pendiente
)

# ORM (audit_logs)
from db_orm import inicializar_bd_orm, SessionLocal, AuditLog

# ================== TZ & helpers ==================
TZ_AR = ZoneInfo("America/Argentina/Buenos_Aires")

def now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def now_stamp_ar() -> str:
    return datetime.now(TZ_AR).strftime("%Y%m%d%H%M%S")

def iso_utc_to_ar_str(iso_utc: str, fmt: str = "%d/%m/%Y %H:%M") -> str:
    if not iso_utc:
        return ""
    iso = iso_utc.replace("Z", "+00:00")
    try:
        dt_utc = datetime.fromisoformat(iso)
    except ValueError:
        try:
            dt_utc = datetime.strptime(iso_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return iso_utc
    return dt_utc.astimezone(TZ_AR).strftime(fmt)

# --- Normalizador robusto de datetimes a UTC aware (para comparaciones seguras) ---
def _parse_dt_utc(value) -> Optional[datetime]:
    """
    Acepta datetime o str (con o sin 'Z') y devuelve datetime con tz UTC.
    Evita 'TypeError: can't subtract offset-naive and offset-aware datetimes'.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)  # Puede devolver naive si no hay offset
        except Exception:
            # fallback "YYYY-MM-DD HH:MM:SS"
            try:
                dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# ================== App & Middlewares ==================
SESSION_SECRET = os.getenv("SESSION_SECRET", "change-this-in-prod")
app = FastAPI(middleware=[
    # Cookie de sesión más robusta y persistente
    Middleware(SessionMiddleware,
               secret_key=SESSION_SECRET,
               same_site="lax",
               max_age=60*60*24*30)  # 30 días
])

# ---------- Garantizar tablas de chat si faltan (fix 'no such table: mensajes') ----------
def ensure_chat_tables():
    """Crea tablas de chat si no existen en usuarios.db (robustez en Render)."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            # mensajes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mensajes(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    de_email   TEXT NOT NULL,
                    para_email TEXT NOT NULL,
                    texto      TEXT,
                    leido      INTEGER NOT NULL DEFAULT 0,
                    fecha      TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mensajes_para_leido ON mensajes(para_email, leido)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mensajes_de_para ON mensajes(de_email, para_email)")
            # hilos ocultos
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hilos_ocultos(
                    owner_email TEXT NOT NULL,
                    otro_email  TEXT NOT NULL,
                    hidden_at   TEXT NOT NULL,
                    PRIMARY KEY(owner_email, otro_email)
                )
            """)
            # adjuntos
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mensajes_adjuntos(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mensaje_id INTEGER NOT NULL,
                    filename   TEXT NOT NULL,
                    original   TEXT,
                    mime       TEXT,
                    size       INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
    except Exception as e:
        print("⚠️ ensure_chat_tables() no pudo crear tablas:", repr(e))
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

# ---------- Bootstrap de admin si DB está vacía ----------
def ensure_default_admin():
    """
    Si no hay usuarios en la DB, crea un admin inicial.
    Variables de entorno:
      - DEFAULT_ADMIN_EMAIL
      - DEFAULT_ADMIN_NAME
      - DEFAULT_ADMIN_PASSWORD  (fallback a DEFAULT_NEW_USER_PASSWORD o '1234')
    """
    try:
        usuarios = listar_usuarios()
    except Exception:
        usuarios = []
    if usuarios:
        return

    default_email = (os.getenv("DEFAULT_ADMIN_EMAIL", "admin@suizo.com") or "").lower()
    default_name  = os.getenv("DEFAULT_ADMIN_NAME", "Admin")
    default_pwd   = os.getenv("DEFAULT_ADMIN_PASSWORD", os.getenv("DEFAULT_NEW_USER_PASSWORD", "1234"))

    try:
        if not obtener_usuario_por_email(default_email):
            agregar_usuario(
                nombre=default_name,
                email=default_email,
                password=default_pwd,
                rol="admin",
                actor_user_id=None,
                ip=None
            )
            print(f"✅ Admin inicial creado: {default_email}")
    except Exception as e:
        print("⚠️ ensure_default_admin() error:", repr(e))

ensure_default_admin()
# ---------- fin bootstrap ----------

# Static
os.makedirs("static", exist_ok=True)
os.makedirs("generated_pdfs", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated_pdfs", StaticFiles(directory="generated_pdfs"), name="generated_pdfs")

templates = Jinja2Templates(directory="templates")
templates.env.globals['os'] = os

# ⭐ desactivar cache de Jinja y forzar FileSystemLoader (evita plantillas viejas en Render/CDN)
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

# ⭐ No-cache para HTML (evita que el browser/CDN te muestre UI vieja)
@app.middleware("http")
async def _no_cache_html(request, call_next):
    resp = await call_next(request)
    ctype = (resp.headers.get("content-type") or "").lower()
    if "text/html" in ctype:
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

# ================== Guardas/Dependencias de auth/roles ==================
def require_auth(request: Request):
    if not request.session.get("usuario"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")

def require_admin(request: Request):
    email = request.session.get("usuario")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")
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
    True si el cliente espera JSON (útil para que /incidencias/cerrar|eliminar
    respondan {ok:true} cuando vienen por fetch).
    """
    acc = (request.headers.get("accept") or "").lower()
    xrw = (request.headers.get("x-requested-with") or "").lower()
    return ("application/json" in acc) or (xrw in ("fetch", "xmlhttprequest")) or (request.query_params.get("_json") == "1")

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
                if not self._by_user[email]:
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
        "ts": now_iso_utc()
    }
    await manager.send_to_user(para_email, payload)

# ================== Archivos de chat (adjuntos) ==================
CHAT_ATTACH_DIR = os.path.join("static", "chat_adjuntos")
os.makedirs(CHAT_ATTACH_DIR, exist_ok=True)

CHAT_ALLOWED_EXT = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".txt", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".pptx"
}
CHAT_MAX_FILES = 10
CHAT_MAX_TOTAL_MB = 50

# (Se elimina aquí el bloque duplicado de adjuntos de incidencias; la versión definitiva está en PARTE 4)

# ================== Avatares (perfil) ==================
AVATAR_DIR = os.path.join("static", "avatars")
os.makedirs(AVATAR_DIR, exist_ok=True)
AVATAR_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
AVATAR_MAX_MB = 2  # MB

# ================== Helpers ==================
def _actor_info(request: Request):
    email = request.session.get("usuario")
    row = obtener_usuario_por_email(email) if email else None
    actor_user_id = row[0] if row else None
    ip = request.client.host if request.client else None
    return actor_user_id, ip

def _safe_basename(name: str) -> str:
    base = os.path.splitext(name or "archivo")[0]
    base = "".join(c for c in base if c.isalnum() or c in ("-", "_", "."))
    base = base[:50] or "file"
    return base

def _email_safe(s: str) -> str:
    return (s or "anon").replace("@", "_at_").replace(".", "_dot_")

async def _save_upload_stream(upload: UploadFile, dst_path: str) -> int:
    size = 0
    with open(dst_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            f.write(chunk)
    await upload.seek(0)
    return size

def _validate_ext(filename: str):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in CHAT_ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido: {ext}")

def _build_audit_filters(q, filtros):
    if not filtros:
        return q
    acc = (filtros.get("accion") or "").strip()
    d   = (filtros.get("desde")  or "").strip()
    h   = (filtros.get("hasta")  or "").strip()
    term= (filtros.get("term")   or "").strip()

    if acc and hasattr(AuditLog, "accion"):
        q = q.filter(AuditLog.accion == acc)

    if d and hasattr(AuditLog, "fecha"):
        q = q.filter(AuditLog.fecha >= f"{d} 00:00:00")
    if h and hasattr(AuditLog, "fecha"):
        q = q.filter(AuditLog.fecha <= f"{h} 23:59:59")

    if term:
        like = f"%{term}%"
        ors = []
        for col in ("usuario", "nombre", "accion", "entidad", "entidad_id", "ip", "before", "after"):
            if hasattr(AuditLog, col):
                ors.append(getattr(AuditLog, col).like(like))
        if ors:
            q = q.filter(or_(*ors))
    return q

# --- helpers extra para rating/identificación por timestamp/nombre ---
_TS_RE = re.compile(r'(\d{14})')

def _extraer_ts_de_nombre(nombre_pdf: str) -> str:
    if not nombre_pdf:
        return ""
    m = _TS_RE.search(nombre_pdf)
    return m.group(1) if m else ""

def _buscar_historial_usuario(
    user: str,
    timestamp: Optional[str] = None,
    nombre_pdf: Optional[str] = None
) -> Optional[dict]:
    try:
        items = obtener_historial_completo()
    except Exception:
        items = []

    user_items = [h for h in items if (h.get("usuario") == user)]

    nombre_pdf = (nombre_pdf or "").strip()
    timestamp = (timestamp or "").strip()

    if timestamp:
        for h in user_items:
            if str(h.get("timestamp") or "") == timestamp:
                return h

    if nombre_pdf:
        for h in user_items:
            if (h.get("nombre_archivo") or "") == nombre_pdf:
                return h

    if timestamp:
        for h in user_items:
            ts_h = _extraer_ts_de_nombre(h.get("nombre_archivo") or "")
            if ts_h == timestamp:
                return h

    if user_items:
        try:
            user_items.sort(key=lambda x: x.get("fecha",""), reverse=True)
        except Exception:
            pass
        return user_items[0]

    return None

# --- Config pública para el front (límites de adjuntos) ---
@app.get("/chat/config")
async def chat_config():
    return {
        "allowed_ext": sorted(list({e.lstrip(".").lower() for e in CHAT_ALLOWED_EXT})),
        "max_files": CHAT_MAX_FILES,
        "max_total_mb": CHAT_MAX_TOTAL_MB
    }

# ================== Rutas base ==================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "rol": request.session.get("rol", "usuario")
    })

@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    # Si ya estás logueado, te llevo al home.
    if request.session.get("usuario"):
        return RedirectResponse("/", status_code=303)
    # Permite mostrar estado por query (opcional)
    msg = None
    if request.query_params.get("logout") == "1":
        msg = "Sesión cerrada."
    return templates.TemplateResponse("login.html", {"request": request, "error": None, "mensaje": msg})
# =========================
# main.py — PARTE 2 / 5
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
        c.execute("""
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
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS notificaciones(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                titulo TEXT NOT NULL,
                cuerpo TEXT,
                created_at TEXT NOT NULL,
                leida INTEGER NOT NULL DEFAULT 0
            )
        """)
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
            (user or "Desconocido", titulo, cuerpo, _now_iso())
        )

async def notify_async(user: str, titulo: str, cuerpo: str = ""):
    _notify(user, titulo, cuerpo)
    await emit_alert(user, titulo, cuerpo)

# ====== NUEVO: rating pendiente liviano (sidecar) ======
def init_rating_pending_db():
    with cal_conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS pending_ratings(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                historial_id TEXT,
                timestamp TEXT,
                nombre_pdf TEXT,
                created_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_pending_ratings_user ON pending_ratings(user)")
init_rating_pending_db()

def _pr_add(user: str, historial_id: Optional[str], timestamp: str, nombre_pdf: str):
    with cal_conn() as c:
        c.execute("DELETE FROM pending_ratings WHERE user=?", (user,))
        c.execute(
            "INSERT INTO pending_ratings(user, historial_id, timestamp, nombre_pdf, created_at) VALUES(?,?,?,?,?)",
            (user, str(historial_id) if historial_id is not None else None, timestamp, nombre_pdf, _now_iso())
        )

def _pr_get(user: str):
    with cal_conn() as c:
        r = c.execute(
            "SELECT historial_id, timestamp, nombre_pdf FROM pending_ratings WHERE user=? ORDER BY id DESC LIMIT 1",
            (user,)
        ).fetchone()
        if r:
            return {"historial_id": r["historial_id"], "timestamp": r["timestamp"], "nombre_pdf": r["nombre_pdf"]}
    return None

def _pr_clear(user: str):
    with cal_conn() as c:
        c.execute("DELETE FROM pending_ratings WHERE user=?", (user,))


# ================== Login/Logout ==================
@app.post("/login")
async def login(request: Request,
                email: str = Form(...),
                password: str = Form(...),
                remember: Optional[str] = Form(default=None)):  # "on" si tildan Recordarme
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

    usuario = obtener_usuario_por_email(email)

    # usuario tuple esperado: (id, nombre, email, password, rol, activo)
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
        ip_s = request.client.host if request.client else None
        ua_s = request.headers.get("user-agent", "")
        now_iso = now_iso_utc()
        with cal_conn() as c:
            c.execute("""
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
            """)
            c.execute("""
                INSERT INTO sessions(id, user, nombre, ip, ua, login_at, last_seen, logout_at, closed_reason)
                VALUES(?,?,?,?,?,?,?,?,?)
            """, (sid, request.session["usuario"], nombre_s, ip_s, ua_s, now_iso, now_iso, None, None))

        return RedirectResponse("/", status_code=303)

    # Mensajes de error más claros
    if usuario and not is_active:
        err = "Tu usuario está desactivado. Consultá con un administrador."
    else:
        err = "Credenciales incorrectas"

    # Mantener la UX del login (mensajes + status)
    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": err,
        "mensaje": None
    }, status_code=401)


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
    confirmar: str = Form(...)
):
    email = request.session.get("usuario")
    if not email:
        return RedirectResponse("/login", status_code=303)

    if (not nueva) or (nueva != confirmar):
        return templates.TemplateResponse(
            "cambiar_password.html",
            {"request": request, "error": "Las contraseñas no coinciden."},
            status_code=400
        )

    row = obtener_usuario_por_email(email)
    if not row:
        return templates.TemplateResponse(
            "cambiar_password.html",
            {"request": request, "error": "Usuario no encontrado."},
            status_code=404
        )

    # Validación de contraseña actual (según tu esquema plano)
    if str(row[3]) != str(actual):
        return templates.TemplateResponse(
            "cambiar_password.html",
            {"request": request, "error": "La contraseña actual es incorrecta."},
            status_code=400
        )

    actor_user_id, ip = _actor_info(request)
    try:
        actualizar_password(email.lower(), nueva, actor_user_id=actor_user_id, ip=ip)
    except Exception as e:
        print("❌ cambiar_password_post:", repr(e))
        return templates.TemplateResponse(
            "cambiar_password.html",
            {"request": request, "error": "No se pudo actualizar la contraseña."},
            status_code=500
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
    confirmar: str = Form(...)
):
    return await cambiar_password_post(request, actual=actual, nueva=nueva, confirmar=confirmar)

# NUEVO: aceptar guion_bajo en POST (compatibilidad)
@app.post("/cambiar_password")
async def cambiar_password_underscore_post(
    request: Request,
    actual: str = Form(...),
    nueva: str = Form(...),
    confirmar: str = Form(...)
):
    return await cambiar_password_post(request, actual=actual, nueva=nueva, confirmar=confirmar)

# NUEVO: aceptar guion_bajo + barra final en POST
@app.post("/cambiar_password/")
async def cambiar_password_underscore_post_slash(
    request: Request,
    actual: str = Form(...),
    nueva: str = Form(...),
    confirmar: str = Form(...)
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
                "historial_id": hid if isinstance(hid, int) else None
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
            hid = h.get("historial_id") or h.get("id")  # ⭐ fix: or (no bitwise)
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
        print("❌ Error enviar_rating:", repr(e))
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


@app.post("/analizar-pliego")
async def analizar_pliego(request: Request, archivos: List[UploadFile] = File(...)):
    usuario = request.session.get("usuario", "Anónimo")

    try:
        pr = _pr_get(usuario)
        if pr or tiene_valoracion_pendiente(usuario):
            payload = {"error": "Tienes una valoración pendiente. Califica el análisis anterior para continuar."}
            if pr:
                payload["pending"] = True
                payload["last"] = pr
            resp = JSONResponse(payload, status_code=409)
            resp.headers["X-Require-Rating"] = "1"
            return resp
    except Exception as e:
        print("⚠️ Warning al chequear pendiente:", repr(e))

    if not archivos:
        return JSONResponse({"error": "Subí al menos un archivo"}, status_code=400)

    for a in archivos:
        if not a or not a.filename:
            continue
        _validate_ext(a.filename)

    try:
        resumen = await run_in_threadpool(analizar_anexos, archivos)
    except Exception as e:
        return JSONResponse({"error": f"Fallo en el análisis: {e}"}, status_code=500)

    try:
        timestamp = now_stamp_ar()
        nombre_archivo_pdf = f"resumen_{timestamp}.pdf"
        await run_in_threadpool(generar_pdf_con_plantilla, resumen, nombre_archivo_pdf)
    except Exception as e:
        return JSONResponse({"error": f"Fallo al generar PDF: {e}", "resumen": resumen}, status_code=500)

    analisis_id = uuid.uuid4().hex
    try:
        historial_id = iniciar_analisis_historial(
            usuario=usuario,
            nombre_archivo=nombre_archivo_pdf,
            ruta_pdf=nombre_archivo_pdf,
            analisis_id=analisis_id,
            resumen_texto=resumen
        )
    except Exception as e:
        print("❌ Error iniciar_analisis_historial:", repr(e))
        try:
            guardar_en_historial(timestamp, usuario, nombre_archivo_pdf, nombre_archivo_pdf, resumen)
        except Exception:
            pass
        historial_id = None

    try:
        _pr_add(usuario, historial_id, timestamp, nombre_archivo_pdf)
    except Exception as e:
        print("⚠️ No se pudo registrar pending_ratings:", repr(e))

    return {
        "resumen": resumen,
        "pdf": nombre_archivo_pdf,
        "timestamp": timestamp,
        "historial_id": historial_id,
        "analisis_id": analisis_id
    }
# =========================
# main.py — PARTE 3 / 5
# (historial, usuario/avatares, diagnóstico)
# =========================

# ================== Historial ==================
@app.get("/historial")
async def ver_historial():
    return JSONResponse(obtener_historial())

@app.get("/historia")
async def alias_historia():
    return RedirectResponse("/?goto=historial", status_code=307)

@app.get("/analisis")
@app.get("/analisis/nuevo")
@app.get("/report")
async def alias_analisis():
    return RedirectResponse("/?goto=analisis", status_code=307)

@app.get("/descargar/{archivo}")
async def descargar_pdf(archivo: str):
    archivo = os.path.basename(archivo)
    ruta = os.path.join("generated_pdfs", archivo)
    if os.path.exists(ruta) and os.path.isfile(ruta):
        return FileResponse(ruta, media_type='application/pdf', filename=archivo)
    return {"error": "Archivo no encontrado"}

@app.delete("/eliminar/{timestamp}")
async def eliminar_archivo(timestamp: str):
    eliminar_del_historial(timestamp)
    ruta = os.path.join("generated_pdfs", f"resumen_{os.path.basename(timestamp)}.pdf")
    if os.path.exists(ruta):
        os.remove(ruta)
    return {"mensaje": "Eliminado correctamente"}


# ================== Usuario actual ==================
@app.get("/usuario-actual")
async def usuario_actual(request: Request):
    email = request.session.get("usuario", "")
    rol = request.session.get("rol", "usuario")

    row = obtener_usuario_por_email(email) if email else None
    nombre = (row[1] if row else None) or request.session.get("nombre") or (email or "Desconocido")

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
        "avatar_url": avatar_url
    }

# ===== Subir/actualizar avatar =====
@app.post("/perfil/avatar")
async def subir_avatar(request: Request, avatar: UploadFile = File(...)):
    email = request.session.get("usuario")
    if not email:
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    orig = avatar.filename or ""
    ext = os.path.splitext(orig)[1].lower()
    if ext not in AVATAR_ALLOWED_EXT:
        return JSONResponse({"error": f"Formato no permitido: {ext}"}, status_code=400)

    data = await avatar.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > AVATAR_MAX_MB:
        return JSONResponse({"error": f"Máximo {AVATAR_MAX_MB} MB"}, status_code=400)

    prefix = _email_safe(email)
    dst = os.path.join(AVATAR_DIR, prefix + ext)

    # elimina variantes anteriores (si existían)
    for e in (".webp", ".png", ".jpg", ".jpeg"):
        p = os.path.join(AVATAR_DIR, prefix + e)
        if os.path.isfile(p) and p != dst:
            try:
                os.remove(p)
            except:
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
        "accept": request.headers.get("accept", "")
    }
    return {
        "logged_in": bool(sess.get("usuario")),
        "session_keys": sorted(list(sess.keys())),
        "session_preview": {
            "usuario": sess.get("usuario"),
            "rol": sess.get("rol"),
            "nombre": sess.get("nombre"),
            "sid_present": bool(sess.get("sid"))
        },
        "cookie_present": ("session" in (request.cookies or {})),
        "route": str(request.url),
        "headers": headers
    }

# ⭐ Diagnóstico de templates/loader
@app.get("/__diag/templates")
async def _diag_templates():
    if (os.getenv("ENABLE_DIAG", "0") != "1"):
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "loader": str(getattr(templates.env, "loader", "")),
        "auto_reload": bool(getattr(templates.env, "auto_reload", False))
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
            "sid_present": bool(sess.get("sid"))
        },
        "cookie_present": ("session" in (request.cookies or {})),
        "route": str(request.url)
    }
# =========================
# main.py — PARTE 4 / 5
# (chat OpenAI, chat interno, auditoría, admin + Incidencias con adjuntos)
# =========================

# ================== API puente (Chat OpenAI) ==================
@app.post("/chat-openai")
async def chat_openai(request: Request):
    data = await request.json()
    mensaje = data.get("mensaje", "")
    usuario_actual = request.session.get("usuario", "Desconocido")

    try:
        historial = obtener_historial_completo()
    except Exception:
        historial = []

    ultimo_analisis_usuario = next(
        (h for h in historial if h.get("usuario") == usuario_actual and h.get("resumen")),
        None
    )

    if ultimo_analisis_usuario:
        ultimo_resumen = f"""
📌 Último análisis del usuario actual:
- Fecha: {ultimo_analisis_usuario.get('fecha')}
- Archivo: {ultimo_analisis_usuario.get('nombre_archivo')}
- Resumen:
{ultimo_analisis_usuario.get('resumen')}
"""
    else:
        ultimo_resumen = "(El usuario aún no tiene análisis registrados.)"

    contexto_general = "\n".join([
        f"- [{h.get('fecha')}] {h.get('usuario')} analizó '{h.get('nombre_archivo')}' y obtuvo:\n{h.get('resumen')}\n"
        for h in historial if h.get("resumen")
    ])

    contexto = f"{ultimo_resumen}\n\n📚 Historial completo:\n{contexto_general}"

    respuesta = await run_in_threadpool(responder_chat_openai, mensaje, contexto, usuario_actual)
    return JSONResponse({"respuesta": respuesta})

@app.post("/api/chat-openai")
async def api_chat_openai(request: Request, payload: dict = Body(...)):
    mensaje = (payload or {}).get("message", "").strip()
    usuario_actual = request.session.get("usuario", "Desconocido")

    if not mensaje:
        return JSONResponse({"reply": "Decime qué necesitás revisar del pliego 👌"})

    try:
        historial = obtener_historial_completo()
    except Exception:
        historial = []

    ultimo_analisis_usuario = next(
        (h for h in historial if h.get("usuario") == usuario_actual and h.get("resumen")),
        None
    )

    if ultimo_analisis_usuario:
        ultimo_resumen = f"""
📌 Último análisis del usuario actual:
- Fecha: {ultimo_analisis_usuario.get('fecha')}
- Archivo: {ultimo_analisis_usuario.get('nombre_archivo')}
- Resumen:
{ultimo_analisis_usuario.get('resumen')}
"""
    else:
        ultimo_resumen = "(El usuario aún no tiene análisis registrados.)"

    contexto_general = "\n".join([
        f"- [{h.get('fecha')}] {h.get('usuario')} analizó '{h.get('nombre_archivo')}' y obtuvo:\n{h.get('resumen')}\n"
        for h in historial if h.get("resumen")
    ])

    contexto = f"{ultimo_resumen}\n\n📚 Historial completo:\n{contexto_general}"

    respuesta = await run_in_threadpool(responder_chat_openai, mensaje, contexto, usuario_actual)
    return JSONResponse({"reply": respuesta})

# ===== Mini vista embebida para el widget del topbar/FAB =====
@app.get("/chat_openai_embed", response_class=HTMLResponse)
async def chat_openai_embed(request: Request):
    if not request.session.get("usuario"):
        return HTMLResponse("<div style='padding:12px'>Iniciá sesión para usar el chat.</div>")
    html = """
    <!doctype html><html><head>
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
        const log  = document.getElementById('log');
        const ta   = document.getElementById('t');
        const btn  = document.getElementById('send');
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
            const r = await fetch('/chat-openai', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mensaje:v}) });
            const j = await r.json().catch(()=>({}));
            add('<div class="mt-1"><b>IA:</b> '+(j.respuesta||'')+'</div>');
          }catch(_){ add('<div class="text-danger mt-1"><b>Error:</b> No se pudo enviar.</div>'); }
          finally{ busy=false; btn.disabled=false; ta.focus(); }
        }
        ta.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); } });
        btn.addEventListener('click', (e)=>{ e.preventDefault(); send(); });
      </script>
    </body></html>
    """
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
    if s.startswith("admin"): return "admin"
    if s.startswith("usuar"): return "usuario"
    if s == "borrado": return "borrado"
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
        print("❌ Error api_buscar_usuarios:", repr(e))
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
        msg_id = enviar_mensaje(de_email=de, para_email=para, texto=texto,
                                actor_user_id=actor_user_id, ip=ip)
        await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=texto)
        return JSONResponse({"ok": True, "id": msg_id})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": False, "error": "Inicialicé las tablas de chat, intentá de nuevo."}, status_code=503)
        print("❌ Error chat_enviar:", repr(e))
        return JSONResponse({"error": "No se pudo enviar el mensaje"}, status_code=500)

@app.post("/chat/enviar-archivos")
async def chat_enviar_archivos(
    request: Request,
    para: str = Form(...),
    texto: str = Form(default=""),
    archivos: List[UploadFile] = File(default=[])
):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    de = request.session.get("usuario")

    files = [a for a in archivos if a and a.filename]
    if len(files) > CHAT_MAX_FILES:
        return JSONResponse({"error": f"Máximo {CHAT_MAX_FILES} archivos por mensaje"}, status_code=400)

    actor_user_id, ip = _actor_info(request)
    try:
        msg_id = enviar_mensaje(de_email=de, para_email=para, texto=texto or "", actor_user_id=actor_user_id, ip=ip)
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"ok": False, "error": "Inicialicé las tablas de chat, intentá de nuevo."}, status_code=503)
        print("❌ Error creando mensaje:", repr(e))
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
        if (total_bytes / (1024 * 1024)) > INCID_MAX_TOTAL_MB:
            try:
                os.remove(path)
            except:
                pass
            return JSONResponse({"error": f"Tamaño total supera {CHAT_MAX_TOTAL_MB} MB"}, status_code=400)

        try:
            guardar_adjunto(
                mensaje_id=msg_id,
                filename=safe_name,
                original=orig,
                mime=archivo.content_type or "",
                size=written
            )
        except Exception as e:
            print("❌ Error guardar_adjunto:", repr(e))

    await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=(texto or "[Adjuntos]"))
    return JSONResponse({"ok": True, "id": msg_id})

@app.post("/chat/enviar-archivo")
async def chat_enviar_archivo(
    request: Request,
    para: str = Form(...),
    texto: str = Form(default=""),
    archivo: UploadFile = File(...)
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
        print("❌ Error chat_hilos:", repr(e))
        return JSONResponse({"error": "No se pudieron obtener los hilos"}, status_code=500)

@app.get("/chat/mensajes")
async def chat_mensajes(request: Request, con: str, limit: int = 100):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    yo = request.session.get("usuario")
    if not con:
        return JSONResponse({"error": "Falta parámetro 'con' (email del contacto)"}, status_code=400)
    try:
        mensajes = obtener_mensajes_entre(yo, con, limit=limit)
        return JSONResponse({"entre": [yo, con], "mensajes": mensajes})
    except Exception as e:
        if _is_no_table_error(e):
            ensure_chat_tables()
            return JSONResponse({"entre": [yo, con], "mensajes": []})
        print("❌ Error chat_mensajes:", repr(e))
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
        print("❌ Error chat_marcar_leidos:", repr(e))
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
        print("❌ Error chat_no_leidos:", repr(e))
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
        print("❌ Error chat_ocultar:", repr(e))
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
        print("❌ Error chat_restaurar:", repr(e))
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
        print("❌ Error chat_abrir:", repr(e))
        return JSONResponse({"error": "No se pudo abrir el hilo"}, status_code=500)


# ================== Auditoría (vista audit_logs) ==================
@app.get("/auditoria", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def ver_auditoria(request: Request):
    logs = obtener_auditoria()
    return templates.TemplateResponse("auditoria.html", {
        "request": request,
        "logs": logs
    })

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
    rol: str = "usuario"   # acepta "Administrador"/"Usuario"

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
        print("❌ admin_users_list/listar_usuarios:", repr(e))
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
    if obtener_usuario_por_email(email):
        return JSONResponse({"error": "El email ya existe"}, status_code=409)
    try:
        agregar_usuario(
            nombre=payload.nombre.strip(),
            email=email,
            password=DEFAULT_NEW_USER_PASSWORD,
            rol=_norm_rol(payload.rol),
            actor_user_id=actor_user_id,
            ip=ip
        )
        return {"ok": True}
    except Exception as e:
        print("❌ admin_users_create:", repr(e))
        return JSONResponse({"error": "No se pudo crear el usuario"}, status_code=500)

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
        print("❌ admin_users_password:", repr(e))
        return JSONResponse({"error": "No se pudo actualizar la contraseña"}, status_code=500)

@app.post("/api/admin/users/toggle")
async def admin_users_toggle(request: Request, payload: AdminToggleIn):
    require_admin(request)
    actor_user_id, ip = _actor_info(request)
    try:
        cambiar_estado_usuario(payload.email.lower(), 1 if payload.activo else 0, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("❌ admin_users_toggle:", repr(e))
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
        print("❌ admin_users_role:", repr(e))
        return JSONResponse({"error": "No se pudo cambiar el rol"}, status_code=500)

@app.delete("/api/admin/users/{email:path}")
async def admin_users_delete(request: Request, email: str, hard: bool = Query(default=False)):
    require_admin(request)
    actor_user_id, ip = _actor_info(request)
    try:
        borrar_usuario(email.lower(), actor_user_id=actor_user_id, ip=ip, soft=(not hard))
        return {"ok": True}
    except Exception as e:
        print("❌ admin_users_delete:", repr(e))
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
        rol=(data.get("rol") or "usuario").strip()
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
        password=new_pwd
    )
    return await admin_users_password(request, payload)

@app.post("/admin/usuarios/toggle")
async def legacy_admin_toggle(request: Request):
    data = await _json_or_form(request)
    activo_raw = str(data.get("activo", "")).lower()
    activo = activo_raw in ("1", "true", "t", "yes", "on", "si", "sí")
    payload = AdminToggleIn(
        email=(data.get("email") or "").strip(),
        activo=activo
    )
    return await admin_users_toggle(request, payload)

@app.post("/admin/usuarios/rol")
async def legacy_admin_role(request: Request):
    data = await _json_or_form(request)
    payload = AdminRoleIn(
        email=(data.get("email") or "").strip(),
        rol=(data.get("rol") or "").strip()
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

# 👇 NUEVOS endpoints legacy que tu admin.html ya usa (activar/desactivar/reset sesión)
@app.post("/admin/desactivar-usuario", dependencies=[Depends(require_admin)])
async def legacy_admin_disable_user(request: Request):
    data = await _json_or_form(request)
    email = (data.get("email") or "").strip().lower()
    actor_user_id, ip = _actor_info(request)
    try:
        cambiar_estado_usuario(email, 0, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("❌ legacy_admin_disable_user:", repr(e))
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
        print("❌ legacy_admin_enable_user:", repr(e))
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
                (now, "admin-reset", email)
            )
        return {"ok": True}
    except Exception as e:
        print("❌ legacy_admin_reset_session:", repr(e))
        return JSONResponse({"error": "No se pudo reiniciar la sesión"}, status_code=500)


# ================== Incidencias (vista + API con adjuntos) ==================
# ⚠️ Importante: REUSAMOS INCID_ATTACH_DIR / INCID_ALLOWED_EXT / INCID_MAX_TOTAL_MB
# definidos en la PARTE 1 (no los redefinimos acá para evitar inconsistencias).

def _incid_list_attachments(ticket_id: int):
    """Lista adjuntos por convención de nombre '<id>_...ext' en el dir de incidencias."""
    prefix = f"{int(ticket_id)}_"
    out = []
    for name in sorted(os.listdir(INCID_ATTACH_DIR)):
        if not name.startswith(prefix):
            continue
        path = os.path.join(INCID_ATTACH_DIR, name)
        if not os.path.isfile(path):
            continue
        size = os.path.getsize(path)
        out.append({
            "filename": name,
            "size": size,
            "url": f"/incidencias/adjunto/{ticket_id}/{name}",
            "original": re.sub(rf"^{re.escape(prefix)}\d+_", "", name)  # mejor esfuerzo
        })
    return out

def _parse_iso_utc(s: str):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # asumir UTC si no trae tz
    return dt.astimezone(timezone.utc)

def _infer_ticket_id(usuario: str, titulo: str, descripcion: str, tipo: str, ref_iso_utc: str) -> Optional[int]:
    """
    Heurística para recuperar el ID del ticket recién creado cuando crear_ticket()
    no lo devuelve: busca por user + (titulo, descripcion, tipo) cercano al now.
    """
    try:
        rows = obtener_tickets_por_usuario(usuario) or []
    except Exception:
        rows = []
    ref = _parse_iso_utc(ref_iso_utc) or datetime.now(timezone.utc)
    best = None
    best_dt_diff = 999_999_999
    for r in rows:
        # rows: (id, usuario, titulo, descripcion, tipo, estado, fecha)
        if (r[2] or "").strip() != titulo.strip():
            continue
        if (r[3] or "").strip() != (descripcion or "").strip():
            continue
        if (r[4] or "").strip() != (tipo or "General").strip():
            continue
        dt = _parse_iso_utc(r[6]) or ref
        diff = abs(int((dt - ref).total_seconds()))
        if diff < best_dt_diff:
            best_dt_diff = diff
            best = int(r[0])
    return best

def _validate_incid_ext(filename: str):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in INCID_ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido: {ext}")

@app.get("/incidencias", response_class=HTMLResponse)
@app.get("/incidencias/", response_class=HTMLResponse)  # alias con barra final
async def incidencias_view(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    email = request.session.get("usuario")
    rol = request.session.get("rol", "usuario")

    # admin ve todo, usuario ve las suyas
    rows = obtener_todos_los_tickets() if (rol == "admin" or es_admin(email)) else obtener_tickets_por_usuario(email)

    tickets = []
    for r in rows:
        try:
            fecha_leg = iso_utc_to_ar_str(r[6]) if r[6] else ""
        except Exception:
            try:
                fecha_leg = datetime.strptime(r[6], "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
            except Exception:
                fecha_leg = r[6] or ""
        t = {
            "id": r[0], "usuario": r[1], "titulo": r[2], "descripcion": r[3],
            "tipo": r[4], "estado": r[5], "fecha": r[6], "fecha_legible": fecha_leg
        }
        # 👉 adjuntos
        try:
            t["adjuntos"] = _incid_list_attachments(int(r[0]))
        except Exception:
            t["adjuntos"] = []
        tickets.append(t)

    return templates.TemplateResponse("incidencias.html", {
        "request": request,
        "tickets": tickets,
        "usuario_actual": {"nombre": request.session.get("nombre") or email, "rol": rol}
    })

@app.get("/api/incidencias")
async def incidencias_list_json(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    email = request.session.get("usuario")
    rol = request.session.get("rol", "usuario")
    rows = obtener_todos_los_tickets() if (rol == "admin" or es_admin(email)) else obtener_tickets_por_usuario(email)
    items = []
    for r in rows:
        items.append({
            "id": r[0], "usuario": r[1], "titulo": r[2], "descripcion": r[3],
            "tipo": r[4], "estado": r[5], "fecha": r[6],
            "adjuntos": _incid_list_attachments(int(r[0]))
        })
    return {"items": items}

@app.post("/incidencias/crear")
async def incidencias_crear(
    request: Request,
    titulo: str = Form(...),
    descripcion: str = Form(""),
    tipo: str = Form("General"),
    archivos: List[UploadFile] = File(default=[])  # 👈 opcional, múltiples
):
    if not request.session.get("usuario"):
        return JSONResponse({"error":"No autenticado"}, status_code=401)
    usuario = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)

    # Creamos primero el ticket
    now_iso = now_iso_utc()
    crear_ticket(usuario, titulo.strip(), (descripcion or "").strip(), (tipo or "General").strip(),
                 actor_user_id=actor_user_id, ip=ip)

    # Intentamos inferir el ID del ticket recién creado
    ticket_id = _infer_ticket_id(usuario, titulo, descripcion or "", tipo or "General", now_iso)

    # Guardar adjuntos (si pudimos resolver el id)
    files = [a for a in (archivos or []) if (a and a.filename)]
    if ticket_id and files:
        total_bytes = 0
        for i, f in enumerate(files, start=1):
            _validate_incid_ext(f.filename)
            ext = os.path.splitext(f.filename)[1].lower()
            safe = _safe_basename(f.filename)
            name = f"{ticket_id}_{i:02d}_{safe}{ext}"
            path = os.path.join(INCID_ATTACH_DIR, name)
            written = await _save_upload_stream(f, path)
            total_bytes += written
            if (total_bytes / (1024 * 1024)) > INCID_MAX_TOTAL_MB:
                try: os.remove(path)
                except: pass
                return JSONResponse({"error": f"Tamaño total supera {INCID_MAX_TOTAL_MB} MB"}, status_code=400)

    return ({"ok": True, "id": ticket_id} if wants_json(request) else RedirectResponse("/incidencias", status_code=303))

# --- Aliases para aceptar también POST a /incidencias (JSON sin archivos) ---
@app.post("/incidencias")
@app.post("/incidencias/")
async def incidencias_post_alias(
    request: Request,
    titulo: Optional[str] = Form(default=None),
    descripcion: Optional[str] = Form(default=""),
    tipo: Optional[str] = Form(default="General"),
):
    # Si vino por fetch JSON:
    if titulo is None:
        try:
            data = await request.json()
        except Exception:
            data = {}
        titulo = (data.get("titulo") or data.get("title") or "").strip()
        descripcion = (data.get("descripcion") or data.get("description") or "")
        tipo = (data.get("tipo") or data.get("type") or "General")
    return await incidencias_crear(
        request,
        titulo=titulo or "",
        descripcion=descripcion or "",
        tipo=tipo or "General",
        archivos=[],
    )

@app.get("/incidencias/adjunto/{ticket_id}/{filename}")
async def incidencias_adjunto(ticket_id: int, filename: str):
    """Sirve un adjunto de incidencia, validando que el filename pertenezca al ticket."""
    filename = os.path.basename(filename)
    if not filename.startswith(f"{int(ticket_id)}_"):
        return JSONResponse({"error": "Adjunto inválido"}, status_code=400)
    path = os.path.join(INCID_ATTACH_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "No encontrado"}, status_code=404)
    return FileResponse(path)

@app.post("/incidencias/cerrar")
async def incidencias_cerrar(request: Request, id: int = Form(...)):
    if not request.session.get("usuario"):
        return JSONResponse({"error":"No autenticado"}, status_code=401)
    actor_user_id, ip = _actor_info(request)
    actualizar_estado_ticket(id, "Cerrado", actor_user_id=actor_user_id, ip=ip)
    return ({"ok": True} if wants_json(request) else RedirectResponse("/incidencias", status_code=303))

@app.post("/incidencias/eliminar")
async def incidencias_eliminar(request: Request, id: int = Form(...)):
    if not request.session.get("usuario"):
        return JSONResponse({"error":"No autenticado"}, status_code=401)
    actor_user_id, ip = _actor_info(request)
    eliminar_ticket(id, actor_user_id=actor_user_id, ip=ip)
    # también podemos eliminar adjuntos del disco
    prefix = f"{int(id)}_"
    try:
        for name in list(os.listdir(INCID_ATTACH_DIR)):
            if name.startswith(prefix):
                try:
                    os.remove(os.path.join(INCID_ATTACH_DIR, name))
                except:
                    pass
    except Exception:
        pass
    return ({"ok": True} if wants_json(request) else RedirectResponse("/incidencias", status_code=303))

@app.post("/incidencias/cerrar/{id}")
async def incidencias_cerrar_path(id: int, request: Request):
    return await incidencias_cerrar(request, id=id)

@app.post("/incidencias/eliminar/{id}")
async def incidencias_eliminar_path(id: int, request: Request):
    return await incidencias_eliminar(request, id=id)

# (Opcional) diagnóstico de rutas
@app.get("/__diag/routes")
def _diag_routes():
    return {"routes": sorted({getattr(r, "path", "") for r in app.routes})}
# =========================
# main.py — PARTE 5 / 5
# (Calendario, Notificaciones, Presencia/Online, Auditoría de actividad + CSV)
# =========================

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
        return JSONResponse({"error":"No autenticado"}, status_code=401)
    data = await request.json()
    title = (data.get("title") or "").strip()
    start = data.get("start")
    end   = data.get("end")
    all_day = 1 if data.get("AllDay") or data.get("allDay") else 0
    desc  = (data.get("description") or "").strip()
    color = (data.get("color") or "#0ea5e9").strip()
    if not title or not start:
        return JSONResponse({"error":"Faltan campos: title, start"}, status_code=400)

    evt_id = uuid.uuid4().hex
    now = _now_iso()
    created_by = request.session.get("usuario","Desconocido")
    with cal_conn() as c:
        c.execute("""
            INSERT INTO eventos(id,title,description,start,end,all_day,color,created_by,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)
        """, (evt_id,title,desc,start,end,all_day,color,created_by,now,now))
    await notify_async(created_by, "Evento creado", f"{title} • {start}{(' → '+end) if end else ''}")
    return {
        "id": evt_id, "title": title, "description": desc, "start": start, "end": end,
        "allDay": bool(all_day), "color": color
    }

@app.patch("/calendario/eventos/{evt_id}")
async def cal_update(evt_id: str, request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error":"No autenticado"}, status_code=401)
    data = await request.json()

    def to_iso(v):
        if v is None:
            return None
        return v if isinstance(v, str) else str(v)

    title = data.get("title")
    desc  = data.get("description")
    color = data.get("color")
    start = to_iso(data.get("start"))
    end   = to_iso(data.get("end"))
    all_day = data.get("AllDay") if "AllDay" in data else data.get("allDay")

    sets, vals = [], []
    if title is not None: sets.append("title=?"); vals.append(title)
    if desc  is not None: sets.append("description=?"); vals.append(desc)
    if color is not None: sets.append("color=?"); vals.append(color)
    if start is not None: sets.append("start=?"); vals.append(start)
    if end   is not None: sets.append("end=?"); vals.append(end)
    if all_day is not None: sets.append("all_day=?"); vals.append(1 if all_day else 0)
    sets.append("updated_at=?"); vals.append(_now_iso())
    vals.append(evt_id)

    if len(sets) == 1:
        return JSONResponse({"error":"Nada para actualizar"}, status_code=400)

    with cal_conn() as c:
        cur = c.execute(f"UPDATE eventos SET {', '.join(sets)} WHERE id=?", vals)
        if cur.rowcount == 0:
            return JSONResponse({"error":"Evento no encontrado"}, status_code=404)

    await notify_async(request.session.get("usuario","Desconocido"), "Evento actualizado", f"ID: {evt_id}")
    return {"ok": True}

@app.delete("/calendario/eventos/{evt_id}")
async def cal_delete(evt_id: str, request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error":"No autenticado"}, status_code=401)
    with cal_conn() as c:
        cur = c.execute("DELETE FROM eventos WHERE id=?", (evt_id,))
        if cur.rowcount == 0:
            return JSONResponse({"error":"Evento no encontrado"}, status_code=404)
    await notify_async(request.session.get("usuario","Desconocido"), "Evento eliminado", f"ID: {evt_id}")
    return {"ok": True}


# =====================================================================
# ========================== NOTIFICACIONES ============================
# =====================================================================
@app.get("/notificaciones")
async def notificaciones(request: Request,
                         q: Optional[str] = Query(default=None),
                         only_unread: Optional[bool] = Query(default=None),
                         limit: int = Query(default=20, ge=1, le=200),
                         offset: int = Query(default=0, ge=0)):
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
            "SELECT COUNT(1) FROM notificaciones WHERE user=? AND leida=0", (user,)
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
        items = [{
            "id": r["id"],
            "titulo": r["titulo"],
            "cuerpo": r["cuerpo"],
            "fecha_legible": iso_utc_to_ar_str(r["created_at"]),
            "leida": bool(r["leida"])
        } for r in cur.fetchall()]

    return {"total_unread": total_unread, "items": items}

@app.get("/notificaciones/vista", response_class=HTMLResponse)
async def notificaciones_vista(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("notificaciones.html", {"request": request})

# 👉 Aliases/redirects para que la campana y “Ver todas” SIEMPRE abran la vista HTML
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
            c.execute(f"UPDATE notificaciones SET leida=1 WHERE user=? AND id IN ({placeholders})", (user, *ids))
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
        c.execute("""
            CREATE TABLE IF NOT EXISTS presence(
                user TEXT PRIMARY KEY,
                nombre TEXT,
                last_seen TEXT NOT NULL,
                ip TEXT,
                ua TEXT
            )
        """)
        c.execute("""
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
        """)
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
        c.execute("""
            INSERT INTO presence(user, nombre, last_seen, ip, ua)
            VALUES(?,?,?,?,?)
            ON CONFLICT(user) DO UPDATE SET
              nombre=excluded.nombre,
              last_seen=excluded.last_seen,
              ip=excluded.ip,
              ua=excluded.ua
        """, (email, nombre, now, ip, ua))

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
                items.append({
                    "email": r["user"],
                    "nombre": r["nombre"] or r["user"],
                    "last_seen": r["last_seen"],
                    "ip": r["ip"] or "",
                    "ua": r["ua"] or ""
                })
    return {"items": items}

@app.get("/usuarios-activos", response_class=HTMLResponse)
async def usuarios_activos(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    data = await presence_online(minutes=5)
    return templates.TemplateResponse("usuarios_activos.html", {
        "request": request,
        "items": data.get("items", [])
    })


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
    limit: int = Query(default=500, ge=1, le=5000)
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
            last_dt  = _to_dt(r["last_seen"])
            logout_dt= _to_dt(r["logout_at"])

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

            rows_out.append({
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
                "duracion_seg": dur_sec
            })

    return {"items": rows_out, "timeout_min": SESSION_TIMEOUT_MIN, "now_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")}

@app.get("/auditoria/actividad.csv", dependencies=[Depends(require_admin)])
async def auditoria_actividad_csv(
    request: Request,
    usuario: Optional[str] = Query(default=None),
    desde: Optional[str] = Query(default=None),
    hasta: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000)
):
    data = await auditoria_actividad(request, usuario=usuario, desde=desde, hasta=hasta, limit=limit)
    items = data.get("items", [])

    headers = ["estado","usuario","nombre","login_at","last_seen","logout_at","duracion_seg","ip","ua","sid","closed_reason"]
    lines = [",".join(headers)]
    for it in items:
        row = [
            it.get("estado",""),
            it.get("usuario",""),
            it.get("nombre",""),
            it.get("login_at",""),
            it.get("last_seen",""),
            it.get("logout_at","") or "",
            str(it.get("duracion_seg") or 0),
            it.get("ip",""),
            (it.get("ua","") or "").replace(",", " "),
            it.get("id",""),
            it.get("closed_reason","").replace(",", " ")
        ]
        lines.append(",".join(row))

    csv_body = "\n".join(lines)
    filename = "auditoria_actividad.csv"
    return Response(
        content=csv_body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
