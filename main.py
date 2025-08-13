# main.py
import os
import sqlite3
import uuid
import asyncio
import re
from typing import List, Optional, Dict, Set
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Body, WebSocket, WebSocketDisconnect, Depends, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.concurrency import run_in_threadpool
from datetime import datetime
from sqlalchemy import or_, and_

from utils import (
    extraer_texto_de_pdf,
    analizar_con_openai,
    generar_pdf_con_plantilla,
    responder_chat_openai,
    analizar_anexos
)

from database import (
    # Inicialización
    inicializar_bd,
    # Usuarios
    obtener_usuario_por_email, agregar_usuario, listar_usuarios,
    actualizar_password, cambiar_estado_usuario, borrar_usuario,
    buscar_usuarios,
    # Historial
    guardar_en_historial, obtener_historial, eliminar_del_historial,
    obtener_historial_completo,
    # Tickets
    crear_tabla_tickets, crear_ticket, obtener_todos_los_tickets, obtener_tickets_por_usuario,
    actualizar_estado_ticket, eliminar_ticket,
    # Auditoría
    obtener_auditoria,
    # Chat interno
    enviar_mensaje, obtener_hilos_para, obtener_mensajes_entre,
    marcar_mensajes_leidos, contar_no_leidos,
    # Hilos ocultos
    ocultar_hilo, restaurar_hilo,
    # Adjuntos (chat)
    guardar_adjunto,
    # Roles helpers
    es_admin
)

# ORM (audit_logs)
from db_orm import inicializar_bd_orm, SessionLocal, AuditLog

# ================== App & Middlewares ==================
app = FastAPI(middleware=[
    Middleware(SessionMiddleware, secret_key="clave_secreta_super_segura")
])

# Inicializa BD SQLite (usuarios, historial, tickets, mensajes, hilos_ocultos, adjuntos)
inicializar_bd()
# Inicializa ORM (audit_logs) según DATABASE_URL
inicializar_bd_orm()

# Static
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated_pdfs", StaticFiles(directory="generated_pdfs"), name="generated_pdfs")

templates = Jinja2Templates(directory="templates")
templates.env.globals['os'] = os

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
                await websocket.send_json({"event": "ws:pong", "ts": datetime.utcnow().isoformat() + "Z"})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, email)
    except Exception:
        manager.disconnect(websocket, email)

async def emit_alert(email: str, title: str, body: str = "", extra: dict = None):
    payload = {"event": "alert:new", "title": title, "body": body, "ts": datetime.utcnow().isoformat() + "Z"}
    if extra:
        payload["extra"] = extra
    await manager.send_to_user(email, payload)

async def emit_chat_new_message(para_email: str, de_email: str, msg_id: int, preview: str = ""):
    payload = {"event": "chat:new_message", "from": de_email, "id": msg_id, "preview": preview[:120], "ts": datetime.utcnow().isoformat() + "Z"}
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

# --- Filtros para eliminación masiva de auditoría ---
def _build_audit_filters(q, filtros):
    if not filtros:
        return q
    acc = (filtros.get("accion") or "").strip()
    d   = (filtros.get("desde")  or "").strip()
    h   = (filtros.get("hasta")  or "").strip()
    term= (filtros.get("term")   or "").strip()

    if acc:
        q = q.filter(AuditLog.accion == acc)
    if d:
        q = q.filter(AuditLog.fecha >= f"{d} 00:00:00")
    if h:
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

# --- Saneo de informe (extra por si llega ruido del modelo) ---
_TITLE_RE = re.compile(
    r'^\s*(informe\s+estandarizad[oa]\s+de\s+pliego\s+de\s+licitación)\s*:?\s*$',
    re.IGNORECASE
)
def _sanear_informe(texto: str) -> str:
    if not texto:
        return texto
    lines = texto.splitlines()
    out = []
    title_seen = False
    for ln in lines:
        if _TITLE_RE.match(ln or ""):
            if title_seen:
                continue
            title_seen = True
            out.append("Informe Estandarizado de Pliego de Licitación")
        else:
            out.append(ln)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

# --- Config pública para el front ---
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
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    usuario = obtener_usuario_por_email(email)
    if usuario and usuario[3] == password:
        request.session["usuario"] = usuario[2]   # email
        request.session["email"] = usuario[2]
        request.session["rol"] = usuario[4]
        request.session["nombre"] = usuario[1] or usuario[2]
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Credenciales incorrectas"})

@app.post("/logout")
async def logout_post(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

@app.get("/logout")
async def logout_get(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

# ================== Análisis ==================
@app.post("/analizar-pliego")
async def analizar_pliego(request: Request, archivos: List[UploadFile] = File(...)):
    """
    Unifica TODOS los anexos recibidos en un solo análisis y devuelve:
      - resumen (texto)
      - nombre del PDF generado
    """
    usuario = request.session.get("usuario", "Anónimo")

    if not archivos:
        return JSONResponse({"error": "Subí al menos un archivo"}, status_code=400)

    # Validación de extensiones
    for a in archivos:
        if not a or not a.filename:
            continue
        _validate_ext(a.filename)

    # 1) Análisis integrado
    try:
        resumen = await run_in_threadpool(analizar_anexos, archivos)
        resumen = _sanear_informe(resumen)
    except Exception as e:
        return JSONResponse({"error": f"Fallo en el análisis: {e}"}, status_code=500)

    # 2) Generar PDF
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        nombre_archivo = f"resumen_{timestamp}.pdf"
        await run_in_threadpool(generar_pdf_con_plantilla, resumen, nombre_archivo)
    except Exception as e:
        return JSONResponse({"error": f"Fallo al generar PDF: {e}", "resumen": resumen}, status_code=500)

    # 3) Persistir en historial
    try:
        guardar_en_historial(timestamp, usuario, nombre_archivo, nombre_archivo, resumen)
    except Exception:
        pass

    return {"resumen": resumen, "pdf": nombre_archivo}

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
    ruta = os.path.join("generated_pdfs", archivo)
    if os.path.exists(ruta):
        return FileResponse(ruta, media_type='application/pdf', filename=archivo)
    return {"error": "Archivo no encontrado"}

@app.delete("/eliminar/{timestamp}")
async def eliminar_archivo(timestamp: str):
    eliminar_del_historial(timestamp)
    ruta = os.path.join("generated_pdfs", f"resumen_{timestamp}.pdf")
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

    return {"usuario": email or "Desconocido", "rol": rol, "nombre": nombre, "avatar_url": avatar_url}

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

    for e in (".webp", ".png", ".jpg", ".jpeg"):
        p = os.path.join(AVATAR_DIR, prefix + e)
        if os.path.isfile(p) and p != dst:
            try: os.remove(p)
            except: pass

    with open(dst, "wb") as f:
        f.write(data)

    url = f"/{dst.replace(os.sep, '/')}"

    try:
        await emit_alert(email, "Perfil actualizado", "Tu avatar se actualizó correctamente")
    except Exception:
        pass

    return {"ok": True, "avatar_url": url}

# ================== Incidencias ==================
@app.get("/incidencias", response_class=HTMLResponse)
async def vista_incidencias(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    usuario = request.session.get("usuario")
    rol = request.session.get("rol")
    tickets_raw = obtener_todos_los_tickets() if rol == "admin" else obtener_tickets_por_usuario(usuario)
    tickets = []
    for t in tickets_raw:
        if len(t) < 7:
            continue
        fecha_legible = datetime.strptime(t[6], "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M")
        carpeta_adjuntos = os.path.join("static", "adjuntos_incidencias")
        prefix = f"{t[1]}_{t[6].replace(':','').replace('-','').replace(' ','')[:14]}"
        adjuntos = []
        if os.path.exists(carpeta_adjuntos):
            for file in os.listdir(carpeta_adjuntos):
                if file.startswith(prefix):
                    adjuntos.append(file)
        tickets.append({
            "id": t[0], "usuario": t[1], "titulo": t[2], "descripcion": t[3],
            "tipo": t[4], "estado": t[5], "fecha": t[6],
            "fecha_legible": fecha_legible, "adjuntos": adjuntos
        })
    return templates.TemplateResponse("incidencias.html", {
        "request": request, "tickets": tickets,
        "usuario_actual": {"nombre": usuario, "rol": rol}
    })

@app.post("/incidencias")
async def crear_incidencia_form(
    request: Request,
    titulo: str = Form(...),
    descripcion: str = Form(...),
    tipo: str = Form(...),
    archivos: List[UploadFile] = File(default=[])
):
    usuario = request.session.get("usuario", "Anónimo")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    actor_user_id, ip = _actor_info(request)
    crear_ticket(usuario, titulo, descripcion, tipo, actor_user_id=actor_user_id, ip=ip)

    carpeta_adjuntos = os.path.join("static", "adjuntos_incidencias")
    os.makedirs(carpeta_adjuntos, exist_ok=True)

    for archivo in archivos:
        if archivo.filename:
            nombre_archivo = f"{usuario}_{timestamp}_{archivo.filename}".replace(" ", "_")
            ruta_archivo = os.path.join(carpeta_adjuntos, nombre_archivo)
            with open(ruta_archivo, "wb") as buffer:
                buffer.write(await archivo.read())

    return RedirectResponse("/incidencias", status_code=303)

@app.post("/incidencias/cerrar/{id}")
async def cerrar_incidencia_form(request: Request, id: int):
    actor_user_id, ip = _actor_info(request)
    actualizar_estado_ticket(id, "Cerrado", actor_user_id=actor_user_id, ip=ip)
    return RedirectResponse("/incidencias", status_code=303)

@app.post("/incidencias/eliminar/{id}")
async def eliminar_incidencia_form(request: Request, id: int):
    if request.session.get("rol") != "admin":
        return JSONResponse({"error": "Acceso denegado"}, status_code=403)
    actor_user_id, ip = _actor_info(request)
    eliminar_ticket(id, actor_user_id=actor_user_id, ip=ip)
    return RedirectResponse("/incidencias", status_code=303)

# ================== Cambio de contraseña ==================
@app.get("/cambiar-password", response_class=HTMLResponse)
async def cambiar_password_form(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("cambiar_password.html", {
        "request": request, "mensaje": "", "error": ""
    })

@app.post("/cambiar-password", response_class=HTMLResponse)
async def cambiar_password_submit(request: Request,
                                 old_password: str = Form(...),
                                 new_password: str = Form(...),
                                 confirm_password: str = Form(...)):
    usuario = request.session.get("usuario")
    if not usuario:
        return RedirectResponse("/login")
    datos = obtener_usuario_por_email(usuario)
    if not datos or datos[3] != old_password:
        return templates.TemplateResponse("cambiar_password.html", {
            "request": request, "mensaje": "", "error": "La contraseña actual es incorrecta."
        })
    if new_password != confirm_password:
        return templates.TemplateResponse("cambiar_password.html", {
            "request": request, "mensaje": "", "error": "La nueva contraseña no coincide en ambos campos."
        })
    actor_user_id, ip = _actor_info(request)
    actualizar_password(usuario, new_password, actor_user_id=actor_user_id, ip=ip)
    return templates.TemplateResponse("cambiar_password.html", {
        "request": request, "mensaje": "Contraseña cambiada correctamente.", "error": ""
    })

# ================== Admin ==================
@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def vista_admin(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/admin/usuarios", dependencies=[Depends(require_admin)])
async def listar_usuarios_api():
    return JSONResponse(listar_usuarios())

@app.post("/admin/crear-usuario", dependencies=[Depends(require_admin)])
async def crear_usuario_api(request: Request):
    try:
        data = await request.json()
        nombre = data.get("nombre")
        email = data.get("email")
        rol = data.get("rol")
        if not nombre or not email or not rol:
            return JSONResponse({"error": "Faltan campos: nombre, email, rol"}, status_code=400)
        actor_user_id, ip = _actor_info(request)
        agregar_usuario(nombre, email, "1234", rol, actor_user_id=actor_user_id, ip=ip)
        return JSONResponse({"mensaje": "Usuario creado correctamente con contraseña: 1234"})
    except Exception as e:
        print("❌ Error crear-usuario:", repr(e))
        return JSONResponse({"error": f"Error al crear usuario: {e}"}, status_code=400)

@app.post("/admin/blanquear-password", dependencies=[Depends(require_admin)])
async def blanquear_password(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        if not email:
            return JSONResponse({"error": "Falta email"}, status_code=400)
        actor_user_id, ip = _actor_info(request)
        actualizar_password(email, "1234", actor_user_id=actor_user_id, ip=ip)
        return JSONResponse({"mensaje": "Contraseña blanqueada a 1234"})
    except Exception as e:
        print("❌ Error blanquear-password:", repr(e))
        return JSONResponse({"error": f"Error al blanquear: {e}"}, status_code=400)

@app.post("/admin/desactivar-usuario", dependencies=[Depends(require_admin)])
async def desactivar_usuario(request: Request):
    data = await request.json()
    email = data.get("email")
    if not email:
        return JSONResponse({"error": "Falta email"}, status_code=400)
    actor_user_id, ip = _actor_info(request)
    cambiar_estado_usuario(email, 0, actor_user_id=actor_user_id, ip=ip)
    return JSONResponse({"mensaje": "Usuario desactivado"})

@app.post("/admin/activar-usuario", dependencies=[Depends(require_admin)])
async def activar_usuario(request: Request):
    data = await request.json()
    email = data.get("email")
    if not email:
        return JSONResponse({"error": "Falta email"}, status_code=400)
    actor_user_id, ip = _actor_info(request)
    cambiar_estado_usuario(email, 1, actor_user_id=actor_user_id, ip=ip)
    return JSONResponse({"mensaje": "Usuario activado"})

@app.post("/admin/eliminar-usuario", dependencies=[Depends(require_admin)])
async def eliminar_usuario(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        if not email:
            return JSONResponse({"error": "Falta email"}, status_code=400)
        actor_user_id, ip = _actor_info(request)
        ok = borrar_usuario(email, actor_user_id=actor_user_id, ip=ip, soft=False)
        if not ok:
            return JSONResponse({"error": "Usuario no encontrado"}, status_code=404)
        return JSONResponse({"mensaje": "Usuario eliminado definitivamente."})
    except Exception as e:
        print("❌ Error eliminar-usuario:", repr(e))
        return JSONResponse({"error": f"Error al eliminar: {e}"}, status_code=400)

# ================== Chat OpenAI flotante ==================
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

# ================== API puente (drawer) ==============
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
        const log=document.getElementById('log'); const ta=document.getElementById('t'); const btn=document.getElementById('send');
        function esc(s){ return (s||'').replaceAll('<','&lt;').replaceAll('>','&gt;'); }
        function add(b){ const p=document.createElement('div'); p.innerHTML=b; log.appendChild(p); log.scrollTop=log.scrollHeight; }
        function autosize(){ ta.style.height='auto'; ta.style.height = Math.min(ta.scrollHeight, 150) + 'px'; }
        ta.addEventListener('input', autosize); autosize();
        let busy=false;
        async function send(){
          if(busy) return; const v=ta.value.trim(); if(!v) return; busy=true; btn.disabled=true;
          add('<div><b>Tú:</b> '+esc(v)+'</div>'); ta.value=''; autosize();
          try{
            const r=await fetch('/chat-openai',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mensaje:v})});
            const j=await r.json().catch(()=>({})); add('<div class="mt-1"><b>IA:</b> '+(j.respuesta||'')+'</div>');
          }catch(_){ add('<div class="text-danger mt-1"><b>Error:</b> No se pudo enviar.</div>'); }
          finally{ busy=false; btn.disabled=false; ta.focus(); }
        }
        ta.addEventListener('keydown',(e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
        btn.addEventListener('click',(e)=>{ e.preventDefault(); send(); });
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
@app.get("/api/usuarios")
async def api_buscar_usuarios(request: Request, term: str = "", limit: int = 8):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    term = (term or "").strip()
    if not term:
        return {"items": []}
    try:
        items = buscar_usuarios(term, limit=limit)
        return {"items": [{"id": u["id"], "nombre": u.get("nombre") or "", "email": u["email"]} for u in items]}
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
        msg_id = enviar_mensaje(de_email=de, para_email=para, texto=texto, actor_user_id=actor_user_id, ip=ip)
        await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=texto)
        return JSONResponse({"ok": True, "id": msg_id})
    except Exception as e:
        print("❌ Error chat_enviar:", repr(e))
        return JSONResponse({"error": "No se pudo enviar el mensaje"}, status_code=500)

# ---- Enviar mensaje con múltiples archivos ----
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
        print("❌ Error creando mensaje:", repr(e))
        return JSONResponse({"error": "No se pudo crear el mensaje"}, status_code=500)

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
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
        if (total_bytes / (1024 * 1024)) > CHAT_MAX_TOTAL_MB:
            try: os.remove(path)
            except: pass
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
            # continúa

    await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=(texto or "[Adjuntos]"))
    return JSONResponse({"ok": True, "id": msg_id})

# ---- Compat: 1 archivo ----
@app.post("/chat/enviar-archivo")
async def chat_enviar_archivo(
    request: Request,
    para: str = Form(...),
    texto: str = Form(default=""),
    archivo: UploadFile = File(...)
):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    archivos = [archivo] if archivo and archivo.filename else []
    return await chat_enviar_archivos(request, para=para, texto=texto, archivos=archivos)

# ---- Descargar adjunto por nombre ----
@app.get("/chat/adjunto/{filename}")
async def chat_adjunto(filename: str):
    path = os.path.join(CHAT_ATTACH_DIR, filename)
    if not os.path.isfile(path):
        return JSONResponse({"error": "No encontrado"}, status_code=404)
    return FileResponse(path)

# ---- Hilos, mensajes, ocultar/restaurar, leídos ----
@app.get("/chat/hilos")
async def chat_hilos(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    yo = request.session.get("usuario")
    try:
        hilos = obtener_hilos_para(yo)
        return JSONResponse({"hilos": hilos})
    except Exception as e:
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
        print("❌ Error chat_mensajes:", repr(e))
        return JSONResponse({"error": "No se pudieron obtener los mensajes"}, status_code=500)

@app.post("/chat/ocultar-hilo")
async def chat_ocultar_hilo(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    data = await request.json()
    con = (data or {}).get("con")
    if not con:
        return JSONResponse({"error": "Falta 'con' (email del contacto)"}, status_code=400)
    yo = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)
    try:
        ocultar_hilo(yo, con, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("❌ Error ocultar_hilo:", repr(e))
        return JSONResponse({"error": "No se pudo ocultar el hilo"}, status_code=500)

@app.post("/chat/restaurar-hilo")
async def chat_restaurar_hilo(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    data = await request.json()
    con = (data or {}).get("con")
    if not con:
        return JSONResponse({"error": "Falta 'con' (email del contacto)"}, status_code=400)
    yo = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)
    try:
        restaurar_hilo(yo, con, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("❌ Error restaurar_hilo:", repr(e))
        return JSONResponse({"error": "No se pudo restaurar el hilo"}, status_code=500)

@app.post("/chat/marcar-leidos")
async def chat_marcar_leidos(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    data = await request.json()
    con = (data or {}).get("con")
    if not con:
        return JSONResponse({"error": "Falta 'con' (email del contacto)"}, status_code=400)
    yo = request.session.get("usuario")
    actor_user_id, ip = _actor_info(request)
    try:
        marcar_mensajes_leidos(yo, con, actor_user_id=actor_user_id, ip=ip)
        return {"ok": True}
    except Exception as e:
        print("❌ Error marcar_leidos:", repr(e))
        return JSONResponse({"error": "No se pudo marcar como leídos"}, status_code=500)

@app.get("/chat/no-leidos")
async def chat_no_leidos(request: Request):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    yo = request.session.get("usuario")
    try:
        n = contar_no_leidos(yo)
        return {"no_leidos": n}
    except Exception as e:
        print("❌ Error no_leidos:", repr(e))
        return JSONResponse({"error": "No se pudo obtener el contador"}, status_code=500)

# ================== Auditoría: listar y eliminar ==================
@app.get("/auditoria")
async def auditoria_list(
    request: Request,
    accion: Optional[str] = Query(default=None),
    desde: Optional[str] = Query(default=None),   # YYYY-MM-DD
    hasta: Optional[str] = Query(default=None),   # YYYY-MM-DD
    term: Optional[str]  = Query(default=None),
    limit: int = Query(default=200, ge=1, le=5000),
):
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)

    # Si tu DB tiene función obtener_auditoria, úsala (fallback ORM si no filtra)
    try:
        rows = obtener_auditoria(limit=limit)  # puede traer sin filtrar
    except Exception:
        rows = []

    # Si necesitás filtrar con ORM por performance:
    try:
        db = SessionLocal()
        q = db.query(AuditLog).order_by(AuditLog.id.desc())
        filtros = {"accion": accion, "desde": desde, "hasta": hasta, "term": term}
        q = _build_audit_filters(q, filtros)
        logs = q.limit(limit).all()
        out = []
        for a in logs:
            out.append({
                "id": a.id,
                "fecha": a.fecha,
                "usuario": a.usuario,
                "nombre": a.nombre,
                "accion": a.accion,
                "entidad": a.entidad,
                "entidad_id": a.entidad_id,
                "ip": a.ip,
                "before": a.before,
                "after": a.after
            })
        return {"items": out}
    except Exception as e:
        print("❌ Error auditoria_list ORM:", repr(e))
        # Fallback: devolver rows crudos si existen
        return {"items": rows[:limit]}
    finally:
        try:
            db.close()
        except Exception:
            pass

@app.post("/auditoria/eliminar")
async def auditoria_eliminar(
    request: Request,
    filtros: dict = Body(default={})
):
    """
    Elimina en base a filtros (acción/fecha/term). Usa ORM para DELETE.
    Ejemplo body:
    {
      "accion": "login",
      "desde": "2025-08-01",
      "hasta": "2025-08-13",
      "term": "usuario@example.com"
    }
    """
    if request.session.get("rol") != "admin":
        return JSONResponse({"error": "Solo admins"}, status_code=403)
    try:
        db = SessionLocal()
        q = db.query(AuditLog)
        q = _build_audit_filters(q, filtros)
        count = q.count()
        q.delete(synchronize_session=False)
        db.commit()
        return {"eliminados": count}
    except Exception as e:
        print("❌ Error auditoria_eliminar:", repr(e))
        return JSONResponse({"error": "No se pudo eliminar la auditoría con esos filtros"}, status_code=500)
    finally:
        try: db.close()
        except: pass
