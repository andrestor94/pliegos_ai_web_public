import os
import sqlite3
import uuid
import asyncio
from typing import List, Optional, Dict, Set
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException, Body, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.concurrency import run_in_threadpool
from datetime import datetime

from utils import (
    extraer_texto_de_pdf,
    analizar_con_openai,
    generar_pdf_con_plantilla,
    responder_chat_openai
)

from database import (
    # Inicialización
    inicializar_bd,
    # Usuarios
    obtener_usuario_por_email, agregar_usuario, listar_usuarios,
    actualizar_password, cambiar_estado_usuario, borrar_usuario,
    buscar_usuarios,                      # 👈 autocompletar
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
from db_orm import inicializar_bd_orm

# ================== App & Middlewares ==================
app = FastAPI(middleware=[
    Middleware(SessionMiddleware, secret_key="clave_secreta_super_segura")
])

# Inicializa BD SQLite (usuarios, historial, tickets, mensajes, hilos_ocultos, adjuntos)
inicializar_bd()
# Inicializa ORM (audit_logs) según DATABASE_URL
inicializar_bd_orm()

# Static
# Si tus archivos están en backend/static, cambia a directory="backend/static"
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated_pdfs", StaticFiles(directory="generated_pdfs"), name="generated_pdfs")

templates = Jinja2Templates(directory="templates")
templates.env.globals['os'] = os

# ================== Guardas/Dependencias de auth/roles ==================
def require_auth(request: Request):
    """Obliga a tener sesión iniciada."""
    if not request.session.get("usuario"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")

def require_admin(request: Request):
    """
    Requiere rol admin.
    Valida primero contra sesión y, por robustez, revalida en BD si la sesión dice que no.
    """
    email = request.session.get("usuario")
    if not email:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autenticado")
    rol = request.session.get("rol")
    if rol == "admin":
        return
    # Fallback: por si el rol cambió en BD y la sesión quedó vieja
    try:
        if es_admin(email):
            request.session["rol"] = "admin"  # refrescamos sesión
            return
    except Exception:
        pass
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Solo admins")

# ================== Alert/WS manager ==================
class ConnectionManager:
    """
    Mantiene websockets por usuario (email). Permite enviar eventos en tiempo real.
    """
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
        # Limpieza
        for ws in dead:
            self._by_user.get(email, set()).discard(ws)

    async def broadcast(self, payload: dict):
        for email in list(self._by_user.keys()):
            await self.send_to_user(email, payload)

manager = ConnectionManager()

def _get_ws_email(websocket: WebSocket) -> str:
    """
    Intenta obtener el email de la sesión. Si no, toma query param ?email=.
    """
    email = None
    try:
        # Disponible gracias a SessionMiddleware
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
            # Mantener la conexión viva (recibir pings del front).
            _ = await websocket.receive_text()
            # Si querés responder a pings:
            try:
                await websocket.send_json({"event": "ws:pong", "ts": datetime.utcnow().isoformat() + "Z"})
            except Exception:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, email)
    except Exception:
        manager.disconnect(websocket, email)

async def emit_alert(email: str, title: str, body: str = "", extra: dict = None):
    """
    Empuja una alerta genérica a un usuario.
    """
    payload = {"event": "alert:new", "title": title, "body": body, "ts": datetime.utcnow().isoformat() + "Z"}
    if extra:
        payload["extra"] = extra
    await manager.send_to_user(email, payload)

async def emit_chat_new_message(para_email: str, de_email: str, msg_id: int, preview: str = ""):
    payload = {
        "event": "chat:new_message",
        "from": de_email,
        "id": msg_id,
        "preview": preview[:120],
        "ts": datetime.utcnow().isoformat() + "Z"
    }
    await manager.send_to_user(para_email, payload)

# ================== Archivos de chat (adjuntos) ==================
CHAT_ATTACH_DIR = os.path.join("static", "chat_adjuntos")
os.makedirs(CHAT_ATTACH_DIR, exist_ok=True)

# Límites y validaciones para adjuntos del chat (alineados con el front)
CHAT_ALLOWED_EXT = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".txt", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".pptx"
}
CHAT_MAX_FILES = 10
CHAT_MAX_TOTAL_MB = 50

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

async def _save_upload_stream(upload: UploadFile, dst_path: str) -> int:
    """Guarda el UploadFile en disco por chunks. Devuelve tamaño en bytes."""
    size = 0
    with open(dst_path, "wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)  # 1MB
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
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    usuario = obtener_usuario_por_email(email)
    if usuario and usuario[3] == password:
        # Guardamos ambas claves para compatibilidad con plantillas
        request.session["usuario"] = usuario[2]   # email
        request.session["email"] = usuario[2]
        request.session["rol"] = usuario[4]
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Credenciales incorrectas"})

# Logout por POST (existente) y por GET (para el link del menú)
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
    usuario = request.session.get("usuario", "Anónimo")
    texto_total = ""
    for archivo in archivos:
        texto = extraer_texto_de_pdf(archivo)
        texto_total += texto + "\n\n"
    resumen = analizar_con_openai(texto_total)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    nombre_archivo = f"resumen_{timestamp}.pdf"
    generar_pdf_con_plantilla(resumen, nombre_archivo)

    guardar_en_historial(timestamp, usuario, nombre_archivo, nombre_archivo, resumen)
    return {"resumen": resumen, "pdf": nombre_archivo}

# ================== Historial ==================
@app.get("/historial")
async def ver_historial():
    return JSONResponse(obtener_historial())

# 🔁 Alias de vista para el botón de la barra lateral
@app.get("/historia")
async def alias_historia():
    return RedirectResponse("/?goto=historial", status_code=307)

# 🔁 Alias para “nuevo análisis”
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
    return {
        "usuario": request.session.get("usuario", "Desconocido"),
        "rol": request.session.get("rol", "usuario")
    }

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
        # Adjuntos
        carpeta_adjuntos = os.path.join("static", "adjuntos_incidencias")
        prefix = f"{t[1]}_{t[6].replace(':','').replace('-','').replace(' ','')[:14]}"
        adjuntos = []
        if os.path.exists(carpeta_adjuntos):
            for file in os.listdir(carpeta_adjuntos):
                if file.startswith(prefix):
                    adjuntos.append(file)
        tickets.append({
            "id": t[0],
            "usuario": t[1],
            "titulo": t[2],
            "descripcion": t[3],
            "tipo": t[4],
            "estado": t[5],
            "fecha": t[6],
            "fecha_legible": fecha_legible,
            "adjuntos": adjuntos
        })
    return templates.TemplateResponse("incidencias.html", {
        "request": request,
        "tickets": tickets,
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
        "request": request,
        "mensaje": "",
        "error": ""
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
            "request": request,
            "mensaje": "",
            "error": "La contraseña actual es incorrecta."
        })
    if new_password != confirm_password:
        return templates.TemplateResponse("cambiar_password.html", {
            "request": request,
            "mensaje": "",
            "error": "La nueva contraseña no coincide en ambos campos."
        })
    actor_user_id, ip = _actor_info(request)
    actualizar_password(usuario, new_password, actor_user_id=actor_user_id, ip=ip)
    return templates.TemplateResponse("cambiar_password.html", {
        "request": request,
        "mensaje": "Contraseña cambiada correctamente.",
        "error": ""
    })

# ================== Admin (todas las rutas protegidas) ==================
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
        ok = borrar_usuario(email, actor_user_id=actor_user_id, ip=ip, soft=False)  # borrado duro
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

    historial = obtener_historial_completo()
    ultimo_analisis_usuario = next(
        (h for h in historial if h["usuario"] == usuario_actual and h["resumen"]),
        None
    )

    if ultimo_analisis_usuario:
        ultimo_resumen = f"""
📌 Último análisis del usuario actual:
- Fecha: {ultimo_analisis_usuario['fecha']}
- Archivo: {ultimo_analisis_usuario['nombre_archivo']}
- Resumen:
{ultimo_analisis_usuario['resumen']}
"""
    else:
        ultimo_resumen = "(El usuario aún no tiene análisis registrados.)"

    contexto_general = "\n".join([
        f"- [{h['fecha']}] {h['usuario']} analizó '{h['nombre_archivo']}' y obtuvo:\n{h['resumen']}\n"
        for h in historial if h['resumen']
    ])

    contexto = f"{ultimo_resumen}\n\n📚 Historial completo:\n{contexto_general}"

    respuesta = await run_in_threadpool(responder_chat_openai, mensaje, contexto, usuario_actual)
    return JSONResponse({"respuesta": respuesta})

# ================== API puente para el drawer (formato reply) ==============
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

# ===== Mini vista embebida para el widget del topbar/FAB (Enter envía) =====
@app.get("/chat_openai_embed", response_class=HTMLResponse)
async def chat_openai_embed(request: Request):
    if not request.session.get("usuario"):
        return HTMLResponse("<div style='padding:12px'>Iniciá sesión para usar el chat.</div>")
    html = """
    <!doctype html><html><head>
    <meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
    <style>
      #t{ resize:none; min-height:42px; max-height:150px; }
    </style>
    </head><body class="p-2" style="background:transparent">
      <div id="log" class="mb-2" style="height:410px; overflow:auto; background:#f6f8fb; border-radius:12px; padding:8px;"></div>
      <form id="f" class="d-flex gap-2">
        <textarea id="t" class="form-control" placeholder="Escribe tu mensaje..." autocomplete="off" autofocus></textarea>
        <button id="send" type="submit" class="btn btn-primary">Enviar</button>
      </form>
      <script>
        const log = document.getElementById('log');
        const form = document.getElementById('f');
        const ta = document.getElementById('t');

        function esc(s){ return (s||'').replaceAll('<','&lt;').replaceAll('>','&gt;'); }
        function add(b){ const p=document.createElement('div'); p.innerHTML=b; log.appendChild(p); log.scrollTop=log.scrollHeight; }

        // Auto-altura del textarea
        function autosize(){ ta.style.height='auto'; ta.style.height = Math.min(ta.scrollHeight, 150) + 'px'; }
        ta.addEventListener('input', autosize); autosize();

        // Enter = enviar | Shift+Enter = salto de línea
        ta.addEventListener('keydown', (e)=>{
          if(e.key === 'Enter' && !e.shiftKey){
            e.preventDefault();          // evita insertar salto y submit duplicado
            form.requestSubmit();        // dispara un único submit
          }
        });

        form.addEventListener('submit', async (e)=>{
          e.preventDefault();
          const v = ta.value.trim();
          if(!v) return;
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
          }
        });
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
    """Autocompletar de usuarios por nombre/email."""
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
        msg_id = enviar_mensaje(de_email=de, para_email=para, texto=texto,
                                actor_user_id=actor_user_id, ip=ip)
        # 🔔 Push en tiempo real para el receptor
        await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=texto)
        return JSONResponse({"ok": True, "id": msg_id})
    except Exception as e:
        print("❌ Error chat_enviar:", repr(e))
        return JSONResponse({"error": "No se pudo enviar el mensaje"}, status_code=500)

# ---- NUEVO: enviar mensaje con múltiples archivos -------------------------
@app.post("/chat/enviar-archivos")
async def chat_enviar_archivos(
    request: Request,
    para: str = Form(...),
    texto: str = Form(default=""),
    archivos: List[UploadFile] = File(default=[])
):
    """Envía un mensaje con 0..N adjuntos. Guarda cada archivo y registra metadatos."""
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
            # Continuamos; se podría informar un warning al front.

    # 🔔 Push en tiempo real para el receptor
    await emit_chat_new_message(para_email=para, de_email=de, msg_id=msg_id, preview=(texto or "[Adjuntos]"))
    return JSONResponse({"ok": True, "id": msg_id})

# ---- Compat: enviar mensaje con 1 archivo (reusa la lógica nueva) ---------
@app.post("/chat/enviar-archivo")
async def chat_enviar_archivo(
    request: Request,
    para: str = Form(...),
    texto: str = Form(default=""),
    archivo: UploadFile = File(...)
):
    """Compat: 1 archivo. Internamente llama a /chat/enviar-archivos."""
    if not request.session.get("usuario"):
        return JSONResponse({"error": "No autenticado"}, status_code=401)
    archivos = [archivo] if archivo and archivo.filename else []
    return await chat_enviar_archivos(request, para=para, texto=texto, archivos=archivos)

# ---- Servir adjunto por nombre --------------------------------------------
@app.get("/chat/adjunto/{filename}")
async def chat_adjunto(filename: str):
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
        hilos = obtener_hilos_para(yo)  # ya filtra ocultos y reaparece si hay mensajes nuevos
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
        print("❌ Error chat_no_leidos:", repr(e))
        return JSONResponse({"error": "No se pudo obtener el conteo"}, status_code=500)

# ---- Hilos: ocultar / restaurar / abrir ----------------------------------
@app.post("/chat/ocultar")
async def chat_ocultar(request: Request):
    """‘Eliminar chat’: oculta el hilo para el usuario actual (no borra mensajes)."""
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
        print("❌ Error chat_ocultar:", repr(e))
        return JSONResponse({"error": "No se pudo ocultar el hilo"}, status_code=500)

@app.post("/chat/restaurar")
async def chat_restaurar(request: Request):
    """Revierte el ocultamiento del hilo (vuelve a aparecer en el sidebar)."""
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
        print("❌ Error chat_restaurar:", repr(e))
        return JSONResponse({"error": "No se pudo restaurar el hilo"}, status_code=500)

@app.post("/chat/abrir")
async def chat_abrir(request: Request):
    """
    Atajo para el buscador ‘@’: restaura (si estaba oculto) y no devuelve mensajes.
    Úsalo antes de llamar a /chat/mensajes.
    """
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
        print("❌ Error chat_abrir:", repr(e))
        return JSONResponse({"error": "No se pudo abrir el hilo"}, status_code=500)

# ================== Auditoría (vista) ==================
@app.get("/auditoria", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def ver_auditoria(request: Request):
    logs = obtener_auditoria()
    return templates.TemplateResponse("auditoria.html", {
        "request": request,
        "logs": logs
    })

# =====================================================================
# ========================== CALENDARIO ===============================
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
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

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
    """
    Síncrono (compat) — guarda en BD.
    Para push en vivo, usar notify_async().
    """
    with cal_conn() as c:
        c.execute(
            "INSERT INTO notificaciones(user, titulo, cuerpo, created_at, leida) VALUES(?,?,?,?,0)",
            (user or "Desconocido", titulo, cuerpo, _now_iso())
        )

async def notify_async(user: str, titulo: str, cuerpo: str = ""):
    """
    Inserta la notificación en BD y además emite un evento en tiempo real.
    """
    _notify(user, titulo, cuerpo)
    await emit_alert(user, titulo, cuerpo)

@app.get("/calendario", response_class=HTMLResponse)
async def calendario_view(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("calendario.html", {"request": request})

@app.get("/calendario/eventos")
async def cal_list():
    with cal_conn() as c:
        cur = c.execute("SELECT * FROM eventos ORDER BY start ASC")
        rows = [ _event_row_to_dict(r) for r in cur.fetchall() ]
    return rows

# Alias para la Home (index) que espera {events:[...]}
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
    all_day = 1 if data.get("allDay") else 0
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
    if end   is not None: sets.append("end=?");   vals.append(end)
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

# ================== Notificaciones mínimas usadas por calendario.html ======
@app.get("/notificaciones")
async def list_notifs(request: Request, limit: int = 20):
    user = request.session.get("usuario", "Desconocido")
    with cal_conn() as c:
        total_unread = c.execute(
            "SELECT COUNT(1) FROM notificaciones WHERE user=? AND leida=0", (user,)
        ).fetchone()[0]
        cur = c.execute(
            "SELECT id, titulo, cuerpo, created_at, leida FROM notificaciones WHERE user=? ORDER BY id DESC LIMIT ?",
            (user, limit)
        )
        items = []
        for r in cur.fetchall():
            items.append({
                "id": r["id"],
                "titulo": r["titulo"],
                "cuerpo": r["cuerpo"],
                "fecha_legible": r["created_at"],
                "leida": bool(r["leida"])
            })
    return {"total_unread": total_unread, "items": items}

@app.post("/notificaciones/marcar-leidas")
async def mark_read(request: Request):
    user = request.session.get("usuario", "Desconocido")
    with cal_conn() as c:
        c.execute("UPDATE notificaciones SET leida=1 WHERE user=?", (user,))
    return {"ok": True}

# =====================================================================
# ========================== PRESENCIA / ONLINE =======================
# =====================================================================

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

# Llamamos a la init de presencia después de la del calendario
init_presence_db()

# ========================== Endpoints presencia ======================

@app.post("/presence/ping")
async def presence_ping(request: Request):
    """
    El front llama a este endpoint cada ~30s para marcar al usuario como activo.
    """
    email = request.session.get("usuario")
    if not email:
        return JSONResponse({"ok": False, "error": "No autenticado"}, status_code=401)

    # Intentamos sacar nombre desde la BD de usuarios
    row = obtener_usuario_por_email(email)
    nombre = row[1] if row else email

    ip = request.client.host if request.client else None
    ua = request.headers.get("user-agent", "")
    now = _now_iso()  # UTC con 'Z'

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
    return {"ok": True}


@app.get("/presence/online")
async def presence_online(minutes: int = 5):
    """
    Devuelve usuarios con last_seen dentro de los últimos N minutos (default 5).
    """
    threshold_ts = datetime.utcnow().timestamp() - (minutes * 60)

    items = []
    with cal_conn() as c:
        cur = c.execute("SELECT user, nombre, last_seen, ip, ua FROM presence ORDER BY last_seen DESC")
        for r in cur.fetchall():
            try:
                dt = datetime.strptime(r["last_seen"], "%Y-%m-%dT%H:%M:%SZ")
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
    """
    Vista HTML con el listado (usa templates/usuarios_activos.html).
    """
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    data = await presence_online(minutes=5)  # reutilizamos el JSON
    return templates.TemplateResponse("usuarios_activos.html", {
        "request": request,
        "items": data.get("items", [])
    })
