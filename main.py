import os
import sqlite3
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, JSONResponse
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
    crear_tabla_usuarios, crear_tabla_historial, crear_tabla_tickets,
    obtener_usuario_por_email, agregar_usuario,
    guardar_en_historial, obtener_historial, eliminar_del_historial,
    listar_usuarios, actualizar_password, cambiar_estado_usuario,
    crear_ticket, obtener_todos_los_tickets, obtener_tickets_por_usuario,
    actualizar_estado_ticket, eliminar_ticket,
    obtener_historial_completo, obtener_auditoria, borrar_usuario
)

# 👇 crear/verificar tablas ORM (audit_logs) al iniciar
from db_orm import inicializar_bd_orm

# Inicialización de BD SQLite (tablas existentes)
crear_tabla_usuarios()
crear_tabla_historial()
crear_tabla_tickets()

app = FastAPI(middleware=[
    Middleware(SessionMiddleware, secret_key="clave_secreta_super_segura")
])

# ✅ Verifica/crea tablas ORM (incluye audit_logs) en el motor configurado por DATABASE_URL
inicializar_bd_orm()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated_pdfs", StaticFiles(directory="generated_pdfs"), name="generated_pdfs")

templates = Jinja2Templates(directory="templates")
templates.env.globals['os'] = os  # Acceso a 'os' desde las plantillas

# ---- Helpers actor/IP ----
def _actor_info(request: Request):
    email = request.session.get("usuario")
    row = obtener_usuario_por_email(email) if email else None
    actor_user_id = row[0] if row else None
    ip = request.client.host if request.client else None
    return actor_user_id, ip

# ------------ FUNCIÓN DE ADJUNTOS PARA INCIDENCIAS ----------------
def obtener_adjuntos_para_ticket(usuario, fecha):
    carpeta_adjuntos = os.path.join("static", "adjuntos_incidencias")
    prefix = f"{usuario}_{fecha.replace(':', '').replace('-', '').replace(' ', '')[:14]}"
    adjuntos = []
    if os.path.exists(carpeta_adjuntos):
        for file in os.listdir(carpeta_adjuntos):
            if file.startswith(prefix):
                adjuntos.append(file)
    return adjuntos

# Página principal
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not request.session.get("usuario"):
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "rol": request.session.get("rol", "usuario")
    })

# Login
@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    usuario = obtener_usuario_por_email(email)
    if usuario and usuario[3] == password:
        request.session["usuario"] = usuario[2]
        request.session["rol"] = usuario[4]
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Credenciales incorrectas"})

# Logout
@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

# Análisis de pliegos
@app.post("/analizar-pliego")
async def analizar_pliego(request: Request, archivos: list[UploadFile] = File(...)):
    usuario = request.session.get("usuario", "Anónimo")
    texto_total = ""
    for archivo in archivos:
        texto = extraer_texto_de_pdf(archivo)
        texto_total += texto + "\n\n"
    resumen = analizar_con_openai(texto_total)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    nombre_archivo = f"resumen_{timestamp}.pdf"
    ruta_pdf = generar_pdf_con_plantilla(resumen, nombre_archivo)

    guardar_en_historial(timestamp, usuario, nombre_archivo, nombre_archivo, resumen)

    return {"resumen": resumen, "pdf": nombre_archivo}

# Historial
@app.get("/historial")
async def ver_historial():
    return JSONResponse(obtener_historial())

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

# Usuario actual
@app.get("/usuario-actual")
async def usuario_actual(request: Request):
    return {
        "usuario": request.session.get("usuario", "Desconocido"),
        "rol": request.session.get("rol", "usuario")
    }

# ------------------- GESTIÓN DE INCIDENCIAS -------------------
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
        adjuntos = obtener_adjuntos_para_ticket(t[1], t[6])
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
    archivos: list[UploadFile] = File(default=[])
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

# ------------------- CAMBIO DE CONTRASEÑA POR EL USUARIO -------------------
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

# ------------------- ADMIN -------------------
@app.get("/admin", response_class=HTMLResponse)
async def vista_admin(request: Request):
    if request.session.get("rol") != "admin":
        return RedirectResponse("/")
    return templates.TemplateResponse("admin.html", {"request": request})

# 📋 Auditoría
@app.get("/auditoria", response_class=HTMLResponse)
async def ver_auditoria(request: Request):
    if request.session.get("rol") != "admin":
        return RedirectResponse("/")
    logs = obtener_auditoria()
    return templates.TemplateResponse("auditoria.html", {
        "request": request,
        "logs": logs
    })

@app.get("/admin/usuarios")
async def listar_usuarios_api():
    return JSONResponse(listar_usuarios())

# ---- ENDPOINTS ADMIN (JSON body) ----
@app.post("/admin/crear-usuario")
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

@app.post("/admin/blanquear-password")
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

@app.post("/admin/desactivar-usuario")
async def desactivar_usuario(request: Request):
    data = await request.json()
    email = data.get("email")
    if not email:
        return JSONResponse({"error": "Falta email"}, status_code=400)
    actor_user_id, ip = _actor_info(request)
    cambiar_estado_usuario(email, 0, actor_user_id=actor_user_id, ip=ip)
    return JSONResponse({"mensaje": "Usuario desactivado"})

@app.post("/admin/activar-usuario")
async def activar_usuario(request: Request):
    data = await request.json()
    email = data.get("email")
    if not email:
        return JSONResponse({"error": "Falta email"}, status_code=400)
    actor_user_id, ip = _actor_info(request)
    cambiar_estado_usuario(email, 1, actor_user_id=actor_user_id, ip=ip)
    return JSONResponse({"mensaje": "Usuario activado"})

@app.post("/admin/eliminar-usuario")
async def eliminar_usuario(request: Request):
    try:
        if request.session.get("rol") != "admin":
            return JSONResponse({"error": "Acceso denegado"}, status_code=403)
        data = await request.json()
        email = data.get("email")
        if not email:
            return JSONResponse({"error": "Falta email"}, status_code=400)
        actor_user_id, ip = _actor_info(request)
        ok = borrar_usuario(email, actor_user_id=actor_user_id, ip=ip, soft=False)  # 🔥 borrado duro
        if not ok:
            return JSONResponse({"error": "Usuario no encontrado"}, status_code=404)
        return JSONResponse({"mensaje": "Usuario eliminado definitivamente."})
    except Exception as e:
        print("❌ Error eliminar-usuario:", repr(e))
        return JSONResponse({"error": f"Error al eliminar: {e}"}, status_code=400)

# ------------------- CHAT FLOTANTE -------------------
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
        for h in historial if h["resumen"]
    ])

    contexto = f"{ultimo_resumen}\n\n📚 Historial completo:\n{contexto_general}"

    respuesta = await run_in_threadpool(responder_chat_openai, mensaje, contexto, usuario_actual)

    return JSONResponse({"respuesta": respuesta})
