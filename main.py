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
    ocultar_hilo, restaurar_hilo
)

# ORM (audit_logs)
from db_orm import inicializar_bd_orm

# ================== App & Middlewares ==================
app = FastAPI(middleware=[
    Middleware(SessionMiddleware, secret_key="clave_secreta_super_segura")
])

# Inicializa BD SQLite (usuarios, historial, tickets, mensajes, hilos_ocultos)
inicializar_bd()
# Inicializa ORM (audit_logs) según DATABASE_URL
inicializar_bd_orm()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated_pdfs", StaticFiles(directory="generated_pdfs"), name="generated_pdfs")

templates = Jinja2Templates(directory="templates")
templates.env.globals['os'] = os

# ================== Helpers ==================
def _actor_info(request: Request):
    email = request.session.get("usuario")
    row = obtener_usuario_por_email(email) if email else None
    actor_user_id = row[0] if row else None
    ip = request.client.host if request.client else None
    return actor_user_id, ip

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
        request.session["usuario"] = usuario[2]
        request.session["rol"] = usuario[4]
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Credenciales incorrectas"})

@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

# ================== Análisis ==================
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
    generar_pdf_con_plantilla(resumen, nombre_archivo)

    guardar_en_historial(timestamp, usuario, nombre_archivo, nombre_archivo, resumen)
    return {"resumen": resumen, "pdf": nombre_archivo}

# ================== Historial ==================
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

# ================== Admin ==================
@app.get("/admin", response_class=HTMLResponse)
async def vista_admin(request: Request):
    if request.session.get("rol") != "admin":
        return RedirectResponse("/")
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/admin/usuarios")
async def listar_usuarios_api():
    return JSONResponse(listar_usuarios())

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
        for h in historial if h["resumen"]
    ])

    contexto = f"{ultimo_resumen}\n\n📚 Historial completo:\n{contexto_general}"

    respuesta = await run_in_threadpool(responder_chat_openai, mensaje, contexto, usuario_actual)
    return JSONResponse({"respuesta": respuesta})

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
        return JSONResponse({"ok": True, "id": msg_id})
    except Exception as e:
        print("❌ Error chat_enviar:", repr(e))
        return JSONResponse({"error": "No se pudo enviar el mensaje"}, status_code=500)

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
        return JSONResponse({"error": "Falta 'de' (email remitente)"}, status_code=400)
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

# ---- NUEVO: ocultar / restaurar / abrir hilos ----------------------------
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
@app.get("/auditoria", response_class=HTMLResponse)
async def ver_auditoria(request: Request):
    if request.session.get("rol") != "admin":
        return RedirectResponse("/")
    logs = obtener_auditoria()
    return templates.TemplateResponse("auditoria.html", {
        "request": request,
        "logs": logs
    })
