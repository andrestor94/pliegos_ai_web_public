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
    crear_ticket, obtener_tickets_por_usuario, obtener_todos_los_tickets,
    actualizar_estado_ticket, eliminar_ticket,
    obtener_historial_completo
)

# Inicialización de BD
crear_tabla_usuarios()
crear_tabla_historial()
crear_tabla_tickets()

app = FastAPI(middleware=[
    Middleware(SessionMiddleware, secret_key="clave_secreta_super_segura")
])

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated_pdfs", StaticFiles(directory="generated_pdfs"), name="generated_pdfs")

templates = Jinja2Templates(directory="templates")
templates.env.globals['os'] = os  # Habilita el acceso a 'os' desde las plantillas Jinja2

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
    tickets = obtener_todos_los_tickets() if rol == "admin" else obtener_tickets_por_usuario(usuario)
    return templates.TemplateResponse("incidencias.html", {
        "request": request,
        "tickets": [
            {
                "id": t[0],
                "usuario": t[1],
                "titulo": t[2],
                "descripcion": t[3],
                "tipo": t[4],
                "estado": t[5],
                "fecha": t[6],
                "fecha_legible": datetime.strptime(t[6], "%Y-%m-%d %H:%M:%S").strftime("%d/%m/%Y %H:%M"),
            }
            for t in tickets
        ],
        "usuario_actual": {
            "nombre": usuario,
            "rol": rol
        }
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

    crear_ticket(usuario, titulo, descripcion, tipo)

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
    actualizar_estado_ticket(id, "Cerrado")
    return RedirectResponse("/incidencias", status_code=303)

@app.post("/incidencias/eliminar/{id}")
async def eliminar_incidencia_form(request: Request, id: int):
    if request.session.get("rol") != "admin":
        return JSONResponse({"error": "Acceso denegado"}, status_code=403)
    eliminar_ticket(id)
    return RedirectResponse("/incidencias", status_code=303)

# ------------------- ADMIN -------------------
@app.get("/admin", response_class=HTMLResponse)
async def vista_admin(request: Request):
    if request.session.get("rol") != "admin":
        return RedirectResponse("/")
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/admin/usuarios")
async def listar_usuarios_api():
    return JSONResponse(listar_usuarios())

@app.post("/admin/crear-usuario")
async def crear_usuario_api(data: dict):
    try:
        agregar_usuario(data["nombre"], data["email"], "1234", data["rol"])
        return JSONResponse({"mensaje": "Usuario creado correctamente con contraseña: 1234"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/admin/blanquear-password")
async def blanquear_password(data: dict):
    try:
        actualizar_password(data["email"], "1234")
        return JSONResponse({"mensaje": "Contraseña blanqueada a 1234"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/admin/desactivar-usuario")
async def desactivar_usuario(data: dict):
    cambiar_estado_usuario(data["email"], 0)
    return JSONResponse({"mensaje": "Usuario desactivado"})

@app.post("/admin/activar-usuario")
async def activar_usuario(data: dict):
    cambiar_estado_usuario(data["email"], 1)
    return JSONResponse({"mensaje": "Usuario activado"})

@app.post("/admin/eliminar-usuario")
async def eliminar_usuario(data: dict):
    try:
        conn = sqlite3.connect("usuarios.db")
        conn.execute("DELETE FROM usuarios WHERE email = ?", (data["email"],))
        conn.commit()
        conn.close()
        return JSONResponse({"mensaje": "Usuario eliminado correctamente"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

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
