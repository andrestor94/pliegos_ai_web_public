[1mdiff --git a/database.py b/database.py[m
[1mindex bf64cce..ae46720 100644[m
[1m--- a/database.py[m
[1m+++ b/database.py[m
[36m@@ -1,4 +1,4 @@[m
[31m-import os[m
[32m+[m[32m﻿import os[m
 import sqlite3[m
 import re[m
 import time[m
[36m@@ -9,13 +9,13 @@[m [mfrom datetime import datetime, timezone[m
 try:[m
     from zoneinfo import ZoneInfo[m
 except Exception:[m
[31m-    ZoneInfo = None  # fallback simple si no está disponible[m
[32m+[m[32m    ZoneInfo = None  # fallback simple si no estÃ¡ disponible[m
 [m
[31m-from db_orm import SessionLocal, AuditLog, Usuario  # auditoría y join con email/nombre[m
[32m+[m[32mfrom db_orm import SessionLocal, AuditLog, Usuario  # auditorÃ­a y join con email/nombre[m
 [m
 DB_PATH = "usuarios.db"[m
 [m
[31m-# ====================== Configuración de zona horaria ======================[m
[32m+[m[32m# ====================== ConfiguraciÃ³n de zona horaria ======================[m
 APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Argentina/Buenos_Aires")[m
 [m
 ACCION_ES = {[m
[36m@@ -23,18 +23,18 @@[m [mACCION_ES = {[m
     "SOFT_DELETE_USER": "Eliminar usuario (suave)",[m
     "HARD_DELETE_USER": "Eliminar usuario (definitivo)",[m
     "TOGGLE_USER_ACTIVE": "Cambiar estado de usuario",[m
[31m-    "UPDATE_PASSWORD": "Actualizar contraseña",[m
[32m+[m[32m    "UPDATE_PASSWORD": "Actualizar contraseÃ±a",[m
     "CREATE_TICKET": "Crear ticket",[m
     "UPDATE_TICKET_STATE": "Cambiar estado de ticket",[m
     "DELETE_TICKET": "Eliminar ticket",[m
[31m-    "SEND_MESSAGE": "Enviar mensaje",  # 👈 chat interno[m
[31m-    "HIDE_THREAD": "Ocultar conversación",[m
[31m-    "UNHIDE_THREAD": "Restaurar conversación",[m
[32m+[m[32m    "SEND_MESSAGE": "Enviar mensaje",  # ðŸ‘ˆ chat interno[m
[32m+[m[32m    "HIDE_THREAD": "Ocultar conversaciÃ³n",[m
[32m+[m[32m    "UNHIDE_THREAD": "Restaurar conversaciÃ³n",[m
     # Roles[m
     "UPDATE_ROLE": "Cambiar rol de usuario",[m
[31m-    # 👇 Nuevas acciones[m
[31m-    "CREATE_ANALYSIS_RECORD": "Registrar análisis",[m
[31m-    "RATE_ANALYSIS": "Valorar análisis",[m
[32m+[m[32m    # ðŸ‘‡ Nuevas acciones[m
[32m+[m[32m    "CREATE_ANALYSIS_RECORD": "Registrar anÃ¡lisis",[m
[32m+[m[32m    "RATE_ANALYSIS": "Valorar anÃ¡lisis",[m
 }[m
 [m
 def _accion_es(codigo: str) -> str:[m
[36m@@ -56,9 +56,9 @@[m [mdef _fmt_fecha(dt_utc):[m
         local = dt_utc.astimezone(timezone(timedelta(hours=-3)))[m
     return local.strftime("%d/%m/%Y %H:%M:%S")[m
 [m
[31m-# ======================= Conexión robusta SQLite ==========================[m
[32m+[m[32m# ======================= ConexiÃ³n robusta SQLite ==========================[m
 def _get_conn():[m
[31m-    """Conexión SQLite con WAL y busy_timeout para reducir 'database is locked'."""[m
[32m+[m[32m    """ConexiÃ³n SQLite con WAL y busy_timeout para reducir 'database is locked'."""[m
     conn = sqlite3.connect(DB_PATH, timeout=10)[m
     try:[m
         conn.execute("PRAGMA journal_mode=WAL;")[m
[36m@@ -69,7 +69,7 @@[m [mdef _get_conn():[m
     return conn[m
 [m
 def _with_retry(callable_fn, retries=5, base_delay=0.15):[m
[31m-    """Reintenta operaciones si SQLite está bloqueada."""[m
[32m+[m[32m    """Reintenta operaciones si SQLite estÃ¡ bloqueada."""[m
     for i in range(retries):[m
         try:[m
             return callable_fn()[m
[36m@@ -79,7 +79,7 @@[m [mdef _with_retry(callable_fn, retries=5, base_delay=0.15):[m
                 continue[m
             raise[m
 [m
[31m-# ============================ Auditoría ORM ===============================[m
[32m+[m[32m# ============================ AuditorÃ­a ORM ===============================[m
 def registrar_auditoria(actor_user_id, action, entity, entity_id, before=None, after=None, ip=None):[m
     """Inserta en audit_logs (SQLAlchemy). Funciona con SQLite local o Postgres (Render)."""[m
     with SessionLocal() as session:[m
[36m@@ -95,20 +95,20 @@[m [mdef registrar_auditoria(actor_user_id, action, entity, entity_id, before=None, a[m
         session.add(log)[m
         session.commit()[m
 [m
[31m-# ===================== Inicialización de Tablas SQLite ====================[m
[32m+[m[32m# ===================== InicializaciÃ³n de Tablas SQLite ====================[m
 def inicializar_bd():[m
     crear_tabla_usuarios()[m
[31m-    _migrar_tabla_usuarios_si_falta_rol_y_activo()  # 👈 asegura columnas en DBs existentes[m
[32m+[m[32m    _migrar_tabla_usuarios_si_falta_rol_y_activo()  # ðŸ‘ˆ asegura columnas en DBs existentes[m
     _crear_indices_usuarios()[m
 [m
     crear_tabla_historial()[m
[31m-    _migrar_historial_add_rating_fields()           # 👈 agrega columnas de rating si faltan[m
[32m+[m[32m    _migrar_historial_add_rating_fields()           # ðŸ‘ˆ agrega columnas de rating si faltan[m
     _crear_indices_historial_rating()[m
 [m
     # crear_tabla_tickets()  # TODO[m
[31m-    crear_tabla_mensajes()       # 👈 chat interno[m
[31m-    crear_tabla_hilos_ocultos()  # 👈 gestión de hilos ocultos[m
[31m-    crear_tabla_adjuntos()       # 👈 adjuntos de mensajes[m
[32m+[m[32m    # crear_tabla_mensajes()  # TODO (deshabilitado en Render)[m
[32m+[m[32m    crear_tabla_hilos_ocultos()  # ðŸ‘ˆ gestiÃ³n de hilos ocultos[m
[32m+[m[32m    crear_tabla_adjuntos()       # ðŸ‘ˆ adjuntos de mensajes[m
 [m
 def crear_tabla_usuarios():[m
     with _get_conn() as conn:[m
[36m@@ -162,7 +162,7 @@[m [mdef crear_tabla_historial():[m
 [m
 def _migrar_historial_add_rating_fields():[m
     """[m
[31m-    Migra 'historial' para soportar valoración obligatoria por análisis.[m
[32m+[m[32m    Migra 'historial' para soportar valoraciÃ³n obligatoria por anÃ¡lisis.[m
     Agrega:[m
       - analisis_id TEXT[m
       - rating INTEGER (1..5)[m
[36m@@ -215,7 +215,7 @@[m [mdef agregar_usuario(nombre, email, password, rol="usuario", actor_user_id=None,[m
             after={"nombre": nombre, "email": email, "rol": rol, "activo": True}, ip=ip[m
         )[m
     except sqlite3.IntegrityError:[m
[31m-        print(f"⚠️ El usuario con email {email} ya existe.")[m
[32m+[m[32m        print(f"âš ï¸ El usuario con email {email} ya existe.")[m
 [m
 def obtener_usuario_por_email(email):[m
     with _get_conn() as conn:[m
[36m@@ -230,7 +230,7 @@[m [mdef obtener_rol_por_email(email: str) -> str:[m
         return row[0] if row else None[m
 [m
 def es_admin(email: str) -> bool:[m
[31m-    """Helper rápido para checks de UI/Backend."""[m
[32m+[m[32m    """Helper rÃ¡pido para checks de UI/Backend."""[m
     rol = obtener_rol_por_email(email)[m
     return rol == "admin"[m
 [m
[36m@@ -297,7 +297,7 @@[m [mdef cambiar_estado_usuario(email, activo, actor_user_id=None, ip=None):[m
 def cambiar_rol(email: str, nuevo_rol: str, actor_user_id=None, ip=None):[m
     """[m
     Cambia el rol del usuario ('admin' | 'usuario' | 'borrado').[m
[31m-    Auditoría incluida.[m
[32m+[m[32m    AuditorÃ­a incluida.[m
     """[m
     nuevo_rol = (nuevo_rol or "usuario").strip().lower()[m
     if nuevo_rol not in ("admin", "usuario", "borrado"):[m
[36m@@ -321,8 +321,8 @@[m [mdef cambiar_rol(email: str, nuevo_rol: str, actor_user_id=None, ip=None):[m
 def borrar_usuario(email, actor_user_id=None, ip=None, soft=True):[m
     """[m
     Soft delete (activo=0, rol='borrado') o hard delete si soft=False.[m
[31m-    IMPORTANTE: cerramos la conexión SQLite ANTES de registrar auditoría[m
[31m-    para evitar 'database is locked' al abrir otra conexión (SQLAlchemy).[m
[32m+[m[32m    IMPORTANTE: cerramos la conexiÃ³n SQLite ANTES de registrar auditorÃ­a[m
[32m+[m[32m    para evitar 'database is locked' al abrir otra conexiÃ³n (SQLAlchemy).[m
     """[m
     def _op():[m
         before = obtener_usuario_por_email(email)[m
[36m@@ -362,8 +362,8 @@[m [mdef _ahora_stamp():[m
 [m
 def guardar_en_historial(timestamp, usuario, nombre_archivo, ruta_pdf, resu