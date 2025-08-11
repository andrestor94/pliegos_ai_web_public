import sqlite3
import re
import time
import json
from datetime import datetime
from db_orm import SessionLocal, AuditLog, Usuario  # para auditoría y join con email/nombre

DB_PATH = "usuarios.db"

# ---------------------- Conexión robusta SQLite ----------------------
def _get_conn():
    """
    Conexión SQLite con WAL y busy_timeout para reducir 'database is locked'.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")  # 5s de espera
    except Exception:
        pass
    return conn

def _with_retry(callable_fn, retries=5, base_delay=0.15):
    """
    Reintenta operaciones si SQLite está bloqueada.
    """
    for i in range(retries):
        try:
            return callable_fn()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < retries - 1:
                time.sleep(base_delay * (i + 1))
                continue
            raise

# --------------------------- Auditoría ORM ---------------------------
def registrar_auditoria(actor_user_id, action, entity, entity_id, before=None, after=None, ip=None):
    """
    Inserta en audit_logs (SQLAlchemy). Funciona con SQLite local o Postgres (Render).
    """
    with SessionLocal() as session:
        log = AuditLog(
            actor_user_id=actor_user_id,
            action=action,
            entity=entity,
            entity_id=entity_id,
            before_json=json.dumps(before, ensure_ascii=False) if before else None,
            after_json=json.dumps(after, ensure_ascii=False) if after else None,
            ip=ip
        )
        session.add(log)
        session.commit()

# ---------------------- Inicialización de Tablas ----------------------
def inicializar_bd():
    crear_tabla_usuarios()
    crear_tabla_historial()
    crear_tabla_tickets()

def crear_tabla_usuarios():
    with _get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                rol TEXT NOT NULL DEFAULT 'usuario',
                activo INTEGER NOT NULL DEFAULT 1
            )
        ''')

def crear_tabla_historial():
    with _get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS historial (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                usuario TEXT NOT NULL,
                nombre_archivo TEXT NOT NULL,
                ruta_pdf TEXT NOT NULL,
                resumen_texto TEXT
            )
        ''')

def crear_tabla_tickets():
    with _get_conn() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario TEXT NOT NULL,
                titulo TEXT NOT NULL,
                descripcion TEXT NOT NULL,
                tipo TEXT NOT NULL,
                estado TEXT NOT NULL DEFAULT 'Abierto',
                fecha TEXT NOT NULL
            )
        ''')

# --------------------------- Gestión Usuarios ---------------------------
def agregar_usuario(nombre, email, password, rol="usuario", actor_user_id=None, ip=None):
    try:
        with _get_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO usuarios (nombre, email, password, rol) VALUES (?, ?, ?, ?)",
                (nombre, email, password, rol)
            )
            new_id = cursor.lastrowid
        registrar_auditoria(
            actor_user_id, "CREATE_USER", "usuarios", new_id,
            after={"nombre": nombre, "email": email, "rol": rol, "activo": True}, ip=ip
        )
    except sqlite3.IntegrityError:
        print(f"⚠️ El usuario con email {email} ya existe.")

def obtener_usuario_por_email(email):
    with _get_conn() as conn:
        cursor = conn.execute("SELECT * FROM usuarios WHERE email = ?", (email,))
        return cursor.fetchone()

def listar_usuarios():
    with _get_conn() as conn:
        cursor = conn.execute("SELECT id, nombre, email, rol, activo FROM usuarios")
        usuarios = []
        for row in cursor.fetchall():
            usuarios.append({
                "id": row[0],
                "nombre": row[1],
                "email": row[2],
                "rol": row[3],
                "activo": bool(row[4])
            })
        return usuarios

def actualizar_password(email, nueva_password, actor_user_id=None, ip=None):
    with _get_conn() as conn:
        before = obtener_usuario_por_email(email)
        conn.execute("UPDATE usuarios SET password = ? WHERE email = ?", (nueva_password, email))
        after = obtener_usuario_por_email(email)
    if before:
        registrar_auditoria(
            actor_user_id, "UPDATE_PASSWORD", "usuarios", before[0],
            before={"email": before[2]}, after={"email": after[2]}, ip=ip
        )

def cambiar_estado_usuario(email, activo, actor_user_id=None, ip=None):
    with _get_conn() as conn:
        before = obtener_usuario_por_email(email)
        conn.execute("UPDATE usuarios SET activo = ? WHERE email = ?", (activo, email))
        after = obtener_usuario_por_email(email)
    if before:
        registrar_auditoria(
            actor_user_id, "TOGGLE_USER_ACTIVE", "usuarios", before[0],
            before={"activo": bool(before[5])}, after={"activo": bool(after[5])}, ip=ip
        )

def borrar_usuario(email, actor_user_id=None, ip=None, soft=True):
    """
    Soft delete (activo=0, rol='borrado') o hard delete si soft=False.
    IMPORTANTE: cerramos la conexión SQLite ANTES de registrar auditoría
    para evitar 'database is locked' al abrir otra conexión (SQLAlchemy).
    """
    def _op():
        # 1) Leer estado previo (abre/cierra su propia conexión)
        before = obtener_usuario_por_email(email)
        if not before:
            return False
        user_id = before[0]

        if soft:
            # 2) Ejecutar actualización y CERRAR conexión
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE usuarios SET activo = 0, rol = 'borrado' WHERE email = ?",
                    (email,)
                )
            # 3) Obtener estado luego de cerrar la conexión y registrar auditoría
            after = obtener_usuario_por_email(email)
            registrar_auditoria(
                actor_user_id, "SOFT_DELETE_USER", "usuarios", user_id,
                before={"email": before[2], "rol": before[4], "activo": bool(before[5])},
                after={"email": after[2], "rol": after[4], "activo": bool(after[5])} if after else None,
                ip=ip
            )
        else:
            # 2) Ejecutar borrado duro y CERRAR conexión
            with _get_conn() as conn:
                conn.execute("DELETE FROM usuarios WHERE email = ?", (email,))
            # 3) Registrar auditoría después de cerrar conexión
            registrar_auditoria(
                actor_user_id, "HARD_DELETE_USER", "usuarios", user_id,
                before={"email": before[2], "rol": before[4], "activo": bool(before[5])},
                ip=ip
            )
        return True

    return _with_retry(_op)

# --------------------------- Historial ---------------------------
def guardar_en_historial(timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto=""):
    try:
        match = re.search(r"(\d{14})", timestamp)
        if match:
            timestamp = match.group(1)
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        with _get_conn() as conn:
            conn.execute('''
                INSERT INTO historial (timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto))
    except Exception as e:
        print(f"❌ Error al guardar en historial: {e}")

def obtener_historial():
    with _get_conn() as conn:
        cursor = conn.execute('''
            SELECT id, timestamp, usuario, nombre_archivo, ruta_pdf
            FROM historial
            ORDER BY id DESC
        ''')
        historial = []
        for row in cursor.fetchall():
            try:
                fecha_legible = datetime.strptime(row[1], "%Y%m%d%H%M%S").strftime("%d/%m/%Y %H:%M")
            except ValueError:
                fecha_legible = "Fecha inválida"
            historial.append({
                "id": row[0],
                "timestamp": row[1],
                "usuario": row[2],
                "nombre_archivo": row[3],
                "ruta_pdf": row[4],
                "fecha_legible": fecha_legible
            })
        return historial

def obtener_historial_completo():
    with _get_conn() as conn:
        cursor = conn.execute('''
            SELECT timestamp, usuario, nombre_archivo, resumen_texto
            FROM historial
            ORDER BY id DESC
        ''')
        historial = []
        for row in cursor.fetchall():
            try:
                fecha_legible = datetime.strptime(row[0], "%Y%m%d%H%M%S").strftime("%d/%m/%Y %H:%M")
            except ValueError:
                fecha_legible = "Fecha inválida"
            historial.append({
                "timestamp": row[0],
                "usuario": row[1],
                "nombre_archivo": row[2],
                "resumen": row[3],
                "fecha": fecha_legible
            })
        return historial

def eliminar_del_historial(timestamp):
    with _get_conn() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM historial WHERE timestamp = ?", (timestamp,))
        if cursor.fetchone()[0]:
            conn.execute("DELETE FROM historial WHERE timestamp = ?", (timestamp,))

def limpiar_historial_invalido():
    with _get_conn() as conn:
        cursor = conn.execute("SELECT id, timestamp FROM historial")
        ids_invalidos = []
        for id_, ts in cursor.fetchall():
            try:
                datetime.strptime(ts, "%Y%m%d%H%M%S")
            except (ValueError, TypeError):
                ids_invalidos.append(id_)
        for id_ in ids_invalidos:
            conn.execute("DELETE FROM historial WHERE id = ?", (id_,))
        print(f"🧹 Registros eliminados: {len(ids_invalidos)}")

# --------------------------- Tickets ---------------------------
def crear_ticket(usuario, titulo, descripcion, tipo, actor_user_id=None, ip=None):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        cursor = conn.execute(
            "INSERT INTO tickets (usuario, titulo, descripcion, tipo, fecha) VALUES (?, ?, ?, ?, ?)",
            (usuario, titulo, descripcion, tipo, fecha)
        )
        new_id = cursor.lastrowid
    registrar_auditoria(
        actor_user_id, "CREATE_TICKET", "tickets", new_id,
        after={"usuario": usuario, "titulo": titulo, "descripcion": descripcion, "tipo": tipo, "estado": "Abierto"},
        ip=ip
    )

def obtener_tickets_por_usuario(usuario):
    with _get_conn() as conn:
        cursor = conn.execute(
            "SELECT id, usuario, titulo, descripcion, tipo, estado, fecha FROM tickets WHERE usuario = ? ORDER BY fecha DESC",
            (usuario,)
        )
        return cursor.fetchall()

def obtener_todos_los_tickets():
    with _get_conn() as conn:
        cursor = conn.execute(
            "SELECT id, usuario, titulo, descripcion, tipo, estado, fecha FROM tickets ORDER BY fecha DESC"
        )
        return cursor.fetchall()

def actualizar_estado_ticket(ticket_id, nuevo_estado, actor_user_id=None, ip=None):
    with _get_conn() as conn:
        cursor = conn.execute("SELECT usuario, titulo, estado FROM tickets WHERE id = ?", (ticket_id,))
        before = cursor.fetchone()
        conn.execute("UPDATE tickets SET estado = ? WHERE id = ?", (nuevo_estado, ticket_id))
    if before:
        registrar_auditoria(
            actor_user_id, "UPDATE_TICKET_STATE", "tickets", ticket_id,
            before={"estado": before[2]}, after={"estado": nuevo_estado}, ip=ip
        )

def agregar_ticket(timestamp, usuario, titulo, descripcion, actor_user_id=None, ip=None):
    tipo = "General"
    crear_ticket(usuario, titulo, descripcion, tipo, actor_user_id=actor_user_id, ip=ip)

def obtener_tickets():
    with _get_conn() as conn:
        cursor = conn.execute('''
            SELECT id, usuario, titulo, descripcion, tipo, estado, fecha 
            FROM tickets 
            ORDER BY fecha DESC
        ''')
        tickets = []
        for row in cursor.fetchall():
            tickets.append({
                "id": row[0],
                "usuario": row[1],
                "titulo": row[2],
                "descripcion": row[3],
                "tipo": row[4],
                "estado": row[5],
                "fecha": row[6]
            })
        return tickets

def marcar_ticket_resuelto(ticket_id, actor_user_id=None, ip=None):
    actualizar_estado_ticket(ticket_id, "Resuelto", actor_user_id=actor_user_id, ip=ip)

def eliminar_ticket(ticket_id, actor_user_id=None, ip=None):
    with _get_conn() as conn:
        cursor = conn.execute("SELECT usuario, titulo FROM tickets WHERE id = ?", (ticket_id,))
        before = cursor.fetchone()
        conn.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
    if before:
        registrar_auditoria(
            actor_user_id, "DELETE_TICKET", "tickets", ticket_id,
            before={"usuario": before[0], "titulo": before[1]}, ip=ip
        )

# --------------------------- Consultar Auditoría ---------------------------
def obtener_auditoria(limit=50):
    """
    Devuelve las últimas acciones de audit_logs, con email/nombre si existe el usuario.
    """
    with SessionLocal() as session:
        logs = (
            session.query(AuditLog, Usuario)
            .join(Usuario, Usuario.id == AuditLog.actor_user_id, isouter=True)
            .order_by(AuditLog.id.desc())
            .limit(limit)
            .all()
        )
        resultado = []
        for log, user in logs:
            resultado.append({
                "fecha": log.at.strftime("%d/%m/%Y %H:%M:%S"),
                "usuario": user.email if user else (f"ID {log.actor_user_id}" if log.actor_user_id else "-"),
                "nombre": user.nombre if user else None,
                "accion": log.action,
                "entidad": log.entity,
                "entidad_id": log.entity_id,
                "before": log.before_json,
                "after": log.after_json,
                "ip": log.ip
            })
        return resultado
