import sqlite3
import re
from datetime import datetime

DB_PATH = "usuarios.db"

# 🔧 Inicialización completa de la base de datos
def inicializar_bd():
    crear_tabla_usuarios()
    crear_tabla_historial()
    crear_tabla_tickets()

def crear_tabla_usuarios():
    with sqlite3.connect(DB_PATH) as conn:
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
    with sqlite3.connect(DB_PATH) as conn:
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
    with sqlite3.connect(DB_PATH) as conn:
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

# 👤 Gestión de usuarios
def agregar_usuario(nombre, email, password, rol="usuario"):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO usuarios (nombre, email, password, rol) VALUES (?, ?, ?, ?)",
                (nombre, email, password, rol)
            )
    except sqlite3.IntegrityError:
        print(f"⚠️ El usuario con email {email} ya existe.")

def obtener_usuario_por_email(email):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT * FROM usuarios WHERE email = ?", (email,))
        return cursor.fetchone()

def listar_usuarios():
    with sqlite3.connect(DB_PATH) as conn:
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

def actualizar_password(email, nueva_password):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE usuarios SET password = ? WHERE email = ?", (nueva_password, email))

def cambiar_estado_usuario(email, activo):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE usuarios SET activo = ? WHERE email = ?", (activo, email))

# 📄 Historial
def guardar_en_historial(timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto=""):
    try:
        match = re.search(r"(\d{14})", timestamp)
        if match:
            timestamp = match.group(1)
        else:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        print(f"📝 Guardando en historial: {timestamp}, {usuario}, {nombre_archivo}, {ruta_pdf}")

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                INSERT INTO historial (timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto))

        print("✅ Registro guardado con éxito.")

    except Exception as e:
        print(f"❌ Error al guardar en historial: {e}")

def obtener_historial():
    with sqlite3.connect(DB_PATH) as conn:
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
    with sqlite3.connect(DB_PATH) as conn:
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
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM historial WHERE timestamp = ?", (timestamp,))
        if cursor.fetchone()[0]:
            conn.execute("DELETE FROM historial WHERE timestamp = ?", (timestamp,))

# 🧹 Limpieza opcional
def limpiar_historial_invalido():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT id, timestamp FROM historial")
        ids_invalidos = []
        for id_, timestamp in cursor.fetchall():
            try:
                datetime.strptime(timestamp, "%Y%m%d%H%M%S")
            except (ValueError, TypeError):
                ids_invalidos.append(id_)
        for id_ in ids_invalidos:
            conn.execute("DELETE FROM historial WHERE id = ?", (id_,))
        print(f"🧹 Registros eliminados: {len(ids_invalidos)}")

# 🎫 Gestión de Tickets
def crear_ticket(usuario, titulo, descripcion, tipo):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO tickets (usuario, titulo, descripcion, tipo, fecha) VALUES (?, ?, ?, ?, ?)",
            (usuario, titulo, descripcion, tipo, fecha)
        )

def obtener_tickets_por_usuario(usuario):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT id, usuario, titulo, descripcion, tipo, estado, fecha FROM tickets WHERE usuario = ? ORDER BY fecha DESC",
            (usuario,)
        )
        return cursor.fetchall()

def obtener_todos_los_tickets():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT id, usuario, titulo, descripcion, tipo, estado, fecha FROM tickets ORDER BY fecha DESC"
        )
        return cursor.fetchall()

def actualizar_estado_ticket(ticket_id, nuevo_estado):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE tickets SET estado = ? WHERE id = ?", (nuevo_estado, ticket_id))

def agregar_ticket(timestamp, usuario, titulo, descripcion):
    tipo = "General"
    crear_ticket(usuario, titulo, descripcion, tipo)

def obtener_tickets():
    with sqlite3.connect(DB_PATH) as conn:
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

def marcar_ticket_resuelto(ticket_id):
    actualizar_estado_ticket(ticket_id, "Resuelto")

def eliminar_ticket(ticket_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
