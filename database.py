# =========================
# database.py — PARTE 1 / 4
# (imports, configuración, helpers, conexión SQLite, auditoría ORM,
#  inicialización y migraciones de tablas/índices)
# =========================

from __future__ import annotations

import os
import sqlite3
import re
import time
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

# zoneinfo para manejar zona horaria local (Python 3.9+)
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # fallback simple si no está disponible

# ORM para auditoría y join con email/nombre
from db_orm import SessionLocal, AuditLog, Usuario

# =============================================================================
# Configuración
# =============================================================================

DB_PATH = os.getenv("SQLITE_PATH", "usuarios.db")  # <- permite override en Render
APP_TIMEZONE = os.getenv("APP_TIMEZONE", "America/Argentina/Buenos_Aires")

ACCION_ES = {
    "CREATE_USER": "Crear usuario",
    "SOFT_DELETE_USER": "Eliminar usuario (suave)",
    "HARD_DELETE_USER": "Eliminar usuario (definitivo)",
    "TOGGLE_USER_ACTIVE": "Cambiar estado de usuario",
    "UPDATE_PASSWORD": "Actualizar contraseña",
    "CREATE_TICKET": "Crear ticket",
    "UPDATE_TICKET_STATE": "Cambiar estado de ticket",
    "DELETE_TICKET": "Eliminar ticket",
    "SEND_MESSAGE": "Enviar mensaje",
    "HIDE_THREAD": "Ocultar conversación",
    "UNHIDE_THREAD": "Restaurar conversación",
    "UPDATE_ROLE": "Cambiar rol de usuario",
    "CREATE_ANALYSIS_RECORD": "Registrar análisis",
    "RATE_ANALYSIS": "Valorar análisis",
}

# =============================================================================
# Utilidades de fecha/hora y helpers
# =============================================================================

def _accion_es(codigo: str) -> str:
    return ACCION_ES.get(codigo, codigo)

def _fmt_fecha(dt_utc: Optional[datetime]) -> str:
    """AuditLog.at se guarda en UTC. Mostramos en zona local (APP_TIMEZONE)."""
    if dt_utc is None:
        return "-"
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    if ZoneInfo:
        try:
            local = dt_utc.astimezone(ZoneInfo(APP_TIMEZONE))
        except Exception:
            local = dt_utc
    else:
        from datetime import timedelta
        local = dt_utc.astimezone(timezone(timedelta(hours=-3)))
    return local.strftime("%d/%m/%Y %H:%M:%S")

def _norm_email(email: Optional[str]) -> str:
    return (email or "").strip().lower()

# =============================================================================
# Conexión SQLite robusta
# =============================================================================

def _ensure_db_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _get_conn() -> sqlite3.Connection:
    """
    Conexión SQLite con WAL, busy_timeout y foreign_keys.
    NOTA: Cada llamada abre una conexión nueva (patrón recomendado con SQLite).
    """
    _ensure_db_dir(DB_PATH)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")  # 5s de espera si está locked
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return conn

def _with_retry(callable_fn, retries: int = 5, base_delay: float = 0.15):
    """Reintenta operaciones si SQLite está bloqueada."""
    for i in range(retries):
        try:
            return callable_fn()
        except sqlite3.OperationalError as e:
            msg = (str(e) or "").lower()
            if (("locked" in msg) or ("busy" in msg)) and i < retries - 1:
                time.sleep(base_delay * (i + 1))
                continue
            raise

# =============================================================================
# Auditoría (ORM)
# =============================================================================

def registrar_auditoria(
    actor_user_id: Optional[int],
    action: str,
    entity: str,
    entity_id: Any,
    before: Optional[Dict[str, Any]] = None,
    after: Optional[Dict[str, Any]] = None,
    ip: Optional[str] = None,
) -> None:
    """
    Inserta en audit_logs (SQLAlchemy). Funciona c/ SQLite local o Postgres.
    Se protege con try/except para no romper operaciones primarias si falla la auditoría.
    """
    try:
        with SessionLocal() as session:
            log = AuditLog(
                actor_user_id=actor_user_id,
                action=action,
                entity=entity,
                entity_id=str(entity_id) if entity_id is not None else None,
                before_json=json.dumps(before, ensure_ascii=False) if before else None,
                after_json=json.dumps(after, ensure_ascii=False) if after else None,
                ip=ip,
            )
            session.add(log)
            session.commit()
    except Exception as e:
        # Evitar romper flujo por errores de auditoría (solo log a stdout)
        print(f"⚠️ registrar_auditoria falló: {repr(e)}")

# =============================================================================
# Inicialización / Migraciones
# =============================================================================

def inicializar_bd() -> None:
    """
    Crea tablas requeridas y aplica migraciones idempotentes.
    Llamar en el startup de la app.
    """
    crear_tabla_usuarios()
    _migrar_tabla_usuarios_si_falta_rol_y_activo()
    _crear_indices_usuarios()

    crear_tabla_historial()
    _migrar_historial_add_rating_fields()
    _crear_indices_historial_rating()

    crear_tabla_tickets()

    crear_tabla_mensajes()
    _migrar_mensajes_add_leido_si_falta()
    _crear_indices_mensajes()

    crear_tabla_hilos_ocultos()
    crear_tabla_adjuntos()

# ----- Usuarios --------------------------------------------------------------

def crear_tabla_usuarios() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                rol TEXT NOT NULL DEFAULT 'usuario',
                activo INTEGER NOT NULL DEFAULT 1
            )
            """
        )

def _migrar_tabla_usuarios_si_falta_rol_y_activo() -> None:
    """Asegura que existan columnas 'rol' y 'activo' en DBs antiguas."""
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("PRAGMA table_info(usuarios)")
        cols = {r["name"] for r in cur.fetchall()}
        if "rol" not in cols:
            try:
                conn.execute("ALTER TABLE usuarios ADD COLUMN rol TEXT NOT NULL DEFAULT 'usuario'")
            except Exception:
                pass
        if "activo" not in cols:
            try:
                conn.execute("ALTER TABLE usuarios ADD COLUMN activo INTEGER NOT NULL DEFAULT 1")
            except Exception:
                pass

def _crear_indices_usuarios() -> None:
    with _get_conn() as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usuarios_email ON usuarios (email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_usuarios_rol ON usuarios (rol)")

# ----- Historial -------------------------------------------------------------

def crear_tabla_historial() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historial (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                usuario TEXT NOT NULL,
                nombre_archivo TEXT NOT NULL,
                ruta_pdf TEXT NOT NULL,
                resumen_texto TEXT
            )
            """
        )

def _migrar_historial_add_rating_fields() -> None:
    """
    Migra 'historial' para soportar valoración obligatoria por análisis.
    Agrega:
      - analisis_id TEXT
      - rating INTEGER (1..5)
      - rating_at TEXT (YYYYMMDDHHMMSS)
      - rating_required INTEGER (0/1)
    """
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("PRAGMA table_info(historial)")
        cols = {r["name"] for r in cur.fetchall()}
        if "analisis_id" not in cols:
            try:
                conn.execute("ALTER TABLE historial ADD COLUMN analisis_id TEXT")
            except Exception:
                pass
        if "rating" not in cols:
            try:
                conn.execute("ALTER TABLE historial ADD COLUMN rating INTEGER")
            except Exception:
                pass
        if "rating_at" not in cols:
            try:
                conn.execute("ALTER TABLE historial ADD COLUMN rating_at TEXT")
            except Exception:
                pass
        if "rating_required" not in cols:
            try:
                conn.execute("ALTER TABLE historial ADD COLUMN rating_required INTEGER NOT NULL DEFAULT 0")
            except Exception:
                pass

def _crear_indices_historial_rating() -> None:
    with _get_conn() as conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_historial_usuario ON historial (usuario)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_historial_usuario_pending ON historial (usuario, rating_required)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_historial_analisis_id ON historial (analisis_id)")

# ----- Tickets ---------------------------------------------------------------

def crear_tabla_tickets() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario TEXT NOT NULL,
                titulo TEXT NOT NULL,
                descripcion TEXT,
                tipo TEXT,
                estado TEXT NOT NULL DEFAULT 'Abierto',
                fecha TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_usuario ON tickets (usuario)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_fecha ON tickets (fecha)")

# ----- Chat interno (mensajes, hilos ocultos, adjuntos) ----------------------

def crear_tabla_mensajes() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mensajes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                de_email   TEXT NOT NULL,
                para_email TEXT NOT NULL,
                texto      TEXT NOT NULL,
                fecha      TEXT NOT NULL,                 -- 'YYYY-MM-DD HH:MM:SS'
                leido      INTEGER NOT NULL DEFAULT 0     -- 0 no leído / 1 leído
            )
            """
        )

def _migrar_mensajes_add_leido_si_falta() -> None:
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("PRAGMA table_info(mensajes)")
        cols = {r["name"] for r in cur.fetchall()}
        if "leido" not in cols:
            try:
                conn.execute("ALTER TABLE mensajes ADD COLUMN leido INTEGER NOT NULL DEFAULT 0")
            except Exception:
                pass

def _crear_indices_mensajes() -> None:
    with _get_conn() as conn:
        # Para contar no leídos y bandeja de entrada
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msj_para_leido ON mensajes (para_email, leido)")
        # Para hilos y listados
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msj_hilo ON mensajes (de_email, para_email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_msj_fecha ON mensajes (fecha)")

def crear_tabla_hilos_ocultos() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hilos_ocultos (
                owner_email TEXT NOT NULL,
                otro_email  TEXT NOT NULL,
                hidden_at   TEXT NOT NULL,
                PRIMARY KEY(owner_email, otro_email)
            )
            """
        )

def crear_tabla_adjuntos() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mensajes_adjuntos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mensaje_id INTEGER NOT NULL,
                filename   TEXT NOT NULL,   -- nombre guardado en disco
                original   TEXT,            -- nombre original
                mime       TEXT,
                size       INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(mensaje_id) REFERENCES mensajes(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_adj_mensaje ON mensajes_adjuntos (mensaje_id)")
# =========================
# database.py — PARTE 2 / 4
# (Usuarios: CRUD + helpers)
# =========================

# =============================================================================
# Usuarios
# =============================================================================

def agregar_usuario(
    nombre: str,
    email: str,
    password: str,
    rol: str = "usuario",
    actor_user_id: Optional[int] = None,
    ip: Optional[str] = None,
) -> Optional[int]:
    email = _norm_email(email)
    rol = (rol or "usuario").strip().lower()
    if rol not in ("admin", "usuario", "borrado"):
        rol = "usuario"

    try:
        with _get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO usuarios (nombre, email, password, rol) VALUES (?, ?, ?, ?)",
                (nombre, email, password, rol),
            )
            new_id = cur.lastrowid
        registrar_auditoria(
            actor_user_id,
            "CREATE_USER",
            "usuarios",
            new_id,
            after={"nombre": nombre, "email": email, "rol": rol, "activo": True},
            ip=ip,
        )
        return new_id
    except sqlite3.IntegrityError:
        print(f"⚠️ El usuario con email {email} ya existe.")
        return None

def obtener_usuario_por_email(email: str):
    """Devuelve la fila completa del usuario (tupla): (id, nombre, email, password, rol, activo)"""
    email = _norm_email(email)
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT id, nombre, email, password, rol, activo FROM usuarios WHERE email = ?",
            (email,),
        )
        return cur.fetchone()

def obtener_rol_por_email(email: str) -> Optional[str]:
    """Devuelve 'admin' / 'usuario' / 'borrado' o None si no existe."""
    email = _norm_email(email)
    with _get_conn() as conn:
        cur = conn.execute("SELECT rol FROM usuarios WHERE email = ?", (email,))
        row = cur.fetchone()
        return row[0] if row else None

def es_admin(email: str) -> bool:
    """Helper rápido para checks de UI/Backend."""
    return obtener_rol_por_email(email) == "admin"

def listar_usuarios():
    """
    Devuelve lista de dicts con campos básicos.
    (main.py tolera dicts o tuplas gracias a su _user_row_to_dict)
    """
    with _get_conn() as conn:
        cur = conn.execute("SELECT id, nombre, email, rol, activo FROM usuarios")
        return [
            {
                "id": row[0],
                "nombre": row[1],
                "email": row[2],
                "rol": row[3],
                "activo": bool(row[4]),
            }
            for row in cur.fetchall()
        ]

def buscar_usuarios(term: str, limit: int = 8):
    """Autocompletar por nombre o email (case-insensitive), solo activos."""
    like = f"%{(term or '').strip()}%"
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT id, nombre, email
              FROM usuarios
             WHERE activo = 1
               AND (LOWER(nombre) LIKE LOWER(?) OR LOWER(email) LIKE LOWER(?))
             ORDER BY 
                CASE 
                    WHEN LOWER(email)  LIKE LOWER(?) THEN 0
                    WHEN LOWER(nombre) LIKE LOWER(?) THEN 1
                    ELSE 2
                END,
                nombre ASC
             LIMIT ?
            """,
            (like, like, like, like, limit),
        )
        return [dict(r) for r in cur.fetchall()]

def actualizar_password(
    email: str,
    nueva_password: str,
    actor_user_id: Optional[int] = None,
    ip: Optional[str] = None,
) -> None:
    email = _norm_email(email)
    with _get_conn() as conn:
        before = obtener_usuario_por_email(email)
        conn.execute("UPDATE usuarios SET password = ? WHERE email = ?", (nueva_password, email))
        after = obtener_usuario_por_email(email)

    if before:
        registrar_auditoria(
            actor_user_id,
            "UPDATE_PASSWORD",
            "usuarios",
            before[0],
            before={"email": before[2]},
            after={"email": after[2]} if after else None,
            ip=ip,
        )

def cambiar_estado_usuario(
    email: str,
    activo: int,
    actor_user_id: Optional[int] = None,
    ip: Optional[str] = None,
) -> None:
    email = _norm_email(email)
    with _get_conn() as conn:
        before = obtener_usuario_por_email(email)
        conn.execute("UPDATE usuarios SET activo = ? WHERE email = ?", (1 if activo else 0, email))
        after = obtener_usuario_por_email(email)

    if before:
        registrar_auditoria(
            actor_user_id,
            "TOGGLE_USER_ACTIVE",
            "usuarios",
            before[0],
            before={"activo": bool(before[5])},
            after={"activo": bool(after[5])} if after else None,
            ip=ip,
        )

def cambiar_rol(
    email: str,
    nuevo_rol: str,
    actor_user_id: Optional[int] = None,
    ip: Optional[str] = None,
) -> bool:
    """
    Cambia el rol del usuario ('admin' | 'usuario' | 'borrado').
    Auditoría incluida.
    """
    email = _norm_email(email)
    nuevo_rol = (nuevo_rol or "usuario").strip().lower()
    if nuevo_rol not in ("admin", "usuario", "borrado"):
        nuevo_rol = "usuario"

    with _get_conn() as conn:
        before = obtener_usuario_por_email(email)
        if not before:
            return False
        conn.execute("UPDATE usuarios SET rol = ? WHERE email = ?", (nuevo_rol, email))
        after = obtener_usuario_por_email(email)

    registrar_auditoria(
        actor_user_id,
        "UPDATE_ROLE",
        "usuarios",
        before[0],
        before={"email": before[2], "rol": before[4]},
        after={"email": after[2], "rol": after[4]} if after else None,
        ip=ip,
    )
    return True

def borrar_usuario(
    email: str,
    actor_user_id: Optional[int] = None,
    ip: Optional[str] = None,
    soft: bool = True,
) -> bool:
    """
    Soft delete (activo=0, rol='borrado') o hard delete si soft=False.
    IMPORTANTE: usamos _with_retry para evitar 'database is locked' cuando coinciden escrituras.
    """

    email = _norm_email(email)

    def _op():
        before = obtener_usuario_por_email(email)
        if not before:
            return False
        user_id = before[0]

        if soft:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE usuarios SET activo = 0, rol = 'borrado' WHERE email = ?",
                    (email,),
                )
            after = obtener_usuario_por_email(email)
            registrar_auditoria(
                actor_user_id,
                "SOFT_DELETE_USER",
                "usuarios",
                user_id,
                before={"email": before[2], "rol": before[4], "activo": bool(before[5])},
                after={"email": after[2], "rol": after[4], "activo": bool(after[5])} if after else None,
                ip=ip,
            )
        else:
            with _get_conn() as conn:
                conn.execute("DELETE FROM usuarios WHERE email = ?", (email,))
            registrar_auditoria(
                actor_user_id,
                "HARD_DELETE_USER",
                "usuarios",
                user_id,
                before={"email": before[2], "rol": before[4], "activo": bool(before[5])},
                ip=ip,
            )
        return True

    return _with_retry(_op)
# =========================
# database.py — PARTE 3 / 4
# (Historial de análisis)
# =========================

# =============================================================================
# Historial de análisis
# =============================================================================

def _ahora_stamp() -> str:
    """Devuelve timestamp local como 'YYYYMMDDHHMMSS' (compatible con historial.timestamp)."""
    return datetime.now().strftime("%Y%m%d%H%M%S")

def guardar_en_historial(
    timestamp: str,
    usuario: str,
    nombre_archivo: str,
    ruta_pdf: str,
    resumen_texto: str = "",
) -> None:
    """
    Inserta un registro “simple” en historial (sin requerir valoración).
    Conserva compatibilidad con implementaciones previas.
    """
    try:
        m = re.search(r"(\d{14})", str(timestamp or ""))
        ts = m.group(1) if m else _ahora_stamp()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO historial (timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, usuario, nombre_archivo, ruta_pdf, resumen_texto),
            )
    except Exception as e:
        print(f"❌ Error al guardar en historial: {e}")

def iniciar_analisis_historial(
    usuario: str,
    nombre_archivo: str,
    ruta_pdf: str,
    analisis_id: str,
    resumen_texto: str = "",
) -> int:
    """
    Crea un registro de análisis en 'historial' y marca rating_required=1.
    Devuelve el id autoincremental del historial.
    """
    def _op():
        ts = _ahora_stamp()
        with _get_conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO historial
                    (timestamp, usuario, nombre_archivo, ruta_pdf, resumen_texto, analisis_id, rating_required)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (ts, usuario, nombre_archivo, ruta_pdf, resumen_texto, analisis_id),
            )
            return cur.lastrowid

    historial_id = _with_retry(_op)

    # Auditoría (no asociamos actor_user_id porque viene de background)
    try:
        registrar_auditoria(
            actor_user_id=None,
            action="CREATE_ANALYSIS_RECORD",
            entity="historial",
            entity_id=historial_id,
            after={"usuario": usuario, "archivo": nombre_archivo, "analisis_id": analisis_id},
            ip=None,
        )
    except Exception:
        pass

    return historial_id

def marcar_valoracion_historial(
    historial_id: int,
    rating: int,
    actor_user_id=None,
    ip: str = None,
) -> None:
    """
    Guarda la valoración (1..5), marca rating_required=0 y registra auditoría.
    Lanza ValueError si el rating es inválido.
    """
    try:
        rating = int(rating)
    except Exception:
        raise ValueError("Rating inválido (debe ser 1..5)")
    if rating < 1 or rating > 5:
        raise ValueError("Rating inválido (debe ser 1..5)")

    def _op():
        ra = _ahora_stamp()
        with _get_conn() as conn:
            conn.execute(
                """
                UPDATE historial
                   SET rating = ?, rating_at = ?, rating_required = 0
                 WHERE id = ?
                """,
                (rating, ra, historial_id),
            )

    _with_retry(_op)

    registrar_auditoria(
        actor_user_id,
        "RATE_ANALYSIS",
        "historial",
        historial_id,
        after={"rating": rating},
        ip=ip,
    )

def tiene_valoracion_pendiente(usuario: str) -> bool:
    """True si el usuario tiene algún análisis sin valorar (rating_required=1)."""
    with _get_conn() as conn:
        cur = conn.execute(
            """
            SELECT 1
              FROM historial
             WHERE usuario = ? AND rating_required = 1
             LIMIT 1
            """,
            (usuario,),
        )
        return cur.fetchone() is not None

def obtener_historial():
    """
    Lista resumida del historial (sin texto completo del resumen).
    Devuelve: [{id, timestamp, usuario, nombre_archivo, ruta_pdf, fecha_legible}, ...]
    """
    with _get_conn() as conn:
        cur = conn.execute(
            """
            SELECT id, timestamp, usuario, nombre_archivo, ruta_pdf
              FROM historial
             ORDER BY id DESC
            """
        )
        out = []
        for row in cur.fetchall():
            ts = row[1]
            try:
                fecha_legible = datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%d/%m/%Y %H:%M")
            except Exception:
                fecha_legible = "Fecha inválida"
            out.append(
                {
                    "id": row[0],
                    "timestamp": ts,
                    "usuario": row[2],
                    "nombre_archivo": row[3],
                    "ruta_pdf": row[4],
                    "fecha_legible": fecha_legible,
                }
            )
        return out

def obtener_historial_completo():
    """
    Lista detallada del historial (incluye resumen y estado de rating).
    Devuelve: [{id, timestamp, usuario, nombre_archivo, resumen, rating, rating_at, rating_required, fecha}, ...]
    """
    with _get_conn() as conn:
        cur = conn.execute(
            """
            SELECT id, timestamp, usuario, nombre_archivo, resumen_texto, rating, rating_at, rating_required
              FROM historial
             ORDER BY id DESC
            """
        )
        out = []
        for row in cur.fetchall():
            ts = row[1]
            try:
                fecha_legible = datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%d/%m/%Y %H:%M")
            except Exception:
                fecha_legible = "Fecha inválida"
            out.append(
                {
                    "id": row[0],
                    "timestamp": ts,
                    "usuario": row[2],
                    "nombre_archivo": row[3],
                    "resumen": row[4],
                    "rating": row[5],
                    "rating_at": row[6],
                    "rating_required": bool(row[7]),
                    "fecha": fecha_legible,
                }
            )
        return out

def eliminar_del_historial(timestamp: str) -> None:
    """Elimina por timestamp exacto (string de 14 dígitos). Silencioso si no existe."""
    ts = (timestamp or "").strip()
    if not re.fullmatch(r"\d{14}", ts):
        # Para compatibilidad, si viene otra cosa intentamos igual por igualdad
        ts = timestamp
    def _op():
        with _get_conn() as conn:
            cur = conn.execute("DELETE FROM historial WHERE timestamp = ?", (ts,))
            return cur.rowcount
    _with_retry(_op)

def limpiar_historial_invalido() -> int:
    """
    Elimina filas cuyo timestamp no cumpla con 'YYYYMMDDHHMMSS'.
    Devuelve la cantidad de registros eliminados.
    """
    def _collect():
        bad_ids = []
        with _get_conn() as conn:
            cur = conn.execute("SELECT id, timestamp FROM historial")
            for _id, ts in cur.fetchall():
                try:
                    datetime.strptime(str(ts or ""), "%Y%m%d%H%M%S")
                except Exception:
                    bad_ids.append(_id)
        return bad_ids

    bad_ids = _collect()
    if not bad_ids:
        return 0

    def _op():
        with _get_conn() as conn:
            for _id in bad_ids:
                conn.execute("DELETE FROM historial WHERE id = ?", (_id,))
        return len(bad_ids)

    return _with_retry(_op)
# =========================
# database.py — PARTE 4 / 4
# (Tickets + Chat + Adjuntos + Hilos ocultos + Auditoría)
# =========================

# =============================================================================
# Tickets (CRUD mínimo)
# =============================================================================

def crear_ticket(usuario, titulo, descripcion, tipo, actor_user_id=None, ip=None):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (usuario, titulo, descripcion, tipo, fecha) VALUES (?, ?, ?, ?, ?)",
            (usuario, titulo, descripcion, tipo, fecha),
        )
        new_id = cur.lastrowid
    registrar_auditoria(
        actor_user_id,
        "CREATE_TICKET",
        "tickets",
        new_id,
        after={"usuario": usuario, "titulo": titulo, "descripcion": descripcion, "tipo": tipo, "estado": "Abierto"},
        ip=ip,
    )

def obtener_tickets_por_usuario(usuario):
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT id, usuario, titulo, descripcion, tipo, estado, fecha "
            "FROM tickets WHERE usuario = ? ORDER BY fecha DESC",
            (usuario,),
        )
        return cur.fetchall()

def obtener_todos_los_tickets():
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT id, usuario, titulo, descripcion, tipo, estado, fecha "
            "FROM tickets ORDER BY fecha DESC"
        )
        return cur.fetchall()

def actualizar_estado_ticket(ticket_id, nuevo_estado, actor_user_id=None, ip=None):
    with _get_conn() as conn:
        cur = conn.execute("SELECT usuario, titulo, estado FROM tickets WHERE id = ?", (ticket_id,))
        before = cur.fetchone()
        conn.execute("UPDATE tickets SET estado = ? WHERE id = ?", (nuevo_estado, ticket_id))
    if before:
        registrar_auditoria(
            actor_user_id,
            "UPDATE_TICKET_STATE",
            "tickets",
            ticket_id,
            before={"estado": before[2]},
            after={"estado": nuevo_estado},
            ip=ip,
        )

def agregar_ticket(timestamp, usuario, titulo, descripcion, actor_user_id=None, ip=None):
    # Alias legado (se ignora timestamp)
    crear_ticket(usuario, titulo, descripcion, "General", actor_user_id=actor_user_id, ip=ip)

def obtener_tickets():
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT id, usuario, titulo, descripcion, tipo, estado, fecha "
            "FROM tickets ORDER BY fecha DESC"
        )
        out = []
        for row in cur.fetchall():
            out.append(
                {
                    "id": row[0],
                    "usuario": row[1],
                    "titulo": row[2],
                    "descripcion": row[3],
                    "tipo": row[4],
                    "estado": row[5],
                    "fecha": row[6],
                }
            )
        return out

def marcar_ticket_resuelto(ticket_id, actor_user_id=None, ip=None):
    actualizar_estado_ticket(ticket_id, "Resuelto", actor_user_id=actor_user_id, ip=ip)

def eliminar_ticket(ticket_id, actor_user_id=None, ip=None):
    with _get_conn() as conn:
        cur = conn.execute("SELECT usuario, titulo FROM tickets WHERE id = ?", (ticket_id,))
        before = cur.fetchone()
        conn.execute("DELETE FROM tickets WHERE id = ?", (ticket_id,))
    if before:
        registrar_auditoria(
            actor_user_id,
            "DELETE_TICKET",
            "tickets",
            ticket_id,
            before={"usuario": before[0], "titulo": before[1]},
            ip=ip,
        )

# =============================================================================
# Chat interno — Operaciones
# =============================================================================

def enviar_mensaje(de_email: str, para_email: str, texto: str, actor_user_id=None, ip=None) -> int:
    """Guarda un mensaje 1:1 y registra auditoría; también ‘desoculta’ el hilo del emisor."""
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO mensajes (de_email, para_email, texto, fecha) VALUES (?, ?, ?, ?)",
            (de_email, para_email, texto, fecha),
        )
        msg_id = cur.lastrowid

    # Al enviar, restauramos el hilo para el emisor (si estaba oculto).
    restaurar_hilo(de_email, para_email, actor_user_id=actor_user_id, ip=ip, silent=True)

    preview = (texto[:120] + "…") if len(texto) > 120 else texto
    registrar_auditoria(
        actor_user_id,
        "SEND_MESSAGE",
        "mensajes",
        msg_id,
        after={"de": de_email, "para": para_email, "texto": preview},
        ip=ip,
    )
    return msg_id

def obtener_hilos_para(email: str):
    """
    Devuelve lista de hilos [{con, ultima_fecha}], sin los ocultos por 'email'
    salvo que haya mensajes posteriores al ocultamiento.
    """
    with _get_conn() as conn:
        cur = conn.execute(
            """
            SELECT otro, MAX(fecha) AS ultima_fecha
              FROM (
                SELECT para_email AS otro, fecha FROM mensajes WHERE de_email = ?
                UNION ALL
                SELECT de_email  AS otro, fecha FROM mensajes WHERE para_email = ?
              ) sub
             GROUP BY otro
             ORDER BY ultima_fecha DESC
            """,
            (email, email),
        )
        hilos = [{"con": r[0], "ultima_fecha": r[1]} for r in cur.fetchall()]

        res = []
        for h in hilos:
            hidden_at = es_hilo_oculto(email, h["con"])
            if not hidden_at:
                res.append(h)
                continue
            # Reaparece si hubo mensajes después de ocultarlo
            cur2 = conn.execute(
                """
                SELECT 1
                  FROM mensajes
                 WHERE (
                        (de_email = ? AND para_email = ?)
                     OR (de_email = ? AND para_email = ?)
                       )
                   AND fecha > ?
                 LIMIT 1
                """,
                (email, h["con"], h["con"], email, hidden_at),
            )
            if cur2.fetchone():
                res.append(h)
        return res

# ---------------- Adjuntos ----------------

def guardar_adjunto(mensaje_id: int, filename: str, original: str, mime: str = None, size: int = None):
    """Registra metadatos del archivo subido para un mensaje."""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO mensajes_adjuntos (mensaje_id, filename, original, mime, size, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (mensaje_id, filename, original, mime, size, created_at),
        )
        return cur.lastrowid

def obtener_adjuntos_por_mensaje(mensaje_id: int):
    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, filename, original, mime, size, created_at "
            "FROM mensajes_adjuntos WHERE mensaje_id = ?",
            (mensaje_id,),
        )
        return [dict(r) for r in cur.fetchall()]

# --------------- Mensajes ---------------

def obtener_mensajes_entre(a: str, b: str, limit: int = 100):
    """Mensajes A↔B en orden cronológico (asc), incluyendo adjuntos."""
    with _get_conn() as conn:
        cur = conn.execute(
            """
            SELECT id, de_email, para_email, texto, leido, fecha
              FROM mensajes
             WHERE (de_email = ? AND para_email = ?)
                OR (de_email = ? AND para_email = ?)
             ORDER BY id DESC
             LIMIT ?
            """,
            (a, b, b, a, limit),
        )
        rows = cur.fetchall()

    rows = list(reversed(rows))  # cronológico
    out = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "de": r[1],
                "para": r[2],
                "texto": r[3],
                "leido": bool(r[4]),
                "fecha": r[5],
                "adjuntos": obtener_adjuntos_por_mensaje(r[0]),
            }
        )
    return out

def marcar_mensajes_leidos(de_email: str, para_email: str):
    """Marca como leídos todos los mensajes entrantes de 'de_email' hacia 'para_email'."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE mensajes
               SET leido = 1
             WHERE de_email = ? AND para_email = ? AND leido = 0
            """,
            (de_email, para_email),
        )

def contar_no_leidos(email: str) -> int:
    """Total de mensajes no leídos para 'email'."""
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM mensajes WHERE para_email = ? AND leido = 0",
            (email,),
        )
        row = cur.fetchone()
        return row[0] if row else 0

# =============================================================================
# Hilos ocultos (ocultar/restaurar/estado)
# =============================================================================

def ocultar_hilo(owner_email: str, otro_email: str, actor_user_id=None, ip=None):
    """
    Oculta el hilo para 'owner_email'. No borra mensajes.
    Si luego llegan mensajes nuevos, reaparece automáticamente.
    """
    hidden_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO hilos_ocultos (owner_email, otro_email, hidden_at)
            VALUES (?, ?, ?)
            ON CONFLICT(owner_email, otro_email)
            DO UPDATE SET hidden_at = excluded.hidden_at
            """,
            (owner_email, otro_email, hidden_at),
        )
    registrar_auditoria(
        actor_user_id,
        "HIDE_THREAD",
        "hilos_ocultos",
        f"{owner_email}|{otro_email}",
        after={"owner": owner_email, "otro": otro_email, "hidden_at": hidden_at},
        ip=ip,
    )

def restaurar_hilo(owner_email: str, otro_email: str, actor_user_id=None, ip=None, silent: bool = False):
    """Quita el ocultamiento del hilo para 'owner_email'."""
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM hilos_ocultos WHERE owner_email = ? AND otro_email = ?",
            (owner_email, otro_email),
        )
    if not silent:
        registrar_auditoria(
            actor_user_id,
            "UNHIDE_THREAD",
            "hilos_ocultos",
            f"{owner_email}|{otro_email}",
            before={"owner": owner_email, "otro": otro_email},
            ip=ip,
        )

def es_hilo_oculto(owner_email: str, otro_email: str):
    """Devuelve fecha de ocultamiento (str) si está oculto; None si no lo está."""
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT hidden_at FROM hilos_ocultos WHERE owner_email = ? AND otro_email = ?",
            (owner_email, otro_email),
        )
        row = cur.fetchone()
        return row[0] if row else None

# =============================================================================
# Consultar Auditoría (últimas N acciones)
# =============================================================================

def obtener_auditoria(limit: int = 50):
    """
    Devuelve las últimas acciones de audit_logs, con email/nombre si existe el usuario.
    Fechas en zona local (APP_TIMEZONE) y acción traducida a español.
    """
    with SessionLocal() as session:
        rows = (
            session.query(AuditLog, Usuario)
            .join(Usuario, Usuario.id == AuditLog.actor_user_id, isouter=True)
            .order_by(AuditLog.id.desc())
            .limit(limit)
            .all()
        )
        resultado = []
        for log, user in rows:
            resultado.append(
                {
                    "fecha": _fmt_fecha(log.at),
                    "usuario": user.email if user else (f"ID {log.actor_user_id}" if log.actor_user_id else "-"),
                    "nombre": user.nombre if user else None,
                    "accion": _accion_es(log.action),
                    "entidad": log.entity,
                    "entidad_id": log.entity_id,
                    "before": log.before_json,
                    "after": log.after_json,
                    "ip": log.ip,
                }
            )
        return resultado
