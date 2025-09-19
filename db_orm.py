import os
from datetime import datetime
from sqlalchemy import create_engine, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, Mapped, mapped_column

# Usamos SIEMPRE Base y Usuario del módulo models
from models import Base, Usuario  # <- IMPORTANTE: una sola Base y una sola Usuario

# --- DATABASE_URL multi-entorno (SQLite local / Postgres en Render)
raw_url = os.getenv("DATABASE_URL", "sqlite:///usuarios.db")
if raw_url.startswith("postgres://"):
    raw_url = raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
elif raw_url.startswith("postgresql://") and "+psycopg2" not in raw_url:
    raw_url = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
DB_URL = raw_url

connect_args = {}
if DB_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DB_URL, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# --- Auditoría: se define aquí pero usando la MISMA Base
class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    entity: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    before_json: Mapped[str | None] = mapped_column(String, nullable=True)
    after_json: Mapped[str | None] = mapped_column(String, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

def _ensure_sqlite_auditlog_columns():
    """
    Si ya existe audit_logs en SQLite pero faltan columnas nuevas,
    las agregamos con ALTER TABLE.
    """
    import sqlite3
    if not DB_URL.startswith("sqlite"):
        return
    db_path = DB_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("PRAGMA table_info(audit_logs)")
        cols = {row[1] for row in cur.fetchall()}
        alters = []
        if "before_json" not in cols:
            alters.append("ALTER TABLE audit_logs ADD COLUMN before_json TEXT")
        if "after_json" not in cols:
            alters.append("ALTER TABLE audit_logs ADD COLUMN after_json TEXT")
        if "ip" not in cols:
            alters.append("ALTER TABLE audit_logs ADD COLUMN ip TEXT")
        if "entity_id" not in cols:
            alters.append("ALTER TABLE audit_logs ADD COLUMN entity_id INTEGER")
        for sql in alters:
            conn.execute(sql)
        if alters:
            conn.commit()
    finally:
        conn.close()

def inicializar_bd_orm():
    # Crea TODAS las tablas registradas en Base (models + AuditLog)
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_auditlog_columns()
    print(f"✅ Tablas ORM verificadas/creadas en {DB_URL}")
