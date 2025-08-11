import os
from datetime import datetime
from sqlalchemy import create_engine, String, Integer, DateTime
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column

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

# --- Base ORM
class Base(DeclarativeBase):
    pass

# --- Modelo que YA existe en tu SQLite (lo usamos en JOINs de auditoría)
class Usuario(Base):
    __tablename__ = "usuarios"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nombre: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    # No necesitamos mapear password/rol/activo para JOINs simples, pero los dejamos si querés:
    # password: Mapped[str] = mapped_column(String(255), nullable=False)
    # rol: Mapped[str] = mapped_column(String(50), nullable=False, default="usuario")
    # activo: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

# --- Tabla de auditoría (SQLAlchemy la crea en Postgres/SQLite si no existe)
class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    entity: Mapped[str] = mapped_column(String(50), nullable=False)
    # Campos opcionales que usamos desde database.py (guardados como JSON en texto)
    before_json: Mapped[str | None] = mapped_column(String, nullable=True)
    after_json: Mapped[str | None] = mapped_column(String, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

def inicializar_bd_orm():
    Base.metadata.create_all(bind=engine)
    print(f"✅ Tablas ORM verificadas/creadas en {DB_URL}")
