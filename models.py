# models.py
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Index,
    UniqueConstraint, Float
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

# ðŸ‘‰ Import robusto de Base: primero db_orm.Base, luego database.Base, y fallback declarative_base
try:
    from db_orm import Base  # comparte metadata con AuditLog
except Exception:
    try:
        from database import Base  # compat si seguÃ­s usando Base ahÃ­
    except Exception:
        from sqlalchemy.orm import declarative_base
        Base = declarative_base()


# ============================
# Usuarios / Agenda / Alerts
# ============================

class Usuario(Base):
    __tablename__ = "usuarios"
    __table_args__ = {"extend_existing": True}  # âœ… tolera doble registro

    id = Column(Integer, primary_key=True, index=True)

    # Nombre visible (no necesariamente Ãºnico; podÃ©s tener homÃ³nimos)
    nombre = Column(String(120), nullable=False)

    # Email para login: Ãºnico y con Ã­ndice
    email = Column(String(255), unique=True, index=True, nullable=False)

    # Hash de contraseÃ±a (no guardes texto plano)
    contrasena = Column(String(255), nullable=False)

    # 'usuario' | 'admin'
    rol = Column(String(20), default="usuario", nullable=False)

    # Habilitado/deshabilitado
    activo = Column(Boolean, default=True, nullable=False)

    # Trazas
    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    actualizado_en = Column(DateTime(timezone=True), onupdate=func.now())
    ultimo_login = Column(DateTime(timezone=True))

    # Relaciones
    eventos = relationship("Evento", back_populates="usuario", cascade="all, delete-orphan")
    notificaciones = relationship("Notificacion", back_populates="usuario", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Usuario id={self.id} email={self.email} rol={self.rol} activo={self.activo}>"


class Evento(Base):
    """
    Evento de calendario.
    - soporte all-day
    - color opcional (p.ej. #2f6adf)
    - visibilidad: 'privado' | 'equipo' (por si luego querÃ©s compartir)
    """
    __tablename__ = "eventos"
    __table_args__ = (
        Index("ix_eventos_usuario", "usuario_id"),
        Index("ix_eventos_inicio", "inicio"),
        {"extend_existing": True},  # âœ…
    )

    id = Column(Integer, primary_key=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False)

    titulo = Column(String(200), nullable=False)
    descripcion = Column(Text)

    inicio = Column(DateTime(timezone=True), nullable=False)
    fin = Column(DateTime(timezone=True), nullable=True)  # puede ser None si all_day
    all_day = Column(Boolean, default=False, nullable=False)

    color = Column(String(16))         # ej: "#2f6adf"
    visibilidad = Column(String(20), default="privado", nullable=False)

    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    actualizado_en = Column(DateTime(timezone=True), onupdate=func.now())

    usuario = relationship("Usuario", back_populates="eventos")

    def __repr__(self) -> str:
        return f"<Evento id={self.id} usuario_id={self.usuario_id} titulo={self.titulo!r}>"


class Notificacion(Base):
    """
    NotificaciÃ³n mostrada en el toast-center y/o campana.
    - tipo: 'info' | 'success' | 'warning' | 'error'
    - link opcional para deep-link (p.ej. a una incidencia o historial)
    - metadata JSON para payload libre (ids, etc.)
    """
    __tablename__ = "notificaciones"
    __table_args__ = (
        Index("ix_notif_usuario_leida", "usuario_id", "leida"),
        Index("ix_notif_creado_en", "creado_en"),
        {"extend_existing": True},  # âœ…
    )

    id = Column(Integer, primary_key=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False)

    titulo = Column(String(180), nullable=False)
    cuerpo = Column(Text)
    tipo = Column(String(20), default="info", nullable=False)

    link = Column(String(512))        # URL interna o relativa (ej: /incidencias/123)
    leida = Column(Boolean, default=False, nullable=False)
    leida_en = Column(DateTime(timezone=True))

    # âš ï¸ 'metadata' es nombre reservado en SQLAlchemy. Usamos atributo 'meta'
    #     pero la columna en BD sigue llamÃ¡ndose "metadata".
    meta = Column("metadata", JSON, default=dict)  # âœ… nombre de columna preservado

    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    usuario = relationship("Usuario", back_populates="notificaciones")

    def __repr__(self) -> str:
        return f"<Notificacion id={self.id} usuario_id={self.usuario_id} tipo={self.tipo} leida={self.leida}>"


# ============================
# Base de Conocimiento (KB)
# ============================

class KBSource(Base):
    """
    Fuente lÃ³gica de conocimiento. Agrupa archivos por rubro/organismo/Ã¡mbito
    y define dÃ³nde se almacenan fÃ­sicamente en el servidor (Render Disk).
    """
    __tablename__ = "kb_sources"
    __table_args__ = {"extend_existing": True}  # âœ…

    id = Column(Integer, primary_key=True)
    name = Column(String(128), unique=True, index=True)   # ej. "Salud-AR Base", "Normativa ANMAT"
    scope = Column(JSON, default=dict)                    # {"rubro":"salud","organismo":"ANMAT", ...}
    storage_path = Column(String(512), nullable=True)     # carpeta base donde se guardan los archivos
    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    files = relationship("KBFile", back_populates="source", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<KBSource id={self.id} name={self.name!r}>"


class KBFile(Base):
    """
    Archivo cargado a la BD de Conocimiento.
    """
    __tablename__ = "kb_files"
    __table_args__ = (
        Index("ix_kbfile_source_name", "source_id", "filename"),
        {"extend_existing": True},  # âœ…
    )

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("kb_sources.id", ondelete="CASCADE"), index=True, nullable=False)

    filename = Column(String(256), nullable=False)
    content_type = Column(String(64))                     # "text/plain", "application/pdf", etc.
    bytes = Column(Integer, default=0)
    hash_sha256 = Column(String(64), index=True)          # deduplicaciÃ³n
    stored_path = Column(String(512))                     # ruta absoluta donde quedÃ³ el archivo
    meta = Column(JSON, default=dict)                     # {"rubro": "...", "tags": [...], ...}
    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    source = relationship("KBSource", back_populates="files")
    chunks = relationship("KBChunk", back_populates="file", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<KBFile id={self.id} filename={self.filename!r} source_id={self.source_id}>"


class KBChunk(Base):
    """
    Fragmentos (chunks) textuales con embedding para RAG.
    """
    __tablename__ = "kb_chunks"
    __table_args__ = (
        Index("ix_kbchunk_file_ord", "file_id", "ord"),
        {"extend_existing": True},  # âœ…
    )

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("kb_files.id", ondelete="CASCADE"), index=True, nullable=False)

    ord = Column(Integer, nullable=False)                 # orden dentro del archivo
    text = Column(Text, nullable=False)                   # fragmento normalizado
    embedding = Column(Text)                              # json.dumps(list[float]) si no usas pgvector
    span = Column(JSON, default=dict)                     # {"page": 3, "from": 120, "to": 480}
    meta = Column(JSON, default=dict)
    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    file = relationship("KBFile", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<KBChunk id={self.id} file_id={self.file_id} ord={self.ord}>"


class KBPriority(Base):
    """
    Puntos/temas que 'no deben faltar' en el informe.
    No es restrictivo: guÃ­a de prioridades por rubro.
    """
    __tablename__ = "kb_priorities"
    __table_args__ = (
        UniqueConstraint("rubric", "label", name="uq_kbprio"),
        {"extend_existing": True},  # âœ…
    )

    id = Column(Integer, primary_key=True)
    rubric = Column(String(128), index=True)              # ej. "salud", "medicamentos", "descartables"
    label = Column(String(256), index=True)               # ej. "Certificaciones ANMAT/PM", "Vencimientos"
    details = Column(Text)                                # explicaciÃ³n breve/reglas suaves
    weight = Column(Float, default=1.0)
    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<KBPriority id={self.id} rubric={self.rubric!r} label={self.label!r} weight={self.weight}>"
