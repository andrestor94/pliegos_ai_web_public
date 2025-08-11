# models.py
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Index
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base


class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)

    # Nombre visible (no necesariamente único; podés tener homónimos)
    nombre = Column(String(120), nullable=False)

    # Email para login: único y con índice
    email = Column(String(255), unique=True, index=True, nullable=False)

    # Hash de contraseña (no guardes texto plano)
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
    - visibilidad: 'privado' | 'equipo' (por si luego querés compartir)
    """
    __tablename__ = "eventos"

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

    __table_args__ = (
        Index("ix_eventos_usuario", "usuario_id"),
        Index("ix_eventos_inicio", "inicio"),
    )

    def __repr__(self) -> str:
        return f"<Evento id={self.id} usuario_id={self.usuario_id} titulo={self.titulo!r}>"


class Notificacion(Base):
    """
    Notificación mostrada en el toast-center y/o campana.
    - tipo: 'info' | 'success' | 'warning' | 'error'
    - link opcional para deep-link (p.ej. a una incidencia o historial)
    - metadata JSON para payload libre (ids, etc.)
    """
    __tablename__ = "notificaciones"

    id = Column(Integer, primary_key=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False)

    titulo = Column(String(180), nullable=False)
    cuerpo = Column(Text)
    tipo = Column(String(20), default="info", nullable=False)

    link = Column(String(512))        # URL interna o relativa (ej: /incidencias/123)
    leida = Column(Boolean, default=False, nullable=False)
    leida_en = Column(DateTime(timezone=True))

    metadata = Column(JSON)           # datos extra opcionales

    creado_en = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    usuario = relationship("Usuario", back_populates="notificaciones")

    __table_args__ = (
        Index("ix_notif_usuario_leida", "usuario_id", "leida"),
        Index("ix_notif_creado_en", "creado_en"),
    )

    def __repr__(self) -> str:
        return f"<Notificacion id={self.id} usuario_id={self.usuario_id} tipo={self.tipo} leida={self.leida}>"
