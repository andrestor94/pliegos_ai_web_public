# crear_db.py
# Crea (o actualiza si no existen) todas las tablas definidas en models.py
# y (opcionalmente) siembra usuarios iniciales.
#
# Uso recomendado:
#   python crear_db.py --reset --seed-users
#
# Flags:
#   --reset       : si la DB es SQLite, borra el archivo antes de crear (CUIDADO en prod)
#   --seed-users  : crea los usuarios Admin/AndrÃ©s si no existen

import os
import argparse
from sqlalchemy.exc import SQLAlchemyError

# Engine / Base provienen de tu mÃ³dulo database (ya usado por models.py)
from database import engine, Base  # Base es la misma que importa models.py
import models  # noqa: F401  (importar registra TODOS los modelos en Base.metadata)

# Para mantener tu lÃ³gica de creaciÃ³n de usuarios
from database import agregar_usuario


def is_sqlite_engine() -> bool:
    """Detecta si el engine apunta a SQLite."""
    try:
        url = str(engine.url)
        return url.startswith("sqlite")
    except Exception:
        return False


def sqlite_path_from_engine() -> str | None:
    """Devuelve la ruta del archivo sqlite si aplica."""
    try:
        if is_sqlite_engine():
            # engine.url.database trae la ruta a archivo sqlite
            return engine.url.database
    except Exception:
        pass
    return None


def create_schema(reset_sqlite: bool = False) -> None:
    """Crea todas las tablas definidas en models.py."""
    if reset_sqlite and is_sqlite_engine():
        db_file = sqlite_path_from_engine()
        if db_file and os.path.exists(db_file):
            os.remove(db_file)
            print(f"ðŸ—‘ï¸  Base SQLite anterior eliminada: {db_file}")

    print("==> Creando/actualizando tablasâ€¦")
    try:
        Base.metadata.create_all(bind=engine)
        print("âœ” Tablas creadas/actualizadas correctamente.")
    except SQLAlchemyError as e:
        print("âœ– Error creando tablas:", e)
        raise


def seed_users() -> None:
    """Crea usuarios iniciales si no existen (usa tu agregar_usuario)."""
    usuarios_iniciales = [
        ("Admin",  "admin@suizo.com",  "admin123",   "admin"),
        ("AndrÃ©s", "andres@suizo.com", "usuario123", "usuario"),
    ]
    for nombre, email, password, rol in usuarios_iniciales:
        try:
            agregar_usuario(nombre=nombre, email=email, rol=rol, password=password)
            print(f"âœ… Usuario creado: {email} ({rol}) - contraseÃ±a: {password}")
        except Exception as e:
            # Si ya existe o hay validaciÃ³n interna, lo informamos y seguimos
            print(f"â„¹ï¸  No se creÃ³ {email} (posible duplicado): {e}")


def main():
    ap = argparse.ArgumentParser(description="Crear esquema de la DB y sembrar usuarios (opcional).")
    ap.add_argument("--reset", action="store_true",
                    help="Si la DB es SQLite, borrar el archivo antes de crear (Â¡no usar en producciÃ³n!).")
    ap.add_argument("--seed-users", action="store_true",
                    help="Crear usuarios iniciales (Admin/AndrÃ©s) si no existen.")
    args = ap.parse_args()

    # Mostrar a quÃ© URL estamos conectados (Ãºtil en Render vs local)
    try:
        print(f"ðŸ”— DATABASE URL: {engine.url}")
    except Exception:
        pass

    create_schema(reset_sqlite=args.reset)

    if args.seed_users:
        seed_users()

    print("==> Listo.")


if __name__ == "__main__":
    main()
