import os
from database import crear_tabla_usuarios, agregar_usuario

DB_PATH = "usuarios.db"

def crear_usuarios_base():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("🗑️ Base de datos anterior eliminada.")

    crear_tabla_usuarios()

    usuarios_iniciales = [
        ("Admin", "admin@suizo.com", "admin123", "admin"),
        ("Andrés", "andres@suizo.com", "usuario123", "usuario")
    ]

    for nombre, email, password, rol in usuarios_iniciales:
        try:
            agregar_usuario(nombre=nombre, email=email, rol=rol, password=password)
            print(f"✅ Usuario creado: {email} ({rol}) - contraseña: {password}")
        except Exception as e:
            print(f"⚠️ Error creando {email}: {e}")

if __name__ == "__main__":
    crear_usuarios_base()
