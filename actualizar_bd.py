import sqlite3

DB_PATH = "usuarios.db"

def agregar_columna_resumen_historial():
    with sqlite3.connect(DB_PATH) as conn:
        try:
            conn.execute("ALTER TABLE historial ADD COLUMN resumen_texto TEXT")
            print("✅ Columna 'resumen_texto' agregada a la tabla historial.")
        except sqlite3.OperationalError as e:
            print(f"⚠️ Ya existe la columna o hubo un error: {e}")

if __name__ == "__main__":
    agregar_columna_resumen_historial()
