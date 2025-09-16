# actualiza_bd.py
# Ejecuta migraciones de BD:
# 1) Agrega columna resumen_texto a historial (si no existe)
# 2) Crea tablas analyses, analysis_sections y section_feedback (si no existen)

import os
import sqlite3

# Ruta de la DB: por defecto "usuarios.db" en la misma carpeta de este archivo
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "usuarios.db"))

def agregar_columna_resumen_historial(conn: sqlite3.Connection):
    """
    Agrega la columna 'resumen_texto' a la tabla 'historial' si no existe.
    """
    # Verificar si la columna ya existe para evitar excepciones
    cur = conn.execute("PRAGMA table_info(historial)")
    cols = [row[1] for row in cur.fetchall()]  # row[1] = name
    if "resumen_texto" in cols:
        print("ℹ️  La columna 'resumen_texto' ya existe en 'historial'.")
        return

    try:
        conn.execute("ALTER TABLE historial ADD COLUMN resumen_texto TEXT")
        print("✅ Columna 'resumen_texto' agregada a la tabla 'historial'.")
    except sqlite3.OperationalError as e:
        # Por si la tabla no existe o cualquier otro caso
        print(f"⚠️ No se pudo agregar 'resumen_texto' a 'historial': {e}")

def crear_tablas_feedback(conn: sqlite3.Connection):
    """
    Crea tablas para guardar análisis y feedback por sección.
    - analyses: cabecera del análisis
    - analysis_sections: snapshot por sección del análisis
    - section_feedback: valoraciones ✅/❌ + comentario por sección
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS analyses (
      id TEXT PRIMARY KEY,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      source_file TEXT,
      portal TEXT,
      buyer TEXT
    );

    CREATE TABLE IF NOT EXISTS analysis_sections (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      analysis_id TEXT,
      section_key TEXT,
      section_title TEXT,
      payload_json TEXT,
      FOREIGN KEY(analysis_id) REFERENCES analyses(id)
    );

    CREATE TABLE IF NOT EXISTS section_feedback (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      analysis_id TEXT,
      section_key TEXT,
      is_correct INTEGER,      -- 1=✅, 0=❌
      comment TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(analysis_id) REFERENCES analyses(id)
    );

    CREATE INDEX IF NOT EXISTS ix_feedback_lookup ON section_feedback(analysis_id, section_key);
    """
    conn.executescript(ddl)
    print("✅ Tablas de análisis y feedback listas (analyses, analysis_sections, section_feedback).")

def main():
    print(f"🗄  Usando DB en: {DB_PATH}")
    # Habilitar claves foráneas por si en algún momento las usás más estricto
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        agregar_columna_resumen_historial(conn)
        crear_tablas_feedback(conn)
        conn.commit()
    print("🎉 Migraciones completadas.")

if __name__ == "__main__":
    main()
