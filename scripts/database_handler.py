import sqlite3
import os
from datetime import datetime

DB_PATH = "database/predictions.db"

def create_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS paciente (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            edad INTEGER,
            genero TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS resultado (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER,
            imagen TEXT,
            clase_predicha TEXT,
            probabilidad REAL,
            FOREIGN KEY(paciente_id) REFERENCES paciente(id)
        )
    ''')

    conn.commit()
    conn.close()

def insert_paciente(nombre, edad, genero):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO paciente (nombre, edad, genero) VALUES (?, ?, ?)", (nombre, edad, genero))
    conn.commit()
    paciente_id = c.lastrowid
    conn.close()
    return paciente_id

def insert_resultado(paciente_id, imagen, clase_predicha, probabilidad):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO resultado (paciente_id, imagen, clase_predicha, probabilidad) VALUES (?, ?, ?, ?)",
              (paciente_id, imagen, clase_predicha, probabilidad))
    conn.commit()
    conn.close()

def get_pacientes():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, nombre FROM paciente")
    pacientes = c.fetchall()
    conn.close()
    return pacientes

def get_resultados(paciente_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT imagen, clase_predicha, probabilidad FROM resultado WHERE paciente_id=?", (paciente_id,))
    resultados = c.fetchall()
    conn.close()
    return resultados

def get_all_data_for_export():
    """Obtiene todos los datos combinados de pacientes y resultados para exportar"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    query = """
    SELECT 
        p.id, p.nombre, p.edad, p.genero, 
        r.imagen, r.clase_predicha, r.probabilidad,
        datetime(r.id, 'unixepoch') as fecha
    FROM paciente p
    LEFT JOIN resultado r ON p.id = r.paciente_id
    ORDER BY p.id
    """
    
    c.execute(query)
    data = c.fetchall()
    conn.close()
    
    # Formatear fecha si es necesario
    formatted_data = []
    for row in data:
        if row[7]:  # Si hay fecha
            fecha = datetime.strptime(row[7], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        else:
            fecha = ""
        formatted_data.append(row[:7] + (fecha,))
    
    return formatted_data

def delete_all_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Eliminar primero de 'resultado' por la clave foránea a 'paciente'
    c.execute("DELETE FROM resultado")
    c.execute("DELETE FROM paciente")
    
    conn.commit()
    conn.close()
