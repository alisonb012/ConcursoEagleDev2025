from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog,
    QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem,
    QInputDialog
)
from PyQt6.QtCore import Qt
from database_handler import create_db, insert_paciente, get_pacientes, insert_resultado, get_resultados, get_all_data_for_export
from predict import load_model, predict_single_image
import os
import zipfile
import random
import csv
from datetime import datetime


class MainWindow(QWidget):
    def __init__(self, model_path, zip_path, class_names):
        super().__init__()
        self.setWindowTitle("Clasificador COVID - Hospital")
        self.resize(900, 600)
        self.model_path = model_path
        self.zip_path = zip_path
        self.class_names = class_names
        self.model = load_model(model_path)

        create_db()
        self.paciente_id = None

        self.setup_ui()
        self.load_pacientes()
        # Cargar todos los resultados por defecto
        self.load_all_resultados()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

        # Left: Patient list + Add patient
        left_layout = QVBoxLayout()
        self.paciente_list = QListWidget()
        self.paciente_list.itemClicked.connect(self.on_paciente_selected)
        left_layout.addWidget(QLabel("Pacientes:"))
        left_layout.addWidget(self.paciente_list)
        self.btn_add_paciente = QPushButton("Añadir Paciente")
        self.btn_add_paciente.clicked.connect(self.add_paciente)
        left_layout.addWidget(self.btn_add_paciente)

        self.btn_export = QPushButton("Exportar Datos para Power BI")
        self.btn_export.clicked.connect(self.export_to_powerbi)
        left_layout.addWidget(self.btn_export)

        self.btn_predict_all = QPushButton("Predecir Todos los Pacientes")
        self.btn_predict_all.clicked.connect(self.predict_all_patients)
        left_layout.addWidget(self.btn_predict_all)

        # Right: Results table
        right_layout = QVBoxLayout()
        
        # Botones para cambiar vista
        view_buttons_layout = QHBoxLayout()
        self.btn_view_selected = QPushButton("Ver Paciente Seleccionado")
        self.btn_view_all = QPushButton("Ver Todos los Resultados")
        self.btn_view_selected.clicked.connect(self.load_resultados_selected)
        self.btn_view_all.clicked.connect(self.load_all_resultados)
        view_buttons_layout.addWidget(self.btn_view_selected)
        view_buttons_layout.addWidget(self.btn_view_all)
        
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["Paciente", "Imagen", "Clase Predicha", "Probabilidad"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(QLabel("Resultados:"))
        right_layout.addLayout(view_buttons_layout)
        right_layout.addWidget(self.result_table)

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 5)

    def load_pacientes(self):
        self.paciente_list.clear()
        pacientes = get_pacientes()
        for pid, nombre in pacientes:
            self.paciente_list.addItem(f"{pid} - {nombre}")

    def add_paciente(self):
        nombre, ok = QInputDialog.getText(self, "Añadir Paciente", "Nombre:")
        if not ok or not nombre.strip():
            return
        edad, ok = QInputDialog.getInt(self, "Añadir Paciente", "Edad:", min=0, max=120)
        if not ok:
            return
        genero, ok = QInputDialog.getItem(self, "Añadir Paciente", "Género:", ["Masculino", "Femenino", "Otro"], editable=False)
        if not ok:
            return

        # Get random image from ZIP
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not image_files:
                    QMessageBox.warning(self, "Error", "No se encontraron imágenes en el ZIP.")
                    return
                random_image = random.choice(image_files)

                # Predict and save results
                paciente_id = insert_paciente(nombre.strip(), edad, genero)
                clase, prob_array = predict_single_image(self.model, self.zip_path, random_image, self.class_names)
                prob = max(prob_array)
                insert_resultado(paciente_id, random_image, clase, prob)

                # Update UI - CORREGIDO: no pasar paciente_id como parámetro
                self.paciente_id = paciente_id
                self.load_pacientes()
                self.load_resultados()  # Sin parámetro
                QMessageBox.information(self, "Éxito", f"Paciente {nombre} agregado.\nImagen asignada: {random_image}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo procesar el ZIP: {str(e)}")

    def on_paciente_selected(self, item):
        text = item.text()
        self.paciente_id = int(text.split(" - ")[0])
        self.load_resultados()

    def load_resultados(self):
        """Carga resultados del paciente seleccionado"""
        self.load_resultados_selected()

    def load_resultados_selected(self):
        """Carga resultados del paciente seleccionado"""
        if not self.paciente_id:
            return
        resultados = get_resultados(self.paciente_id)
        paciente_nombre = next((nombre for pid, nombre in get_pacientes() if pid == self.paciente_id), "Desconocido")
        
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["Paciente", "Imagen", "Clase Predicha", "Probabilidad"])
        
        for imagen, clase, prob in resultados:
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(paciente_nombre))
            self.result_table.setItem(row, 1, QTableWidgetItem(imagen))
            self.result_table.setItem(row, 2, QTableWidgetItem(clase))
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{prob:.4f}"))

    def load_all_resultados(self):
        """Carga todos los resultados de todos los pacientes"""
        data = get_all_data_for_export()
        
        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["Paciente", "Imagen", "Clase Predicha", "Probabilidad"])
        
        for row_data in data:
            if row_data[4]:  # Si tiene imagen (tiene predicción)
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                self.result_table.setItem(row, 0, QTableWidgetItem(row_data[1]))  # Nombre
                self.result_table.setItem(row, 1, QTableWidgetItem(row_data[4]))  # Imagen
                self.result_table.setItem(row, 2, QTableWidgetItem(row_data[5]))  # Clase Predicha
                self.result_table.setItem(row, 3, QTableWidgetItem(f"{row_data[6]:.4f}"))  # Probabilidad

    def predict_all_patients(self):
        """Realiza predicciones para todos los pacientes sin resultados"""
        pacientes = get_pacientes()
        if not pacientes:
            QMessageBox.warning(self, "Error", "No hay pacientes registrados.")
            return

        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_files:
                QMessageBox.warning(self, "Error", "No hay imágenes en el ZIP.")
                return

            success = 0
            for paciente_id, _ in pacientes:
                # Verificar si ya tiene predicciones
                if not get_resultados(paciente_id):
                    random_image = random.choice(image_files)
                    clase, prob_array = predict_single_image(self.model, self.zip_path, random_image, self.class_names)
                    prob = max(prob_array)
                    insert_resultado(paciente_id, random_image, clase, prob)
                    success += 1

            QMessageBox.information(self, "Completado", 
                                  f"Predicciones generadas para {success} pacientes.")

    def export_to_powerbi(self):
        """Exporta todos los datos a un CSV para Power BI"""
        # CORREGIDO: QFileDialog.Options() no existe en PyQt6
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar datos para Power BI", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx)")
        
        if not file_path:
            return

        data = get_all_data_for_export()
        if not data:
            QMessageBox.warning(self, "Error", "No hay datos para exportar.")
            return

        try:
            if file_path.endswith('.csv'):
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID_Paciente', 'Nombre', 'Edad', 'Género', 
                                   'Imagen', 'Clase_Predicha', 'Probabilidad', 'Fecha'])
                    writer.writerows(data)
            else:
                import pandas as pd
                df = pd.DataFrame(data, columns=['ID_Paciente', 'Nombre', 'Edad', 'Género',
                                               'Imagen', 'Clase_Predicha', 'Probabilidad', 'Fecha'])
                df.to_excel(file_path, index=False)

            QMessageBox.information(self, "Éxito", 
                                  f"Datos exportados correctamente a:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo exportar: {str(e)}")