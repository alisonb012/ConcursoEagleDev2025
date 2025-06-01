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
        self.load_all_resultados()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)

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

        right_layout = QVBoxLayout()

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

        image_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen de Paciente", "", 
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if not image_path:
            return

        try:
            paciente_id = insert_paciente(nombre.strip(), edad, genero)
            clase, prob_array = predict_single_image(self.model, image_path=image_path, image_in_zip=None, class_names=self.class_names)
            prob = max(prob_array)
            insert_resultado(paciente_id, os.path.basename(image_path), clase, prob)

            self.paciente_id = paciente_id
            self.load_pacientes()
            self.load_resultados()
            QMessageBox.information(self, "Éxito", f"Paciente {nombre} agregado.\nImagen: {os.path.basename(image_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo procesar la imagen: {str(e)}")

    def on_paciente_selected(self, item):
        text = item.text()
        self.paciente_id = int(text.split(" - ")[0])
        self.load_resultados()

    def load_resultados(self):
        self.load_resultados_selected()

    def load_resultados_selected(self):
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
        data = get_all_data_for_export()

        self.result_table.setRowCount(0)
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["Paciente", "Imagen", "Clase Predicha", "Probabilidad"])

        for row_data in data:
            if row_data[4]:
                row = self.result_table.rowCount()
                self.result_table.insertRow(row)
                self.result_table.setItem(row, 0, QTableWidgetItem(row_data[1]))
                self.result_table.setItem(row, 1, QTableWidgetItem(row_data[4]))
                self.result_table.setItem(row, 2, QTableWidgetItem(row_data[5]))
                self.result_table.setItem(row, 3, QTableWidgetItem(f"{row_data[6]:.4f}"))

    def predict_all_patients(self):
        pacientes = get_pacientes()
        if not pacientes:
            QMessageBox.warning(self, "Error", "No hay pacientes registrados.")
            return

        QMessageBox.information(self, "Información", "Este modo ha sido deshabilitado ya que las imágenes ya no se obtienen de un ZIP.")

    def export_to_powerbi(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Guardar datos para Power BI", "", 
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )

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

            QMessageBox.information(self, "Éxito", f"Datos exportados correctamente a:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo exportar: {str(e)}")
