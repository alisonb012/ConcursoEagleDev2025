from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QHBoxLayout, QGroupBox, QTableWidget,
                           QTableWidgetItem, QHeaderView, QMessageBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from scripts.predict import COVIDClassifier

class AnalysisWindow(QWidget):
    def __init__(self, model_path, metadata_path=None):
        super().__init__()
        self.setWindowTitle("Módulo de Diagnóstico")
        self.setGeometry(200, 200, 900, 700)
        
        # Cargar clasificador
        self.classifier = COVIDClassifier(model_path, metadata_path)
        self.current_image = None
        
        # Inicializar UI
        self.init_ui()
    
    def init_ui(self):
        # Layout principal
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Panel de control
        control_group = QGroupBox("Controles")
        control_layout = QHBoxLayout()
        
        self.load_button = QPushButton("Cargar Imagen")
        self.load_button.clicked.connect(self.load_image)
        
        self.analyze_button = QPushButton("Analizar Imagen")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        
        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.analyze_button)
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Panel de visualización
        image_group = QGroupBox("Visualización")
        image_layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #ccc; background-color: #f9f9f9;")
        self.image_label.setMinimumSize(600, 400)
        
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        main_layout.addWidget(image_group)
        
        # Panel de resultados
        result_group = QGroupBox("Resultados")
        result_layout = QVBoxLayout()
        
        self.result_label = QLabel("Seleccione una imagen para analizar")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # Tabla de probabilidades
        self.prob_table = QTableWidget()
        self.prob_table.setColumnCount(2)
        self.prob_table.setHorizontalHeaderLabels(["Clase", "Probabilidad"])
        self.prob_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.prob_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.prob_table.setVisible(False)
        
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.prob_table)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)
    
    def load_image(self):
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "", 
            "Imágenes (*.png *.jpg *.jpeg);;Todos los archivos (*)", 
            options=options)
        
        if file_path:
            self.current_image = file_path
            self.display_image(file_path)
            self.analyze_button.setEnabled(True)
            self.result_label.setText("Imagen cargada. Haga clic en 'Analizar'")
            self.prob_table.setVisible(False)
    
    def display_image(self, path):
        # Cargar imagen con OpenCV
        cv_image = cv2.imread(path)
        if cv_image is not None:
            # Convertir a RGB para visualización
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Crear QImage
            height, width, channel = cv_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image.data, width, height, 
                           bytes_per_line, QImage.Format.Format_RGB888)
            
            # Escalar manteniendo aspecto
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
    
    def analyze_image(self):
        if not self.current_image:
            return
        
        try:
            # Realizar predicción
            prediction = self.classifier.predict(self.current_image)
            
            # Mostrar resultados principales
            self.result_label.setText(
                f"<b>Diagnóstico:</b> {prediction['class']}<br>"
                f"<b>Confianza:</b> {prediction['confidence']:.1%}")
            
            # Mostrar tabla de probabilidades
            self.show_probability_table(prediction['probabilities'])
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo analizar la imagen:\n{str(e)}")
            self.result_label.setText("<font color='red'>Error en el análisis</font>")
            self.prob_table.setVisible(False)
    
    def show_probability_table(self, probabilities):
        self.prob_table.setRowCount(len(probabilities))
        self.prob_table.setVisible(True)
        
        # Ordenar probabilidades de mayor a menor
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for row, (cls, prob) in enumerate(sorted_probs):
            self.prob_table.setItem(row, 0, QTableWidgetItem(cls))
            self.prob_table.setItem(row, 1, QTableWidgetItem(f"{prob:.2%}"))
        
        # Resaltar la fila con mayor probabilidad
        if self.prob_table.rowCount() > 0:
            for col in range(self.prob_table.columnCount()):
                self.prob_table.item(0, col).setBackground(Qt.GlobalColor.yellow)