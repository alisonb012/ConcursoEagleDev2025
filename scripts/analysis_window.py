from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                           QFileDialog, QHBoxLayout, QGroupBox, QTableWidget,
                           QTableWidgetItem, QHeaderView, QMessageBox, QFrame,
                           QGridLayout, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor
import cv2
import numpy as np
from predict import COVIDClassifier

class ModernCard(QFrame):
    """Componente de tarjeta moderna con sombra y bordes redondeados"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 16px;
                margin: 8px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
            title_label.setStyleSheet("""
                QLabel {
                    color: #1F2937;
                    border: none;
                    padding: 0;
                    margin: 0 0 12px 0;
                }
            """)
            layout.addWidget(title_label)
        
        self.setLayout(layout)
    
    def add_widget(self, widget):
        self.layout().addWidget(widget)

class AnalysisWindow(QWidget):
    def __init__(self, model_path, metadata_path=None):
        super().__init__()
        self.setWindowTitle("Módulo de Diagnóstico - COVID-19")
        self.setGeometry(200, 200, 1200, 800)
        
        # Cargar clasificador
        self.classifier = COVIDClassifier(model_path, metadata_path)
        self.current_image = None
        
        # Configurar tema moderno
        self.setup_theme()
        
        # Inicializar UI
        self.init_ui()
    
    def setup_theme(self):
        """Configura el tema moderno de la aplicación"""
        self.setStyleSheet("""
            QWidget {
                background-color: #F9FAFB;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #63C132, stop: 1 #9EE37D);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                min-height: 16px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #56A429, stop: 1 #8AD967);
                transform: translateY(-1px);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4F9425, stop: 1 #7BC95A);
            }
            
            QPushButton:disabled {
                background: #D1D5DB;
                color: #9CA3AF;
            }
            
            QTableWidget {
                background-color: white;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                gridline-color: #F3F4F6;
                font-size: 13px;
            }
            
            QTableWidget::item {
                padding: 12px 16px;
                border-bottom: 1px solid #F3F4F6;
            }
            
            QTableWidget::item:selected {
                background-color: #EEF2FF;
                color: #1E40AF;
            }
            
            QHeaderView::section {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #AAEFDF, stop: 1 #9EE37D);
                color: #1F2937;
                padding: 12px 16px;
                border: none;
                font-weight: 600;
                font-size: 13px;
            }
            
            QLabel {
                color: #374151;
            }
        """)
    
    def init_ui(self):
        # Layout principal con márgenes modernos
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(20)
        self.setLayout(main_layout)
        
        # Header con título elegante
        header_layout = QHBoxLayout()
        title_label = QLabel("🏥 Diagnóstico Inteligente COVID-19")
        title_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        title_label.setStyleSheet("""
            QLabel {
                color: #1F2937;
                padding: 16px 0;
                border-bottom: 3px solid #63C132;
                margin-bottom: 20px;
            }
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Layout de contenido principal
        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)
        
        # Panel izquierdo - Controles e imagen
        left_panel = QVBoxLayout()
        left_panel.setSpacing(20)
        
        # Tarjeta de controles
        controls_card = ModernCard("🎛️ Controles")
        
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(16)
        
        self.load_button = QPushButton("📁 Cargar Imagen")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setStyleSheet(self.load_button.styleSheet() + """
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #AAEFDF, stop: 1 #CFFCFF);
                color: #1F2937;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #9DE5D3, stop: 1 #B8F5F7);
            }
        """)
        
        self.analyze_button = QPushButton("🔬 Analizar Imagen")
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setEnabled(False)
        
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.analyze_button)
        controls_card.add_widget(QWidget())
        controls_card.layout().itemAt(1).widget().setLayout(controls_layout)
        
        left_panel.addWidget(controls_card)
        
        # Tarjeta de visualización de imagen
        image_card = ModernCard("🖼️ Visualización")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #D1D5DB;
                border-radius: 12px;
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #F9FAFB, stop: 1 #F3F4F6);
                color: #9CA3AF;
                font-size: 16px;
                padding: 40px;
                min-height: 400px;
            }
        """)
        self.image_label.setText("🖼️\n\nArrastre una imagen aquí\no use el botón 'Cargar Imagen'")
        
        image_card.add_widget(self.image_label)
        left_panel.addWidget(image_card)
        
        content_layout.addLayout(left_panel, 2)
        
        # Panel derecho - Resultados
        right_panel = QVBoxLayout()
        right_panel.setSpacing(20)
        
        # Tarjeta de resultado principal
        result_card = ModernCard("📊 Resultado del Diagnóstico")
        
        self.result_label = QLabel("Seleccione una imagen para comenzar el análisis")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: 600;
                color: #6B7280;
                padding: 30px;
                border: 2px dashed #E5E7EB;
                border-radius: 12px;
                background-color: #F9FAFB;
            }
        """)
        
        result_card.add_widget(self.result_label)
        right_panel.addWidget(result_card)
        
        # Tarjeta de probabilidades detalladas
        prob_card = ModernCard("📈 Probabilidades Detalladas")
        
        self.prob_table = QTableWidget()
        self.prob_table.setColumnCount(2)
        self.prob_table.setHorizontalHeaderLabels(["🏷️ Clase", "📊 Probabilidad"])
        self.prob_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.prob_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.prob_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.prob_table.setAlternatingRowColors(True)
        self.prob_table.setVisible(False)
        
        prob_card.add_widget(self.prob_table)
        right_panel.addWidget(prob_card)
        
        content_layout.addLayout(right_panel, 1)
        main_layout.addLayout(content_layout)
    
    def load_image(self):
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "", 
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.tiff);;Todos los archivos (*)", 
            options=options)
        
        if file_path:
            self.current_image = file_path
            self.display_image(file_path)
            self.analyze_button.setEnabled(True)
            self.result_label.setText("✅ Imagen cargada correctamente\n\nHaga clic en 'Analizar Imagen' para continuar")
            self.result_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: 600;
                    color: #059669;
                    padding: 30px;
                    border: 2px solid #10B981;
                    border-radius: 12px;
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #ECFDF5, stop: 1 #F0FDF4);
                }
            """)
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
            max_size = 500
            scaled_pixmap = pixmap.scaled(
                max_size, max_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #63C132;
                    border-radius: 12px;
                    background-color: white;
                    padding: 8px;
                }
            """)
    
    def analyze_image(self):
        if not self.current_image:
            return
        
        # Mostrar estado de carga
        self.result_label.setText("🔄 Analizando imagen...\n\nPor favor espere")
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #2563EB;
                padding: 30px;
                border: 2px solid #3B82F6;
                border-radius: 12px;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #EFF6FF, stop: 1 #F0F9FF);
            }
        """)
        
        try:
            # Realizar predicción
            prediction = self.classifier.predict(self.current_image)
            
            # Determinar color y emoji según el resultado
            class_name = prediction['class']
            confidence = prediction['confidence']
            
            if class_name.lower() == 'normal':
                color = "#059669"
                bg_gradient = "stop: 0 #ECFDF5, stop: 1 #F0FDF4"
                border_color = "#10B981"
                emoji = "✅"
            elif 'covid' in class_name.lower():
                color = "#DC2626"
                bg_gradient = "stop: 0 #FEF2F2, stop: 1 #FFF5F5"
                border_color = "#EF4444"
                emoji = "⚠️"
            else:
                color = "#D97706"
                bg_gradient = "stop: 0 #FFFBEB, stop: 1 #FEF3C7"
                border_color = "#F59E0B"
                emoji = "🔍"
            
            # Mostrar resultados principales
            self.result_label.setText(
                f"{emoji} <b>Diagnóstico:</b> {class_name}<br><br>"
                f"📊 <b>Confianza:</b> {confidence:.1%}")
            
            self.result_label.setStyleSheet(f"""
                QLabel {{
                    font-size: 18px;
                    font-weight: 600;
                    color: {color};
                    padding: 30px;
                    border: 2px solid {border_color};
                    border-radius: 12px;
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        {bg_gradient});
                }}
            """)
            
            # Mostrar tabla de probabilidades
            self.show_probability_table(prediction['probabilities'])
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo analizar la imagen:\n{str(e)}")
            self.result_label.setText("❌ Error en el análisis\n\nIntente con otra imagen")
            self.result_label.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: 600;
                    color: #DC2626;
                    padding: 30px;
                    border: 2px solid #EF4444;
                    border-radius: 12px;
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #FEF2F2, stop: 1 #FFF5F5);
                }
            """)
            self.prob_table.setVisible(False)
    
    def show_probability_table(self, probabilities):
        self.prob_table.setRowCount(len(probabilities))
        self.prob_table.setVisible(True)
        
        # Ordenar probabilidades de mayor a menor
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        # Mapeo de emojis para cada clase
        class_emojis = {
            'normal': '✅',
            'covid': '🦠',
            'covid-19': '🦠',
            'pneumonia': '🫁',
            'opacity': '🔍',
            'viral': '🦠'
        }
        
        for row, (cls, prob) in enumerate(sorted_probs):
            # Obtener emoji apropiado
            emoji = '🏷️'
            for key, value in class_emojis.items():
                if key in cls.lower():
                    emoji = value
                    break
            
            class_item = QTableWidgetItem(f"{emoji} {cls}")
            prob_item = QTableWidgetItem(f"{prob:.2%}")
            
            self.prob_table.setItem(row, 0, class_item)
            self.prob_table.setItem(row, 1, prob_item)
            
            # Colorear la fila con mayor probabilidad
            if row == 0:
                class_item.setBackground(QColor("#FEF3C7"))
                prob_item.setBackground(QColor("#FEF3C7"))
                class_item.setForeground(QColor("#92400E"))
                prob_item.setForeground(QColor("#92400E"))
        
        # Ajustar altura de filas
        self.prob_table.resizeRowsToContents()