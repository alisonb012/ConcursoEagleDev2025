from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QLabel, 
                           QPushButton, QFileDialog, QMessageBox, QProgressBar,
                           QGroupBox, QHBoxLayout, QStatusBar, QMenuBar, QMenu)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon
import os
import joblib
from scripts.preprocessing import preprocess_dataset
from scripts.train_model import train_and_save_model
from scripts.predict import COVIDClassifier

class ProcessingThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, zip_path, max_images=None):
        super().__init__()
        self.zip_path = zip_path
        self.max_images = max_images
    
    def run(self):
        try:
            metadata_path = preprocess_dataset(self.zip_path, max_per_class=self.max_images)
            self.finished.emit(True, f"Procesamiento completado\nDatos guardados en: {metadata_path}")
        except Exception as e:
            self.finished.emit(False, f"Error en procesamiento: {str(e)}")

class TrainingThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)
    
    def __init__(self, metadata_path):
        super().__init__()
        self.metadata_path = metadata_path
    
    def run(self):
        try:
            model_path = train_and_save_model(self.metadata_path)
            self.finished.emit(True, f"Entrenamiento completado\nModelo guardado en: {model_path}")
        except Exception as e:
            self.finished.emit(False, f"Error en entrenamiento: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Diagnóstico COVID-19")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("icon.png"))  # Asegúrate de tener este archivo
        
        # Variables de estado
        self.zip_path = None
        self.metadata_path = "data/processed/metadata.joblib"
        self.model_path = "models/covid_classifier.joblib"
        
        # Inicializar UI
        self.init_ui()
        self.create_actions()
        self.create_menus()
        self.create_status_bar()
        
        # Verificar si hay datos y modelo existentes
        self.check_existing_files()
    
    def init_ui(self):
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Panel de bienvenida
        welcome_label = QLabel("Sistema de Diagnóstico de COVID-19 mediante Radiografías Pulmonares")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(welcome_label)
        
        # Panel de dataset
        dataset_group = QGroupBox("1. Cargar Dataset")
        dataset_layout = QVBoxLayout()
        
        self.dataset_label = QLabel("No se ha cargado ningún dataset")
        self.dataset_label.setWordWrap(True)
        
        self.load_button = QPushButton("Seleccionar Archivo ZIP")
        self.load_button.clicked.connect(self.load_dataset)
        
        dataset_layout.addWidget(self.dataset_label)
        dataset_layout.addWidget(self.load_button)
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Panel de procesamiento
        process_group = QGroupBox("2. Procesar Imágenes")
        process_layout = QVBoxLayout()
        
        self.process_button = QPushButton("Iniciar Procesamiento")
        self.process_button.clicked.connect(self.process_dataset)
        self.process_button.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        process_layout.addWidget(self.process_button)
        process_layout.addWidget(self.progress_bar)
        process_group.setLayout(process_layout)
        layout.addWidget(process_group)
        
        # Panel de entrenamiento
        train_group = QGroupBox("3. Entrenar Modelo")
        train_layout = QVBoxLayout()
        
        self.train_button = QPushButton("Iniciar Entrenamiento")
        self.train_button.clicked.connect(self.train_model)
        self.train_button.setEnabled(False)
        
        train_layout.addWidget(self.train_button)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Panel de diagnóstico
        diag_group = QGroupBox("4. Realizar Diagnóstico")
        diag_layout = QVBoxLayout()
        
        self.diagnose_button = QPushButton("Abrir Módulo de Diagnóstico")
        self.diagnose_button.clicked.connect(self.open_diagnosis_window)
        self.diagnose_button.setEnabled(False)
        
        diag_layout.addWidget(self.diagnose_button)
        diag_group.setLayout(diag_layout)
        layout.addWidget(diag_group)
        
        layout.addStretch()
    
    def create_actions(self):
        # Acción para cargar dataset
        self.load_action = QAction("Cargar Dataset", self)
        self.load_action.triggered.connect(self.load_dataset)
        
        # Acción para salir
        self.exit_action = QAction("Salir", self)
        self.exit_action.triggered.connect(self.close)
        
        # Acción acerca de
        self.about_action = QAction("Acerca de", self)
        self.about_action.triggered.connect(self.show_about)
    
    def create_menus(self):
        # Barra de menú
        menubar = self.menuBar()
        
        # Menú Archivo
        file_menu = menubar.addMenu("Archivo")
        file_menu.addAction(self.load_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        # Menú Ayuda
        help_menu = menubar.addMenu("Ayuda")
        help_menu.addAction(self.about_action)
    
    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo")
    
    def check_existing_files(self):
        # Verificar si ya existe metadata procesada
        if os.path.exists(self.metadata_path):
            self.process_button.setEnabled(False)
            self.train_button.setEnabled(True)
            self.dataset_label.setText("Dataset ya procesado anteriormente")
        
        # Verificar si ya existe modelo entrenado
        if os.path.exists(self.model_path):
            self.train_button.setEnabled(False)
            self.diagnose_button.setEnabled(True)
    
    def load_dataset(self):
        options = QFileDialog.Option.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo ZIP", "", 
            "Archivos ZIP (*.zip)", options=options)
        
        if file_path:
            self.zip_path = file_path
            self.dataset_label.setText(f"Dataset seleccionado:\n{file_path}")
            self.process_button.setEnabled(True)
            self.status_bar.showMessage("Dataset cargado correctamente")
    
    def process_dataset(self):
        if not self.zip_path:
            QMessageBox.warning(self, "Error", "Primero seleccione un archivo ZIP")
            return
        
        # Configurar interfaz durante el procesamiento
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Barra indeterminada
        self.set_buttons_enabled(False)
        self.status_bar.showMessage("Procesando dataset...")
        
        # Ejecutar procesamiento en un hilo separado
        self.processing_thread = ProcessingThread(self.zip_path)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def on_processing_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if success:
            QMessageBox.information(self, "Éxito", message)
            self.train_button.setEnabled(True)
            self.status_bar.showMessage("Procesamiento completado")
        else:
            QMessageBox.critical(self, "Error", message)
            self.status_bar.showMessage("Error en procesamiento")
    
    def train_model(self):
        if not os.path.exists(self.metadata_path):
            QMessageBox.warning(self, "Error", "Primero procese el dataset")
            return
        
        # Configurar interfaz durante el entrenamiento
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Barra indeterminada
        self.set_buttons_enabled(False)
        self.status_bar.showMessage("Entrenando modelo...")
        
        # Ejecutar entrenamiento en un hilo separado
        self.training_thread = TrainingThread(self.metadata_path)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.start()
    
    def on_training_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        
        if success:
            QMessageBox.information(self, "Éxito", message)
            self.diagnose_button.setEnabled(True)
            self.status_bar.showMessage("Entrenamiento completado")
        else:
            QMessageBox.critical(self, "Error", message)
            self.status_bar.showMessage("Error en entrenamiento")
    
    def open_diagnosis_window(self):
        from views.analysis_window import AnalysisWindow
        self.analysis_window = AnalysisWindow(self.model_path, self.metadata_path)
        self.analysis_window.show()
        self.status_bar.showMessage("Módulo de diagnóstico abierto")
    
    def set_buttons_enabled(self, enabled):
        self.load_button.setEnabled(enabled)
        self.process_button.setEnabled(enabled and bool(self.zip_path))
        self.train_button.setEnabled(enabled and os.path.exists(self.metadata_path))
        self.diagnose_button.setEnabled(enabled and os.path.exists(self.model_path))
    
    def show_about(self):
        about_text = """
        <h2>Sistema de Diagnóstico de COVID-19</h2>
        <p>Versión 1.0</p>
        <p>Este sistema permite clasificar radiografías pulmonares en:</p>
        <ul>
            <li>COVID-19</li>
            <li>Opacidad pulmonar</li>
            <li>Normal</li>
            <li>Neumonía viral</li>
        </ul>
        <p>Desarrollado para el Concurso EagleDev 2025</p>
        <p><b>Restricciones cumplidas:</b></p>
        <ul>
            <li>Uso de CPU solamente</li>
            <li>Máximo 12GB de RAM</li>
            <li>Tiempo de procesamiento optimizado</li>
        </ul>
        """
        QMessageBox.about(self, "Acerca de", about_text)

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())