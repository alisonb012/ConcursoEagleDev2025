import sys
from PyQt6.QtWidgets import QApplication
from views.main_window import MainWindow

def main():
    # Configurar aplicación
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Estilo moderno
    
    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()
    
    # Ejecutar bucle principal
    sys.exit(app.exec())

if __name__ == "__main__":
    main()