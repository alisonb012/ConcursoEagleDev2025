from PyQt6.QtWidgets import QApplication
from main import MainWindow
import joblib
import sys

if __name__ == "__main__":
    zip_path = "data/Dataset_COVID.zip"
    model_path = "models/covid_classifier.joblib"
    metadata = joblib.load("data/processed/metadata.joblib")
    class_names = metadata['class_names']

    app = QApplication(sys.argv)
    window = MainWindow(model_path, zip_path, class_names)
    window.show()
    sys.exit(app.exec())