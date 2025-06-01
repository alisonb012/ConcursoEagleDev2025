import os
import time
import psutil
from utils import ZipDatasetProcessor, save_metadata

def preprocess_dataset(zip_path, output_dir="data/processed", max_per_class=None):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"No se encontró el archivo ZIP: {zip_path}")
    os.makedirs(output_dir, exist_ok=True)
    print("Iniciando preprocesamiento del dataset...")

    processor = ZipDatasetProcessor(zip_path)
    features, labels, classes = processor.process_dataset(max_per_class=max_per_class)

    metadata_path = os.path.join(output_dir, "metadata.joblib")
    save_metadata(features, labels, classes, metadata_path)

    ram_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    print("\nPreprocesamiento completado:")
    print(f"- Imágenes procesadas: {len(features)}")
    print(f"- Uso RAM: {ram_usage:.2f} MB")
    print(f"- Datos guardados en: {metadata_path}")
    return metadata_path

if __name__ == "__main__":
    zip_path = "data/Dataset_COVID.zip"
    preprocess_dataset(zip_path, max_per_class=10)
