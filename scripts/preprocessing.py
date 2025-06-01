from utils import ZipDatasetProcessor, save_metadata
import os
import time
import psutil

def preprocess_dataset(zip_path, output_dir="data/processed", max_per_class=None):
    """
    Procesa el dataset completo y guarda los resultados
    """
    # Verificar si el archivo ZIP existe
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"No se encontró el archivo ZIP: {zip_path}")
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Iniciar procesamiento
    start_time = time.time()
    print("Iniciando preprocesamiento del dataset...")
    
    processor = ZipDatasetProcessor(zip_path)
    features, labels, classes = processor.process_dataset(max_per_class=max_per_class)
    
    # Guardar metadatos
    metadata_path = os.path.join(output_dir, "metadata.joblib")
    save_metadata(features, labels, classes, metadata_path)
    
    # Estadísticas finales
    end_time = time.time()
    ram_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    
    print("\nPreprocesamiento completado:")
    print(f"- Imágenes procesadas: {len(features)}")
    print(f"- Tiempo total: {end_time - start_time:.2f} segundos")
    print(f"- Uso máximo de RAM: {ram_usage:.2f} MB")
    print(f"- Datos guardados en: {metadata_path}")
    
    return metadata_path

if __name__ == "__main__":
    # Ejemplo de uso
    zip_path = "data/Dataset_COVID.zip"
    
    try:
        metadata_path = preprocess_dataset(zip_path, max_per_class=10)  # Probar con 10 imágenes por clase
    except Exception as e:
        print(f"Error durante el preprocesamiento: {e}")