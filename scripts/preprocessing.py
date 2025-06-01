import os
import joblib
from utils import ZipDatasetProcessor

def main():
    # Configuración
    zip_path = "C:/Users/Cliente/Documents/GitHub/ConcursoEagleDev/data/Dataset_COVID.zip"
    output_dir = "../data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Procesar dataset
    processor = ZipDatasetProcessor(zip_path)
    features, labels, class_names = processor.process_dataset(max_per_class=None)  # None para todas
    
    # Guardar metadatos
    metadata = {
        'features': features,
        'labels': labels,
        'class_names': class_names
    }
    joblib.dump(metadata, os.path.join(output_dir, 'processed_data.joblib'))
    print("\nPreprocesamiento completado y datos guardados!")

if __name__ == "__main__":
    main()