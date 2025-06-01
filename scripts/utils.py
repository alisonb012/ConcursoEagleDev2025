import os
import zipfile
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
import joblib
import psutil
from multiprocessing import Pool, cpu_count

class ZipDatasetProcessor:
    def __init__(self, zip_path):
        self.zip_path = zip_path
        self.classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.class_map = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def get_image_paths(self):
        """Obtiene todas las rutas de imágenes en el ZIP organizadas por clase"""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            all_files = zip_ref.namelist()
        
        class_files = {cls: [] for cls in self.classes}
        
        for file in all_files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            for cls in self.classes:
                if f"{cls}/" in file:
                    class_files[cls].append(file)
                    break
        
        return class_files
    
    @staticmethod
    def process_single_image(args):
        """Procesa una imagen individual (para multiprocessing)"""
        zip_path, img_path, cls, class_map = args
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                with zip_ref.open(img_path) as file:
                    img_array = np.frombuffer(file.read(), dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    
                    # Preprocesamiento optimizado
                    img = cv2.resize(img, (150, 150))
                    img = cv2.equalizeHist(img)
                    img = img / 255.0
                    
                    # Extracción de características
                    hog_feat = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), feature_vector=True)
                    
                    radius = 3
                    n_points = 8 * radius
                    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
                    lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))
                    lbp_hist = lbp_hist.astype("float")
                    lbp_hist /= (lbp_hist.sum() + 1e-6)
                    
                    return (np.hstack([hog_feat, lbp_hist]), class_map[cls])
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
            return None
    
    def process_dataset(self, max_per_class=None, batch_size=100):
        """Procesa el dataset completo con control de memoria"""
        class_files = self.get_image_paths()
        all_features = []
        all_labels = []
        
        for cls in self.classes:
            files = class_files[cls]
            if max_per_class:
                files = files[:max_per_class]
            
            print(f"\nProcesando {len(files)} imágenes de {cls}...")
            
            # Procesamiento por lotes para control de memoria
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                
                # Preparar argumentos para multiprocessing
                args = [(self.zip_path, img_path, cls, self.class_map) for img_path in batch]
                
                # Usar multiprocessing para acelerar el procesamiento
                with Pool(processes=max(1, cpu_count()-1)) as pool:
                    results = list(tqdm(pool.imap(self.process_single_image, args), 
                                     total=len(batch), desc=f"Lote {i//batch_size + 1}"))
                
                # Recolectar resultados válidos
                valid_results = [r for r in results if r is not None]
                if valid_results:
                    features, labels = zip(*valid_results)
                    all_features.extend(features)
                    all_labels.extend(labels)
                
                # Monitoreo de memoria
                ram_usage = psutil.Process().memory_info().rss / (1024 ** 2)
                print(f"RAM usada: {ram_usage:.2f} MB", end='\r')
        
        return np.array(all_features), np.array(all_labels), self.classes

def save_metadata(features, labels, classes, output_path):
    """Guarda los datos procesados en formato eficiente"""
    import time
    metadata = {
        'features': features,
        'labels': labels,
        'class_names': classes,
        'timestamp': time.time(),
        'shape': features.shape
    }
    joblib.dump(metadata, output_path, compress=3)

def load_metadata(input_path):
    """Carga los datos procesados"""
    return joblib.load(input_path)

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # Ejemplo de uso para pruebas
    zip_path = "data/Dataset_COVID.zip"
    processor = ZipDatasetProcessor(zip_path)
    
    # Probar con 5 imágenes por clase
    features, labels, classes = processor.process_dataset(max_per_class=5)
    
    print(f"\nCaracterísticas procesadas: {features.shape}")
    print(f"Etiquetas procesadas: {labels.shape}")
    print(f"Tiempo de ejecución: {time.time() - start_time:.2f} segundos")
