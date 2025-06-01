import os
import zipfile
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
import joblib

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
    
    def process_image_from_zip(self, zip_ref, img_path):
        """Procesa una imagen individual desde el ZIP"""
        with zip_ref.open(img_path) as file:
            img_array = np.frombuffer(file.read(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            # Preprocesamiento
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
            
            return np.hstack([hog_feat, lbp_hist])
    
    def process_dataset(self, max_per_class=None):
        """Procesa todo el dataset directamente desde el ZIP"""
        class_files = self.get_image_paths()
        features = []
        labels = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            for cls, files in class_files.items():
                if max_per_class:
                    files = files[:max_per_class]
                
                print(f"\nProcesando {len(files)} imágenes de {cls}...")
                for img_path in tqdm(files, desc=cls):
                    try:
                        feat = self.process_image_from_zip(zip_ref, img_path)
                        features.append(feat)
                        labels.append(self.class_map[cls])
                    except Exception as e:
                        print(f"Error procesando {img_path}: {e}")
        
        return np.array(features), np.array(labels), self.classes

# --- Bloque principal para ejecutar cuando se corre el script directamente ---
if __name__ == "__main__":
    zip_path = "C:/Users/Cliente/Documents/GitHub/ConcursoEagleDev/data/Dataset_COVID.zip" 
    processor = ZipDatasetProcessor(zip_path)
    features, labels, classes = processor.process_dataset(max_per_class=5)  # Prueba con 5 imágenes por clase
    print(f"Características procesadas: {features.shape}")
    print(f"Etiquetas procesadas: {labels.shape}")
