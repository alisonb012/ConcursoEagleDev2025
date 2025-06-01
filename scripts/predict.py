import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import joblib
from utils import load_metadata

class COVIDClassifier:
    def __init__(self, model_path, metadata_path=None):
        """
        Inicializa el clasificador cargando el modelo y metadatos
        """
        self.model = joblib.load(model_path)
        
        if metadata_path:
            metadata = load_metadata(metadata_path)
            self.class_names = metadata['class_names']
        else:
            self.class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    
    def preprocess_image(self, image):
        """
        Preprocesa una imagen para la predicción
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar y normalizar
        image = cv2.resize(image, (150, 150))
        image = cv2.equalizeHist(image)
        image = image / 255.0
        
        return image
    
    def extract_features(self, image):
        """
        Extrae características HOG y LBP de la imagen
        """
        # HOG features
        hog_feat = hog(image, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), feature_vector=True)
        
        # LBP features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))
        lbp_hist = lbp_hist.astype("float")
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        
        return np.hstack([hog_feat, lbp_hist])
    
    def predict(self, image_path):
        """
        Realiza una predicción sobre una imagen
        """
        try:
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("No se pudo cargar la imagen")
            
            # Preprocesar y extraer características
            processed_img = self.preprocess_image(image)
            features = self.extract_features(processed_img)
            
            # Realizar predicción
            class_idx = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            # Formatear resultados
            result = {
                'class': self.class_names[class_idx],
                'confidence': probabilities[class_idx],
                'probabilities': {self.class_names[i]: float(prob) 
                                for i, prob in enumerate(probabilities)}
            }
            
            return result
        except Exception as e:
            raise ValueError(f"Error durante la predicción: {str(e)}")
