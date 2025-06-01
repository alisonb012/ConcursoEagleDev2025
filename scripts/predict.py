# import zipfile
# import numpy as np
# import cv2
# from skimage.feature import hog, local_binary_pattern
# import joblib

# def load_model(model_path):
#     model = joblib.load(model_path)
#     return model

# def preprocess_image_from_zip(zip_path, img_path):
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         with zip_ref.open(img_path) as file:
#             img_array = np.frombuffer(file.read(), dtype=np.uint8)
#             img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
#             img = cv2.resize(img, (150, 150))
#             img = cv2.equalizeHist(img)
#             img = img / 255.0

#             hog_feat = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                            cells_per_block=(1, 1), feature_vector=True)

#             radius = 3
#             n_points = 8 * radius
#             lbp = local_binary_pattern(img, n_points, radius, method='uniform')
#             lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))
#             lbp_hist = lbp_hist.astype("float")
#             lbp_hist /= (lbp_hist.sum() + 1e-6)

#             return np.hstack([hog_feat, lbp_hist])

# def predict_single_image(model, zip_path, img_path, class_names):
#     features = preprocess_image_from_zip(zip_path, img_path).reshape(1, -1)
#     pred_idx = model.predict(features)[0]
#     pred_prob = model.predict_proba(features)[0]
#     return class_names[pred_idx], pred_prob

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_image_from_path(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    
    img = cv2.resize(img, (150, 150))
    img = cv2.equalizeHist(img)
    img = img / 255.0

    hog_feat = hog(img, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), feature_vector=True)

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    return np.hstack([hog_feat, lbp_hist])

def predict_single_image(model, image_path=None, image_in_zip=None, class_names=None):
    if image_path:
        features = preprocess_image_from_path(image_path).reshape(1, -1)
    else:
        raise ValueError("Se requiere image_path para predecir una imagen externa.")

    pred_idx = model.predict(features)[0]
    pred_prob = model.predict_proba(features)[0]
    return class_names[pred_idx], pred_prob
