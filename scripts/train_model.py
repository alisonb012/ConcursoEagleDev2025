from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_metadata

def train_model(features, labels, class_names):
    """
    Entrena y evalúa el modelo de clasificación
    """
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42)
    
    # Balancear clases con SMOTE
    print("\nBalanceando clases con SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Configurar modelo optimizado para CPU
    print("Configurando modelo Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,  # Usar todos los cores de CPU
        random_state=42,
        class_weight='balanced',
        verbose=1
    )
    
    # Entrenar modelo
    print("\nIniciando entrenamiento...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar modelo
    print("\nEvaluando modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generar reporte de clasificación
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Generar matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    
    # Crear directorio para reportes si no existe
    os.makedirs("reports", exist_ok=True)
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
    
    # Estadísticas de rendimiento
    ram_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    
    print("\nResultados del entrenamiento:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- Tiempo de entrenamiento: {training_time:.2f} segundos")
    print(f"- Uso máximo de RAM: {ram_usage:.2f} MB")
    print("\nReporte de Clasificación:")
    print(report)
    
    return model, accuracy, report

def train_and_save_model(metadata_path, model_dir="models"):
    """
    Flujo completo de entrenamiento y guardado del modelo
    """
    # Cargar datos preprocesados
    metadata = load_metadata(metadata_path)
    features, labels = metadata['features'], metadata['labels']
    class_names = metadata['class_names']
    
    # Entrenar modelo
    model, accuracy, report = train_model(features, labels, class_names)
    
    # Guardar modelo
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "covid_classifier.joblib")
    joblib.dump(model, model_path)
    
    # Guardar reporte
    report_path = os.path.join("reports", "training_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\nModelo guardado en: {model_path}")
    print(f"Reporte guardado en: {report_path}")
    
    return model_path

if __name__ == "__main__":
    # Ejemplo de uso
    metadata_path = "data/processed/metadata.joblib"
    
    try:
        model_path = train_and_save_model(metadata_path)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")