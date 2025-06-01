import os
import joblib
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from utils import ZipDatasetProcessor

def load_processed_data():
    """Carga los datos preprocesados"""
    data_path = "../data/processed/processed_data.joblib"
    return joblib.load(data_path)

def train_and_evaluate(X, y, class_names):
    """Entrena y evalúa el modelo"""
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Balancear clases
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Configurar modelo
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        class_weight='balanced'
    )
    
    # Entrenar
    print("Entrenando modelo...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Evaluar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Reportes
    print("\nResultados:")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    
    # Guardar reportes
    os.makedirs("../reports", exist_ok=True)
    plt.savefig("../reports/confusion_matrix.png")
    plt.close()
    
    # Métricas de rendimiento
    ram_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # MB
    
    report = {
        'accuracy': accuracy,
        'training_time': training_time,
        'ram_usage': ram_usage,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'classes': class_names
    }
    
    joblib.dump(report, "../reports/training_report.joblib")
    
    return model, report

def main():
    # Cargar datos
    data = load_processed_data()
    X, y, class_names = data['features'], data['labels'], data['class_names']
    
    # Entrenar modelo
    model, report = train_and_evaluate(X, y, class_names)
    
    # Guardar modelo
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/covid_classifier.joblib")
    
    print("\nEntrenamiento completado!")
    print(f"Accuracy final: {report['accuracy']:.4f}")
    print(f"Tiempo de entrenamiento: {report['training_time']/60:.2f} minutos")
    print(f"Uso máximo de RAM: {report['ram_usage']:.2f} MB")

if __name__ == "__main__":
    main()