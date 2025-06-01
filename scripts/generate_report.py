# scripts/generate_report.py
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def generate_performance_report():
    """Genera gráficos y métricas para el reporte"""
    # Cargar datos y modelo
    data = joblib.load("data/processed/metadata.joblib")
    model = joblib.load("models/random_forest_model.joblib")
    
    # Evaluar modelo
    X_test = data['features']
    y_test = data['labels']
    y_pred = model.predict(X_test)
    
    # Matriz de confusión
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=data['class_names'],
               yticklabels=data['class_names'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdaderos')
    plt.xlabel('Predichos')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
    
    # Métricas de rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, 
                                 target_names=data['class_names'])
    
    # Guardar reporte
    with open('reports/performance_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print("Reporte generado en la carpeta 'reports'")