import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_data(csv_path):
    # Cargar datos
    df = pd.read_csv(csv_path, parse_dates=['Fecha'])
    
    # Análisis básico
    print("\nResumen estadístico:")
    print(df.describe())
    
    # Gráficos (se pueden usar estos en Power BI)
    plt.figure(figsize=(12, 6))
    
    # Distribución de clases
    plt.subplot(1, 2, 1)
    df['Clase_Predicha'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Distribución de Diagnósticos')
    
    # Probabilidad por edad
    plt.subplot(1, 2, 2)
    pd.pivot_table(df, values='Probabilidad', 
                  index='Edad', columns='Clase_Predicha').plot()
    plt.title('Probabilidad por Edad')
    plt.tight_layout()
    
    # Guardar gráficos
    plt.savefig('powerbi_analysis.png')
    print("\nGráficos guardados en 'powerbi_analysis.png'")

if __name__ == "__main__":
    csv_file = input("Ruta del archivo CSV exportado: ")
    analyze_data(csv_file)