import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def generarDatosFC():
    x = np.arange(1, 61)  # Tiempos de 1 a 60 segundos
    ruido = np.random.uniform(-5.9, 5.9, size=60)  # Ruido aleatorio
    y = 0.7 * x + 60 + ruido  # Frecuencia cardíaca simulada
    return x, y




def analizarFC(x, y):
    # Adaptar x para LinearRegression usando List Comprehension
    x_reshaped = [[i] for i in x]

    # Crear y ajustar el modelo de regresión lineal
    modelo = LinearRegression()
    modelo.fit(x_reshaped, y)

    # Obtener coeficiente e intercepto
    coeficiente = modelo.coef_[0]
    intercepto = modelo.intercept_
    print(f"Coefficients: [{coeficiente:.8f}] Intercept: {intercepto:.8f}")

    # Obtener los valores predichos de y
    y_pred = modelo.predict(x_reshaped)

    # Graficar los datos simulados y la regresión lineal
    plt.scatter(x, y, label='Datos simulados')
    plt.plot(x, y_pred, color='red', label='Regresión lineal')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Frecuencia cardíaca (bpm)')
    plt.legend()
    plt.show()

# Generar los datos
x, y = generarDatosFC()

# Analizar y visualizar
analizarFC(x, y)