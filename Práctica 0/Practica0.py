#%% [markdown]
### Práctica 0:
# El desarrollo de esta práctica consiste en implementar de dos maneras, una sin usar numpy y otra usándolo, el cálculo
# de la integral de una función comprendida entre dos valores de x (a y b) mediante el método de Monte Carlo.
# Este método, es una aproximación probabilística al resultado de una integral, para ello distribuimos aleatoriamente
# puntos en la gŕafica y contabilizamos el porcentaje de los mismos que residen en el interior de la misma. Ese porcentaje
# multiplicado por el area total del rectángulo formado por (b-a) y M (máximo de la función en el intervalo a b) nos da el
# aproximado total de la integral de la misma.
#
# El desarrollo de esta práctica ha sido llevada a cabo por: Jaime Alonso Fernández y Marcos Gomez Cortes
#
#
#
# Antes de comenzar con las implementaciones, importamos todas las librerias necesarias para la ejecución.
#%%
import random
import time
import matplotlib.pyplot as plt
import random
import time
import numpy as np
#%% [markdown]
### Fórmula de Monte Carlo:
# Cálculo de Monte Carlo para la integral. Este método devuelve el cálculo de la fórmula expresada en el enunciado de la práctica.
#%%
def calcula_integral_monte_carlo(debajo, num_puntos, relacion_a_b, m):
    return (debajo / num_puntos) * relacion_a_b * m
#%% [markdown]
## Implementación modo iterativo:
# Este método devuelve no solo el resultado numérico de la integral si no que también el tiempo que le tomó calcularla.
#%%
# Declaramos una constante para todo el ejercicio
NUMERO_PUNTOS_FUNCION = 100
def integra_mc_it(func, a, b, num_puntos=10000):
    # Caso extremo: Verificar que el rango es válido
    if a == b:
        print('Intervalo inválido (a = b)')
        return

    # Comprobamos el orden correcto de las variables
    if a > b:
        a, b = b, a

    # Iniciamos la evaluación temporal
    start_time = time.time()

    # Buscamos el valor mayor y menor de la gráfica en el intervalo dado, con una precisión de NUMERO_PUNTOS_FUNCION
    m, m_min = 0, 0
    for x in range(NUMERO_PUNTOS_FUNCION):
        y = func(x)
        m = max(m, y)
        m_min = min(m_min, y)

    # Caso erróneo
    if m_min < 0:
        print(f'Función inválida al contar con puntos en el rango {a} - {b} por debajo de 0')
        return -1, -1

    # Inicializamos una variable de cuenta
    numero_puntos_dentro = 0
    # Generamos puntos aleatorios dentro del rectángulo definido por [a, b] y [0, m] y contamos cuántos de ellos
    # están por "dentro" o "debajo" de la función.
    for _ in range(num_puntos):
        x = random.uniform(a, b)
        y = random.random() * m
        if y <= func(x):
            numero_puntos_dentro += 1

    # Realizamos el cálculo final
    resultado = calcula_integral_monte_carlo(numero_puntos_dentro, num_puntos, b - a, m)

    # Finalizamos la evaluación temporal al acabar el cálculo
    end_time = time.time()

    # Devuelve una tupla que almacena el resultado y el tiempo
    return resultado,  end_time - start_time

#%% [markdown]
### Implementación modo vectorial (usando numpy):
# Este método devuelve no solo el resultado numérico de la integral si no que también el tiempo que le tomó calcularla.
# Incluye la opción de representar gráficamente la función con los puntos dentro y fuera de la misma.
#%%

def integra_mc_np(func, a, b, num_puntos=10000, mostrar_plot=False):
    # Caso extremo: Verificar que el rango es válido
    if a == b:
        print('Intervalo inválido (a = b)')
        return

    # Comprobamos el orden correcto de las variables
    if a > b:
        a, b = b, a

    #Iniciamos la evaluación temporal
    start_time = time.time()

    # Obtiene una lista de 100 puntos x equiespaciados en el intervalo [a, b]
    x_points = np.linspace(a, b, NUMERO_PUNTOS_FUNCION)

    # Calcula los valores de la función en cada punto x
    y_points = func(x_points)

    # Encuentra el valor máximo de la función en el intervalo dado
    m = np.max(y_points)

    # Si existen valores negativos de la función en el rango proporcionado, se detiene la ejecución.
    if np.min(y_points) < 0:
        print(f'Función inválida al tener valores por debajo de 0 en el rango {a} - {b}')
        return -1, -1

    # Generamos puntos aleatorios dentro del rectángulo definido por [a, b] y [0, m]
    # Se separan en dos listas: valores de x y valores de y
    x = np.random.uniform(a, b, num_puntos)
    y = np.random.uniform(0, max(func(x)), num_puntos)

    # Contamos los puntos que están por debajo o "dentro" de la función
    numero_puntos_dentro = np.sum(y[y < func(x)])

    # Realizamos el cálculo final
    resultado = calcula_integral_monte_carlo(numero_puntos_dentro, num_puntos, b - a, m)

    #Finalizamos la evaluación temporal
    end_time = time.time()

    # Si se decide mostrar la gráfica, procederemos a hacerlo ahora sin afectar al tiempo.
    if mostrar_plot:
        # Dibuja la función
        plt.plot(x_points, y_points, 'b', label=f'Función')
        # Establece los límites en el eje x e y, para que correspondan con los rangos a-b y 0-m
        plt.xlim(a, b)
        plt.ylim(0, m)
        # Representa en la gráfica los puntos primero de verde todos
        plt.scatter(x, y, s=1, color='g', label='Puntos debajo de f')
        # Colorea los puntos incorrectos de rojo
        plt.scatter(x[y > func(x)], y[y > func(x)], s=1, color='r', label='Puntos encima de f')
        # Título de la gŕafica
        plt.title(f'Método de integración por Monte Carlo (Numpy - {num_puntos}ptos)')
        # Etiquetas de los ejes
        plt.xlabel('x')
        plt.ylabel('y')
        # Posiciona la leyenda fuera del gráfico
        plt.legend(bbox_to_anchor=(1.05, 1))
        # Muestra la gráfica
        plt.show()

    # Devuelve una tupla que almacena el resultado y el tiempo
    return resultado, end_time - start_time


#%% [markdown]
### Resultados:
# Para esta sección comenzamos definiendo 1 funciones. De manera que realizaremos el cálculo de su integral en el rango
# -1, 1, usando un total de 20 puntos diferentes repartidos entre 100 y 10000000 y utilizando las dos implementaciones del cálculo.
# Establecido esto, ejecutaremos alternativamente las dos implementaciones y guardaremos el tiempo de ejecución de cada
# una, para luego pasar a mostrar unos gráficos con la comparativa de ambas implementaciones.
#
# Cabe mencionar también que se muestran también los gráficos siempre que trabajemos con menos de 1000000 puntos en el caso
# de la implementación vectorial.
#
#%%
fun = lambda x: x*x

# Rangos y puntos
puntos_ejecucion = np.linspace(100, 10000000, 20, dtype=int)



# Listas para los tiempos
tiempos_iterativa = []
tiempos_numpy = []

# Iteramos sobre los puntos de ejecución
for puntos in puntos_ejecucion:

    mostrar_plot = puntos <= 1000000
    resultado_iterativa, tiempo_iterativa = integra_mc_it(fun, -1, 1, puntos)

    if resultado_iterativa == -1:
        break

    print(f"Resultado iterativa con {puntos} puntos: {resultado_iterativa}u, Tiempo: {tiempo_iterativa}s")
    tiempos_iterativa.append(tiempo_iterativa)


    resultado_numpy, tiempo_numpy = integra_mc_np(fun, -1, 1, puntos, mostrar_plot)
    tiempos_numpy.append(tiempo_numpy)
    if resultado_iterativa == -1:
        break
    else:
        print(f"Resultado numpy con {puntos} puntos: {resultado_numpy}u, Tiempo: {tiempo_numpy}s")


if tiempos_iterativa[0] != -1 and tiempos_iterativa[0] != -1:
    # Graficamos los resultados para esta función con ambas versiones
    plt.plot(puntos_ejecucion, tiempos_iterativa, 'r', label="Tiempos con la función iterativa")
    plt.plot(puntos_ejecucion, tiempos_numpy, 'b', label="Tiempos con la función vectorial (numpy)")
    plt.xlabel("Número de Puntos")
    plt.ylabel("Tiempo (segundos)")
    plt.title(f"Tiempos de ejecución")
    plt.xticks(puntos_ejecucion, puntos_ejecucion, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()

#%% [markdown]
### Conclusión:
# Con la realización de esta práctica podemos apreciar lo útiles que son las librerías de python, debido a su simplicidad
# y velocidad con respecto al código escrito en el propio python. Investigando un poco hemos visto que esto se consigue
# debido a que estas librerías están programadas en C
#%%