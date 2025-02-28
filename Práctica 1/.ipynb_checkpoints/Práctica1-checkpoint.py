# Parte A: Análisis y procesamiento de un dataset

# Importar librerías necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
df = pd.read_csv('titanic.csv')
df_cleaned = df.dropna().drop_duplicates()
df_cleaned.reset_index(drop=True, inplace=True)
filas_eliminadas = len(df) - len(df_cleaned)
print(f'Filas eliminadas: {filas_eliminadas}')

# 2. Determinar atributos no útiles
atributos_no_utiles = ['Ticket']


# 3. Relaciones entre atributos
sns.pairplot(df_cleaned)
plt.show()

# Seleccionar únicamente columnas numéricas para calcular la matriz de correlación
df_numeric = df_cleaned.select_dtypes(include=[float, int])  # Filtra columnas numéricas
correlaciones = df_numeric.corr()
print(correlaciones)

# 4. Atributos numéricos
estadisticas = df_cleaned.describe()
print(estadisticas)

# 5. Atributos categóricos
categorical_columns = ['Sex', 'Embarked', 'Pclass']
for column in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_cleaned, x=column)
    plt.title(f'Frecuencia de {column}')
    plt.show()
    print(f'{column}:')
    print(f'Valores distintos: {df_cleaned[column].nunique()}')
    print(f'Valor más frecuente: {df_cleaned[column].mode()[0]}')

# 6. Determinar outliers
numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch']
for column in numerical_columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_cleaned, x=column)
    plt.title(f'Boxplot de {column}')
    plt.show()

# 7. Convertir atributos categóricos en valores numéricos
df_onehot = pd.get_dummies(df_cleaned, columns=categorical_columns)
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    label_encoders[column] = le
print(df_onehot.head())
print(df_cleaned.head())

# 8. Normalizar y estandarizar el dataset
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

# MinMaxScaler
df_normalized = pd.DataFrame(
    scaler_minmax.fit_transform(df_numeric),
    columns=df_numeric.columns
)

# StandardScaler
df_standardized = pd.DataFrame(
    scaler_standard.fit_transform(df_numeric),
    columns=df_numeric.columns
)

# Mostrar resultados
print("Normalización (MinMaxScaler):")
print(df_normalized.head())

print("\nEstandarización (StandardScaler):")
print(df_standardized.head())

# Parte B: Evaluación de modelos de AA

# Importar librerías necesarias
from surprise import Dataset, Reader, KNNBasic, SVD, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy

# Definir SEED
SEED = 42

# Cargar el dataset de MovieLens de 100K
data = Dataset.load_builtin('ml-100k')

# Dividir el dataset
trainset, testset = train_test_split(data, test_size=0.75, random_state=SEED) #Esta funcion divide un conjunto de datos en dos subconjuntos: uno para entrenamiento y otro para prueba

# Evaluar distintos algoritmos de recomendación
# Filtrado colaborativo basado en vecinos (KNNBasic)
algo_knn_user = KNNBasic(sim_options={'name': 'pearson', 'user_based': True}, random_state=SEED)
algo_knn_item = KNNBasic(sim_options={'name': 'pearson', 'user_based': False}, random_state=SEED)

# Filtrado colaborativo basado en modelos (SVD y NMF)
algo_svd = SVD(random_state=SEED)
algo_nmf = NMF(random_state=SEED)

# Entrenar y evaluar los algoritmos
algorithms = [("KNN-User-Based", algo_knn_user),
              ("KNN-Item-Based", algo_knn_item),
              ("SVD", algo_svd),
              ("NMF", algo_nmf)]

# Mostrar resultados de 5 predicciones para cada algoritmo
for name, algo in algorithms:
    print(f"Evaluando {name}...")
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    print(f"{name} RMSE: {rmse}")

    print("Algunas predicciones:")
    for pred in predictions[:5]:
        print(pred)
    print()

from sklearn.metrics import precision_score, recall_score, ndcg_score
import numpy as np
def evaluate_metrics(predictions, k=10):
    # Convertir predicciones a un formato manejable
    relevant_items = []
    recommended_items = []
    scores = []

    for uid, iid, true_r, est, details in predictions:
        relevant_items.append(1 if true_r > 4 else 0)  # 1: relevante, 0: no relevante
        scores.append(est)

    # Obtener índices de las top-k recomendaciones
    sorted_indices = np.argsort(scores)[::-1][:k]
    recommended_items = [1 if i in sorted_indices else 0 for i in range(len(scores))]

    # Calcular Precision@k, Recall@k, NDCG@k
    precision = precision_score(relevant_items, recommended_items, zero_division=1)
    recall = recall_score(relevant_items, recommended_items, zero_division=1)
    ndcg = ndcg_score([relevant_items], [scores], k=k)

    return precision, recall, ndcg


# Evaluar cada algoritmo
results = []
for name, algo in algorithms:
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    precision, recall, ndcg = evaluate_metrics(predictions, k=10)

    results.append({
        "Modelo": name,
        "RMSE": rmse,
        "Precision@10": precision,
        "Recall@10": recall,
        "NDCG@10": ndcg
    })
import pandas as pd

# Crear un DataFrame a partir de los resultados
results_df = pd.DataFrame(results)

# Mostrar la tabla
print(results_df)
'''
- **RMSE**: El modelo **SVD** tiene el menor error (RMSE más bajo), lo que indica que realiza predicciones más cercanas a las calificaciones reales.
- **Precision@10**: El modelo **SVD** proporciona la mayor precisión, es decir, recomienda un mayor número de películas relevantes entre las top-10.
- **Recall@10**: **SVD** también tiene el mejor desempeño al recuperar un mayor porcentaje de películas relevantes.
- **NDCG@10**: **SVD** cuenta con el mayor valor de NDCG, lo que indica una mejor organización de recomendaciones relevantes en posiciones altas de la lista.

El modelo **SVD** es el mejor entre los evaluados porque:
1. Tiene el menor RMSE, lo que confirma la calidad de sus predicciones.
2. Sobresale en las métricas de Precision@10, Recall@10 y NDCG@10, demostrando que ofrece películas relevantes y bien ordenadas dentro de las top-10

'''