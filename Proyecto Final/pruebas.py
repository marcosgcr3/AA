#%%
import pandas as pd


df = pd.read_csv("student_depression_dataset.csv")
df.head()
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- ANÁLISIS EXPLORATORIO DE DATOS (EDA) ----------
# ----- 1. ELIMINAR VALORES NULOS Y ATRIBUTOS NO NECESARIOS -----

# Verificamos valores nulos (para un dataset completo)
nulos_por_columna = df.isnull().sum()
print("\nValores nulos por columna:")
print(nulos_por_columna)

# Si hay valores nulos, decidimos cómo tratarlos
if nulos_por_columna.sum() > 0:
    print("\nTratamiento de valores nulos:")

    # Tratamiento de valores nulos para variables numéricas
    for columna in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[columna].isnull().sum() > 0:
            # Imputamos la mediana para variables numéricas
            mediana = df[columna].median()
            df[columna].fillna(mediana, inplace=True)
            print(f"- Columna '{columna}': {df[columna].isnull().sum()} valores nulos imputados con la mediana ({mediana})")

    # Tratamiento de valores nulos para variables categóricas
    for columna in df.select_dtypes(include=['object']).columns:
        if df[columna].isnull().sum() > 0:
            # Imputamos la moda para variables categóricas
            moda = df[columna].mode()[0]
            df[columna].fillna(moda, inplace=True)
            print(f"- Columna '{columna}': {df[columna].isnull().sum()} valores nulos imputados con la moda ({moda})")
else:
    print("\nNo hay valores nulos en el dataset de muestra.")

# Eliminamos atributos no necesarios
# 'id' no es relevante para la predicción, lo eliminamos
df = df.drop(['id'], axis=1)


# Configuración para las visualizaciones
plt.figure(figsize=(12, 10))

# 1. Distribución de la variable objetivo
plt.subplot(3, 3, 1)
sns.countplot(x='Depression', data=df)
plt.title('Distribución de Depresión')

# 2. Relación entre género y depresión
plt.subplot(3, 3, 2)
sns.countplot(x='Gender', hue='Depression', data=df)
plt.title('Depresión por Género')

# 3. Relación entre edad y depresión
plt.subplot(3, 3, 3)
sns.boxplot(x='Depression', y='Age', data=df)
plt.title('Depresión por Edad')

# 4. Relación entre presión académica y depresión
plt.subplot(3, 3, 4)
sns.boxplot(x='Depression', y='Academic Pressure', data=df)
plt.title('Depresión por Presión Académica')

# 5. Relación entre duración del sueño y depresión
plt.subplot(3, 3, 5)
sns.countplot(x='Sleep Duration', hue='Depression', data=df)
plt.title('Depresión por Duración del Sueño')
plt.xticks(rotation=45)

# 6. Relación entre hábitos alimenticios y depresión
plt.subplot(3, 3, 6)
sns.countplot(x='Dietary Habits', hue='Depression', data=df)
plt.title('Depresión por Hábitos Alimenticios')

# 7. Relación entre pensamientos suicidas y depresión
plt.subplot(3, 3, 7)
sns.countplot(x='Have you ever had suicidal thoughts ?', hue='Depression', data=df)
plt.title('Depresión por Pensamientos Suicidas')
plt.xlabel('Pensamientos Suicidas')

# 8. Relación entre estrés financiero y depresión
plt.subplot(3, 3, 8)
sns.boxplot(x='Depression', y='Financial Stress', data=df)
plt.title('Depresión por Estrés Financiero')

# 9. Relación entre antecedentes familiares y depresión
plt.subplot(3, 3, 9)
sns.countplot(x='Family History of Mental Illness', hue='Depression', data=df)
plt.title('Depresión por Antecedentes Familiares')
plt.xlabel('Antecedentes Familiares')

plt.tight_layout()
plt.savefig('EDA_Depression.png')
plt.show()
#%%
from sklearn.preprocessing import LabelEncoder

# Separamos las características y la variable objetivo
X = df.drop(['Depression'], axis=1)
y = df['Depression']
# Identificamos las columnas por tipo
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCaracterísticas categóricas:", categorical_features)
print("\nCaracterísticas numéricas:", numerical_features)

label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le
X = df.drop(['Depression'], axis=1)
y = df['Depression']
df.head()


#%%
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score

# Dividimos los datos en conjuntos de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicamos estandarización a las variables numéricas
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])



#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lime import lime_tabular
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, f1_score, \
    roc_auc_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression




models = {

    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
     "Logistic Regression": LogisticRegression(),
    "MLP": MLPClassifier(hidden_layer_sizes=(10,10,10), activation='tanh', max_iter=500, random_state=42),
    "Arbol": RandomForestClassifier(n_estimators=100, random_state=42)
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)


    results[name] = [mse, r2, accuracy, precision, recall, f1, auc_roc]

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)


    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Clase 0", "Clase 1"], yticklabels=["Clase 0", "Clase 1"])
    plt.xlabel("Prediccion")
    plt.ylabel("Actual")
    plt.title(f"Matriz de confusion - {name}")
    plt.show()



df_results = pd.DataFrame(results, index=["mse","r2","Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"])
df_percentage = df_results.round(4).mul(100)
print(df_percentage)


#%%

# 2. Gráfico de barras comparativo de precisión, recall y f1-score por modelo
# Extracción de métricas del classification report
# (Asumiendo que ya tienes los resultados, de lo contrario deberías calcularlos)

# Para la Regresión Logística - basado en los resultados compartidos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve

# Suponiendo que ya tienes el DataFrame df_percentage

# 1. Gráfico de barras para comparar todas las métricas entre modelos
plt.figure(figsize=(14, 8))
df_results.T.plot(kind='bar', figsize=(14, 8))
plt.title('Comparación de Métricas de Rendimiento Entre Modelos', fontsize=16)
plt.xlabel('Modelo', fontsize=14)
plt.ylabel('Valor (0-1)', fontsize=14)
plt.ylim(0, 1.0)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('comparacion_general_modelos.png')
plt.show()

# 2. Gráfico de radar para visualizar el rendimiento multidimensional
metrics = df_results.index
model_names = df_results.columns

plt.figure(figsize=(10, 10))
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Cerrar el polígono

# Preparar el gráfico
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

for model in model_names:
    values = df_results[model].tolist()
    values += values[:1]  # Cerrar el polígono
    ax.plot(angles, values, linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.1)

# Configurar etiquetas
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
ax.set_ylim(0, 1)
plt.title('Diagrama de Radar de Métricas por Modelo', fontsize=16)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.grid(True)
plt.tight_layout()
plt.savefig('diagrama_radar_modelos.png')
plt.show()

# 3. Heatmap de las métricas para todos los modelos
plt.figure(figsize=(12, 8))
sns.heatmap(df_results, annot=True, cmap="YlGnBu", fmt='.3f', linewidths=.5)
plt.title('Heatmap de Métricas de Rendimiento', fontsize=16)
plt.tight_layout()
plt.savefig('heatmap_metricas.png')
plt.show()

# 4. Gráfico de barras para cada métrica individual
for metric in metrics:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, df_results.loc[metric], color=sns.color_palette("muted"))
    plt.title(f'Comparación de {metric} Entre Modelos', fontsize=16)
    plt.xlabel('Modelo', fontsize=14)
    plt.ylabel(f'Valor de {metric}', fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir etiquetas de valor en cada barra
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'comparacion_{metric.lower().replace("-", "_")}.png')
    plt.show()

# 5. Simulación de curvas ROC para comparar modelos
# Nota: Esto es una simulación basada en los valores de AUC-ROC proporcionados
plt.figure(figsize=(10, 8))

# Colores para cada modelo
colors = ['blue', 'red', 'green', 'purple']
model_aucs = df_results.loc['AUC-ROC']

# Simulación de curvas ROC basadas en los valores de AUC
for i, model in enumerate(model_names):
    auc_value = model_aucs[model]

    # Crear una curva ROC simulada basada en el valor de AUC
    # Esto es una aproximación y no representa la curva ROC real
    fpr = np.linspace(0, 1, 100)

    # Simular una curva basada en el AUC (esto es una aproximación)
    # Una curva con mayor AUC tendrá un mejor rendimiento
    tpr = fpr**(1/auc_value)

    plt.plot(fpr, tpr, color=colors[i], label=f'{model} (AUC = {auc_value:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aleatoria (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
plt.title('Comparación de Curvas ROC (Simulación)', fontsize=16)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacion_curvas_roc.png')
plt.show()

# 6. Gráfico de barras para comparar accuracy y F1-score
plt.figure(figsize=(12, 7))
width = 0.35
x = np.arange(len(model_names))

plt.bar(x - width/2, df_results.loc['Accuracy'], width, label='Accuracy', color='skyblue')
plt.bar(x + width/2, df_results.loc['F1 Score'], width, label='F1 Score', color='salmon')

plt.xlabel('Modelo', fontsize=14)
plt.ylabel('Valor', fontsize=14)
plt.title('Comparación de Accuracy y F1 Score', fontsize=16)
plt.xticks(x, model_names)
plt.ylim(0, 1.0)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir etiquetas encima de las barras
for i, v in enumerate(df_results.loc['Accuracy']):
    plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
for i, v in enumerate(df_results.loc['F1 Score']):
    plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('comparacion_accuracy_f1.png')
plt.show()
"""metrics_log = {
    'precision': [0.82, 0.85],
    'recall': [0.79, 0.87],
    'f1-score': [0.80, 0.86],
    'accuracy': 0.84
}

# Para el MLP - ajusta estos valores según tus resultados reales
metrics_mlp = {
    'precision': [0.80, 0.83],  # Valores de ejemplo, ajusta con tus resultados
    'recall': [0.77, 0.85],     # Valores de ejemplo, ajusta con tus resultados
    'f1-score': [0.78, 0.84],   # Valores de ejemplo, ajusta con tus resultados
    'accuracy': 0.82            # Valores de ejemplo, ajusta con tus resultados
}

# Configuración de gráficos comparativos
classes = ['Clase 0', 'Clase 1']
models = ['Regresión Logística', 'MLP']
metrics = ['Precisión', 'Recall', 'F1-Score']

# Gráfico comparativo de métricas para clase 0
plt.figure(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, [metrics_log['precision'][0], metrics_log['recall'][0], metrics_log['f1-score'][0]],
        width, label='Regresión Logística', color='darkorange')
plt.bar(x + width/2, [metrics_mlp['precision'][0], metrics_mlp['recall'][0], metrics_mlp['f1-score'][0]],
        width, label='MLP', color='green')

plt.ylabel('Valor', fontsize=12)
plt.title('Comparación de métricas para Clase 0', fontsize=14)
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico comparativo de métricas para clase 1
plt.figure(figsize=(12, 6))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, [metrics_log['precision'][1], metrics_log['recall'][1], metrics_log['f1-score'][1]],
        width, label='Regresión Logística', color='darkorange')
plt.bar(x + width/2, [metrics_mlp['precision'][1], metrics_mlp['recall'][1], metrics_mlp['f1-score'][1]],
        width, label='MLP', color='green')

plt.ylabel('Valor', fontsize=12)
plt.title('Comparación de métricas para Clase 1', fontsize=14)
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Gráfico comparativo de exactitud (accuracy) global
plt.figure(figsize=(8, 6))
plt.bar(['Regresión Logística', 'MLP'], [metrics_log['accuracy'], metrics_mlp['accuracy']],
        color=['darkorange', 'green'])
plt.ylabel('Exactitud (Accuracy)', fontsize=12)
plt.title('Comparación de Exactitud Global', fontsize=14)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Visualización con LIME para explicabilidad (para un ejemplo aleatorio)
# Crear el explicador LIME
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),  # Asume que X_train es un DataFrame
    class_names=['Clase 0', 'Clase 1'],
    mode='classification'
)

# Seleccionar un ejemplo aleatorio para explicar
np.random.seed(42)  # Para reproducibilidad
random_idx = np.random.randint(0, len(X_test))
random_instance = X_test.iloc[random_idx].values  # Asume que X_test es un DataFrame

# Explicar la predicción de Regresión Logística
exp_log = explainer.explain_instance(random_instance, log_model.predict_proba, num_features=6)
plt.figure(figsize=(10, 6))
exp_log.as_pyplot_figure()
plt.tight_layout()
plt.title("Explicabilidad LIME - Regresión Logística", fontsize=14)
plt.show()

# Explicar la predicción de MLP
exp_mlp = explainer.explain_instance(random_instance, mlp.predict_proba, num_features=6)
plt.figure(figsize=(10, 6))
exp_mlp.as_pyplot_figure()
plt.tight_layout()
plt.title("Explicabilidad LIME - MLP", fontsize=14)
plt.show()"""