import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# División del dataset y métricas de evaluación
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score

# Modelo de regresión lineal
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

# Modelo de red neuronal


# Para escalado de variables
from sklearn.preprocessing import StandardScaler

# Cargar dataset (asegúrate de que el archivo se encuentra en el mismo directorio o usa la ruta correcta)
df = pd.read_csv("student_depression_dataset.csv")

# Mostrar las primeras filas y detalles del dataset
print(df.head())
print(df.info())
# Verificar datos nulos en cada columna
print(df.isnull().sum())

# Visualizar la distribución de la variable objetivo "Depression"
plt.figure(figsize=(8,5))
sns.countplot(x='Depression', data=df)
plt.title("Distribución de la variable Depression")
plt.show()
# Eliminar la columna de identificador
df = df.drop("id", axis=1)

# Revisar las columnas y determinar cuales son categóricas
print(df.columns)

# Lista de columnas categóricas (puedes ajustar según consideres que algunas pueden tener orden)
categorical_cols = ["Gender", "City", "Profession", "Sleep Duration", "Dietary Habits", "Degree",
                    "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]

# Aplicar one-hot encoding a las variables categóricas
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Verificar el resultado
print(df_encoded.head())

# Separar variables independientes (X) y la variable dependiente (y)
# Usamos "Depression" como variable a predecir
X = df_encoded.drop("Depression", axis=1)
y = df_encoded["Depression"]

# Normalizar las variables independientes para el modelo de red neuronal
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Inicializar y entrenar el modelo de regresión lineal
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Realizar predicciones sobre el conjunto de prueba
y_pred_lr = lr_model.predict(X_test)

# Calcular métricas de evaluación
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
accuracy = accuracy_score(y_test, y_pred_lr)
precision = precision_score(y_test, y_pred_lr)
recall = recall_score(y_test, y_pred_lr)
f1 = f1_score(y_test, y_pred_lr)
auc_roc = roc_auc_score(y_test, y_pred_lr)
print("Modelo Regresión Lineal:")
print("MSE:", mse_lr)
print("R^2:", r2_lr)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("AUC:", auc_roc)
# Recuperar el nombre de las variables a partir de df_encoded (ya que se usó get_dummies)
feature_names = X.columns

coef_df = pd.DataFrame({
    'Variable': feature_names,
    'Coeficiente': lr_model.coef_
})

print(coef_df.sort_values(by='Coeficiente', key=abs, ascending=False))
# Recuperar el nombre de las variables a partir de df_encoded (ya que se usó get_dummies)
feature_names = X.columns

coef_df = pd.DataFrame({
    'Variable': feature_names,
    'Coeficiente': lr_model.coef_
})

print(coef_df.sort_values(by='Coeficiente', key=abs, ascending=False))

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), activation='tanh', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)
accuracy = accuracy_score(y_test, y_pred_mlp)
precision = precision_score(y_test, y_pred_mlp)
recall = recall_score(y_test, y_pred_mlp)
f1 = f1_score(y_test, y_pred_mlp)
auc_roc = roc_auc_score(y_test, y_pred_mlp)
print("Modelo MLP:")
print("MSE:", mse_mlp)
print("R^2:", r2_mlp)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
print("AUC ROC:", auc_roc)
