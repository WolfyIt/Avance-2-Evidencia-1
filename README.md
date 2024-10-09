1. Seleccionar un dataset que contenga variables numéricas adecuadas para aplicar K-Means
En este caso, utilizaremos un dataset de Pokémon que incluye variables como base_attack, base_defense, y base_stamina. Estas variables numéricas son adecuadas para agrupar Pokémon en diferentes clusters según sus estadísticas básicas.

2. Leer el dataset seleccionado en Python
Primero, cargamos el dataset en Python:

python
Copy code
import pandas as pd

# Cargar el dataset desde un archivo CSV
basic1 = pd.read_csv("/content/drive/MyDrive/DataSets/Ejercicios/Practica 2/pokemon.csv")

# Verificar las primeras filas para confirmar la lectura correcta
basic1.head()
3. Realizar una limpieza de datos
Para asegurar que el dataset esté listo para el modelo de K-Means, realizamos la limpieza de datos:

Seleccionamos solo las columnas numéricas (base_attack, base_defense, base_stamina).
Eliminamos las filas que contengan valores nulos en estas columnas.
python
Copy code
# Seleccionar solo las columnas numéricas
basic1_1 = basic1[['base_attack', 'base_defense', 'base_stamina']]

# Eliminar filas con valores nulos
basic1_1 = basic1_1.dropna()

# Verificar que no haya valores faltantes
basic1_1.isnull().sum()
4. Definir el número de clusters
Para definir el número de clusters, aplicamos el método del "Codo de Jambu" usando Yellowbrick:

python
Copy code
from yellowbrick.cluster import kelbow_visualizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Escalar los datos
scaler = StandardScaler()
basic1_scaled = scaler.fit_transform(basic1_1)

# Visualizar el Codo de Jambu usando Yellowbrick
kelbow_visualizer(KMeans(random_state=42), basic1_scaled, k=(2, 10))
Este gráfico nos permitirá determinar el número óptimo de clusters observando el punto donde la curva de WCSS comienza a aplanarse.

5. Entrenamiento del modelo
Una vez que hemos determinado el número óptimo de clusters (suponiendo que el número óptimo sea 4), entrenamos el modelo K-Means:

python
Copy code
# Aplicar K-Means con el número de clusters óptimo
optimal_k = 4  # Cambia este valor según el gráfico de Codo de Jambu
kmeans = KMeans(n_clusters=optimal_k, max_iter=300, init='k-means++', random_state=42, n_init='auto')

# Entrenar el modelo
basic1_1['cluster'] = kmeans.fit_predict(basic1_scaled)
6. Evaluación del modelo
Evaluamos el modelo analizando las agrupaciones y observando los resultados visualmente:

python
Copy code
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Función para visualizar los clusters
def visualize_clusters(dataset, c_num, colors, name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0, c_num):
        ax.scatter(dataset['base_attack'][dataset['cluster'] == i],
                   dataset['base_defense'][dataset['cluster'] == i],
                   dataset['base_stamina'][dataset['cluster'] == i],
                   s=100, c=colors[i], label=f'Cluster {i+1}')

    ax.set_title(f'Clusters of {name} Dataset ({c_num} clusters)')
    ax.set_xlabel('Base Attack')
    ax.set_ylabel('Base Defense')
    ax.set_zlabel('Base Stamina')
    ax.legend()
    plt.show()

# Visualizar los clusters
visualize_clusters(basic1_1, optimal_k, ['red', 'blue', 'cyan', 'magenta'], 'Pokemon')
7. Generar un reporte del resultado de su modelo
El reporte final incluiría:

Número óptimo de clusters: Determinado a partir del Codo de Jambu.
Visualización de los clusters: Utilizando un gráfico 3D, como el generado anteriormente.
Distribución de Pokémon por clusters: Puedes obtener el conteo de Pokémon en cada cluster:
python
Copy code
cluster_distribution = basic1_1['cluster'].value_counts()
print(cluster_distribution)
Interpretación de clusters: Analiza qué tipo de Pokémon (con base en sus atributos de ataque, defensa y resistencia) están agrupados en cada cluster. Esto puede ayudar a identificar patrones, como si un cluster tiende a tener Pokémon con alto ataque o gran resistencia.
Este flujo de trabajo te permitirá aplicar K-Means a un dataset con variables numéricas, limpiar los datos, definir el número de clusters, entrenar el modelo, y evaluar los resultados mediante visualización y análisis de distribución.
