
# <h1 align=center> **PROYECTO INDIVIDUAL** </h1>
# <h1 align=center>*"Movies Recommendation System"* </h1>

<p align="center">
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTd6mQOzPB54ZgMqZKANAZSTDdtOxnQl4kOB6p7NluOeIwC9azUdUy_p3CPQUPxc_NWgfU&usqp=CAU"  height=100>
</p>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

### Proyecto: Desarrollar un Sistema de Recomendación de Películas **`MVP`** (_Minimum Viable Product_), empleando modelado en Machine Learning, usando el framework ***FastAPI*** y ***Render*** como 'deployment', realizando además previamente un trabajo rápido de **`Data Engineer`** (ETL + EDA sobre los datasets originales).


**`Requerimientos`**:

Desarrollo en código Python (Visual Studio Code y Google Colab).
Bibliotecas: FastAPI, uvicorn, pandas, numpy, graphviz, Matplotlib, Scipy, missingno, optuna, requests, Scikit-learn, Seaborn, Scipy, WordCloud. 
Módulos: zipfile, datetime, calendar, re, ast, pydantic, typing, os, entre los mas relevantes.

En el siguiente enlace se accede a la carpeta del entorno virtual para poder implementar el código:

(https://drive.google.com/drive/folders/1FOMVg0mJJUXdo_mBdFtcDGklLA3bhpKa?usp=drive_link)


**`Estructura de carpetas y archivos`**:

1. **Datasets**: carpeta que contiene los datasets del proyecto.
2. **EDA**: carpeta que contiene 'notebook' y gráficos del proceso 'Exploratory Data Analysis' de datos.
3. **ETL**: carpeta que contiene 'notebook' sobre el procesamiento y transformaciones necesarias de los datasets originales.
4. **MLOps**: carpeta que contiene 'notebook' sobre el modelado de 'Machine Learning'.
5. **FastAPI**: carpeta que contiene 'notebook' y subcarpetas para el desarrollo de la API.
6. *main.py*: archivo de Python que gestiona la aplicación del modelo.
7. *movies_API.csv*: archivo comprimido que contiene dataset para API (Consultas 1 a 6).
8. *movies_MLOps.csv*: archivo comprimido que contiene dataset para API (Consulta 7 - Recomendación).
9. *requirements.txt*: este archivo almacena los recursos elementales del proyecto.
10. *README - DN*: este archivo propiamente dicho, que contiene guia orientadora sobre el proyecto.


**`Instructivo para el uso del Sistema de Consultas`**:

1. Acceder a la API mediante el siguiente enlace:

(https://movie-reccommendation-system-dan.onrender.com/docs)

2. Ejecutar las 7 consultas disponibles:

**/cantidad_filmaciones_mes/{mes}**: se ingresa un mes (en español) y devuelve la cantidad de películas que se estrenaron históricamente ese mes del año. Por ejemplo: Enero, etc.

**/cantidad_filmaciones_dia/{dia}**: se ingresa un día de la semana (en español) y devuelve la cantidad de películas que se estrenaron históricamente en ese día. Por ejemplo: Lunes, etc.

**/score_titulo/{titulo}**: se ingresa el título de una película y devuelve como respuesta el título, el año de estreno y el score (índice de popularidad). Por ejemplo: "Jumanji".

**/votos_titulo/{titulo}**: se ingresa el título de una película y devuelve como respuesta el título, la cantidad de votos (valor absoluto) y el valor promedio de las votaciones, si la película consultada cuenta con al menos 2000 valoraciones. Por ejemplo: "Toy Story".

**/get_actor/{nombre_actor}**: se ingresa el nombre de un actor y devuelve como respuesta la cantidad de películas en las que ha participado, incluyendo su éxito (en función del "retorno") y el promedio de retorno por película. Por ejemplo: "Tom Cruise".

**/get_director/{nombre_director}**: se ingresa el nombre de un director y devuelve su éxito (en función del "retorno"), incluyendo la cantidad de películas que ha dirigido. A su vez, devuelve un listado de sus películas, que incluye: Título, Fecha de estreno, Retorno individual, Costo y Ganancia de cada una. Por ejemplo: "John Lasseter".

**/recomendacion/{titulo}**: se ingresa el título de una película de interés, y devuelve un listado de 5 películas recomendadas (ordenadas en forma descendiente en función del "score de similitud"). Por ejemplo: "Cars". 


**`Sistema de recomendación`**

Se empleó un dataset previamente procesado y analizado mediante ETL - EDA cuyas dimensiones se redujeron a 12 campos y 5271 registros. Preparación de datos para un modelo de aprendizaje automático utilizando el algoritmo K-Nearest Neighbors (KNN). Se empleó StandardScaler para estandarizar las características numéricas y asegurarse de que todas tengan un rango similar, y OneHotEncoder para codificar las variables categóricas en una representación numérica binaria. A continuación se definió arbitrariamente la variable k en función de las necesidades requeridas para la implementación del sistema de recomendación, empleando la métrica 'coseno' para calcular la similitud entre las muestras. Luego se ajustaron los parámetros y se procedió al testeo del modelo, obteniendose las distancias y los índices de las 5 películas vecinas más cercanas a la película de referencia (score de similitud). A su vez, se analizaron algunas métricas del modelo generado como el Error Cuadrático Medio (MSE), el Error Absoluto Medio (MAE) y el Coeficiente de Determinación (R^2). En general, se busca minimizar el MSE y el MAE, y aproximar a 1 el R^2 para obtener un modelo de mayor calidad y precisión. En este desarrollo, empleando como parámetro "vote_average" se obtuvieron los mejoreses resultados: MSE = 0.03, MAE = 0.12 y R^2 = 0.96. Cabe destacar que si bien el sistema de recomendación es aceptable, mediante un desarrollo mas evolucinado, podría mejorarse considerablemente. 


**`Video de demostración`**

(https://drive.google.com/file/d/1Tof9R3tRtq-GkzDz0GaRzQ9O3K-qzb80/view?usp=sharing)


**`Fuente de datos de la propuesta de proyecto`**

+ [Dataset](https://drive.google.com/drive/folders/1nvSjC2JWUH48o3pb8xlKofi8SNHuNWeu): Carpeta con los 2 archivos con datos que requieren ser procesados (movies_dataset.csv y credits.csv), tengan en cuenta que hay datos que estan anidados (un diccionario o una lista como valores en la fila).
+ [Diccionario de datos](https://docs.google.com/spreadsheets/d/1QkHH5er-74Bpk122tJxy_0D49pJMIwKLurByOfmxzho/edit#gid=0): Diccionario con algunas descripciones de las columnas disponibles en el dataset original.
<br/>
+ [links de ayuda](hhttps://github.com/HX-PRomero/PI_ML_OPS/raw/main/Material%20de%20apoyo.md).


**`Autor`**

David Nocera

davidandresnocera@gmail.com

GitHub: davidnocera85
