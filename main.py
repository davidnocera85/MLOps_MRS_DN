from fastapi import FastAPI, Request
import zipfile
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import os
import uvicorn


app = FastAPI()

# Ruta del archivo .zip
ruta_archivo_zip = "movies_API.zip"
# Nombre del archivo CSV dentro del .zip
nombre_archivo_csv = "movies_API.csv"

# Abrir el archivo .zip en modo lectura
with zipfile.ZipFile(ruta_archivo_zip, 'r') as archivo_zip:
    # Leer el archivo CSV contenido en el .zip
    with archivo_zip.open(nombre_archivo_csv) as archivo_csv:
        # Leer el archivo CSV con pandas
        df = pd.read_csv(archivo_csv)

# Función para obtener la cantidad de peliculas en un mes específico
@app.get('/cantidad_filmaciones_mes')
async def cantidad_filmaciones_mes(mes: str):
    count = df.loc[df["release_month"].str.contains(mes, case=False)].shape[0]
    return f"{count} películas que fueron estrenadas en el mes de {mes}."

# Función para obtener la cantidad de peliculas en un día específico
@app.get('/cantidad_filmaciones_dia')
async def cantidad_filmaciones_dia(dia: str):
    df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d', errors='coerce')
    def contar_peliculas_por_dia(df, dia):
        dia_esp = {'lunes': 'Monday', 'martes': 'Tuesday','miércoles': 'Wednesday',
            'jueves': 'Thursday','viernes': 'Friday','sábado': 'Saturday','domingo': 'Sunday' }
        day = dia_esp.get(dia.lower(), None)
        if day:
            count = df.loc[df['release_date'].dt.day_name().str.contains(day, case=False)].shape[0]
            return f"{count} películas fueron estrenadas en día {dia}."
        elif dia != day:
            return f"{dia} no es valido, por favor reingrese valor."
        else:
            return 0
    counting = contar_peliculas_por_dia(df, dia)
    return counting.lower().capitalize()

# Función para obtener el título, año de estreno y popularidad de una pelicula
@app.get('/score_titulo')
async def score_titulo(titulo: str):
    filmacion = df[df['title'].str.contains(titulo, case=False)].iloc[0]
    return f"La película {filmacion['title']} fue estrenada en el año {filmacion['release_year']} con indice de popularidad de {filmacion['popularity'].round(2)}."

# Función para obtener el título, cantidad de votos y valor promedio de votaciones de una filmación
@app.get('/votos_titulo')
async def votos_titulo(titulo: str):
    filmacion = df[df['title'].str.contains(titulo, case=False)].iloc[0]
    votos = filmacion['vote_count']
    promedio = filmacion['vote_average']
    if votos < 2000:
        return "La película no cumple con la cantidad mínima de 2000 votos."
    else:
        return (f"La película {filmacion['title']} fue estrenada en el año {filmacion['release_year']}. La misma cuenta con un total de {(votos).astype(int)} votos, con una valoración promedio de {promedio.round(2)}.")

# Función para obtener el éxito de un actor, cantidad de películas y promedio de retorno
@app.get('/get_actor')
async def get_actor(nombre_actor: str):
    cast_list = df['cast_name'].explode()
    cast_list = cast_list.str.strip("[]").str.replace("'", "")
    cast_list = cast_list.str.split(", ")
    individual_cast = [actor for sublist in cast_list for actor in sublist]
    peliculas = pd.Series(individual_cast).value_counts().reset_index()
    peliculas.columns = ['actor', 'peliculas']
    peliculas_actor = peliculas[peliculas['actor'].str.contains(nombre_actor, case=False)]['peliculas'].values[0]
    promedio_retorno = df[df['cast_name'].str.contains(nombre_actor, case=False)]['return'].mean()
    retorno_medio_pelicula = round(promedio_retorno / peliculas_actor, 2)
    return f"{(nombre_actor).title()} ha participado en {peliculas_actor} películas, consiguiendo un retorno total de {promedio_retorno.round(1)}, con un retorno promedio de {retorno_medio_pelicula} por película."
    
# Función para obtener el éxito de un director y detalles de sus películas
@app.get('/get_director')
async def get_director(nombre_director: str):
    director_list = df['director_name'].explode()
    director_list = director_list.str.strip("[]").str.replace("'", "")
    director_list = director_list.str.split(", ")
    individual_director = [director for sublist in director_list if isinstance(sublist, list) for director in sublist]
    Listado_peliculas = pd.Series(individual_director).value_counts().reset_index()
    Listado_peliculas.columns = ['director', 'Listado_peliculas']
    pelis_director = Listado_peliculas[Listado_peliculas['director'].str.contains(nombre_director, case=False)]['Listado_peliculas'].values[0]
    promedio_retorno = df[df['director_name'].str.contains(nombre_director, case=False)]['return'].mean()
    lista_pelis = df[df['director_name'].str.contains(nombre_director, case=False)][['title', 'release_date', 'return', 'budget', 'revenue']].to_dict(orient='records')

    respuesta = {
        'Director': nombre_director,
        'Retorno Total Promedio': promedio_retorno.round(1),
        'Cantidad de películas dirigidas': pelis_director,
        'Lista de películas': []
    }

    for pelicula in lista_pelis:
        respuesta['Lista de películas'].append({
            'Título': pelicula['title'],
            'Año': pelicula['release_date'].strftime('%Y-%m-%d'),
            'Retorno': pelicula['return'],
            'Budget': pelicula['budget'],
            'Revenue': pelicula['revenue']
        })

    return respuesta

# Funcion para ingresar titulo de pelicula y recibir 5 peliculas recomendadas
@app.get('/recomendacion')
async def recomendacion(titulo: str):
    ruta_archivo_zip = "movies_MLOps.zip"
    nombre_archivo_csv = "movies_MLOps.csv"
    with zipfile.ZipFile(ruta_archivo_zip, 'r') as archivo_zip:
        with archivo_zip.open(nombre_archivo_csv) as archivo_csv:
            df = pd.read_csv(archivo_csv)

    keep_cols = ['title', 'release_year', 'budget', 'revenue', 'return',
                 'popularity', 'vote_count', 'vote_average', 'runtime',
                 'cast_name', 'director_name', 'day_of_week']
    df_filtered = df.loc[:, keep_cols]

    numeric_pipe = Pipeline([('scaler', StandardScaler())])
    categorical_pipe = Pipeline([('encoder', OneHotEncoder(drop='first'))])
    col_transf = ColumnTransformer([
        ('numeric', numeric_pipe, ['budget', 'revenue', 'return', 'popularity',
                                   'vote_count', 'vote_average', 'runtime']),
        ('categorical', categorical_pipe, ['title', 'cast_name', 'director_name', 'day_of_week'])
    ])
    df_transformed = col_transf.fit_transform(df_filtered)

    n_neighbors = 6
    knneighbors = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knneighbors.fit(df_transformed)

    titulo_indices = df[df['title'] == titulo].index
    if len(titulo_indices) == 0:
        return {'mensaje': 'No se encontró la película especificada'}

    titulo_indice = titulo_indices[0]
    distancias, indices_vecinos = knneighbors.kneighbors(df_transformed[titulo_indice, :].reshape(1, -1))

    peliculas_recomendadas = []
    for indice_vecino in indices_vecinos[0]:
        if df['title'].iloc[indice_vecino] != titulo:
            peliculas_recomendadas.append(df['title'].iloc[indice_vecino])

    return f"Lista recomendada de 5 películas: {peliculas_recomendadas}"
