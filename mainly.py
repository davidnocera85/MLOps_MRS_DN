from fastapi import FastAPI
import zipfile
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import uvicorn

app = FastAPI()
@app.get('/') #ruta raíz
def get_root():
    return ' API Henry'


# Leer el archivo CSV con pandas
df = pd.read_csv("movies_API.csv")

# Función para obtener la cantidad de películas en un mes específico
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    count = df.loc[df["release_month"].str.contains(mes, case=False)].shape[0]
    return {"mensaje": f"{count} películas que fueron estrenadas en el mes de {mes}."}

# Función para obtener la cantidad de películas en un día específico
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    df['release_date'] = pd.to_datetime(df['release_date'], format='%Y-%m-%d', errors='coerce')
    def contar_peliculas_por_dia(df, dia):
        dia_esp = {'lunes': 'Monday', 'martes': 'Tuesday','miércoles': 'Wednesday',
                   'jueves': 'Thursday','viernes': 'Friday','sábado': 'Saturday','domingo': 'Sunday'}
        day = dia_esp.get(dia.lower(), None)
        if day:
            count = df.loc[df['release_date'].dt.day_name().str.contains(day, case=False)].shape[0]
            return {"mensaje": f"{count} películas fueron estrenadas en día {dia}."}
        elif dia != day:
            return {"mensaje": f"{dia} no es válido, por favor reingrese el valor."}
        else:
            return {"mensaje": "No se encontraron películas para el día especificado."}
    counting = contar_peliculas_por_dia(df, dia)
    return counting

# Función para obtener el título, año de estreno y popularidad de una película
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    filmacion = df[df['title'].str.contains(titulo, case=False)].iloc[0]
    return {"mensaje": f"La película {filmacion['title']} fue estrenada en el año {filmacion['release_year']} con índice de popularidad de {filmacion['popularity'].round(2)}."}

# Función para obtener el título, cantidad de votos y valor promedio de votaciones de una filmación
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    filmacion = df[df['title'].str.contains(titulo, case=False)].iloc[0]
    votos = filmacion['vote_count']
    promedio = filmacion['vote_average']
    if votos < 2000:
        return {"mensaje": "La película no cumple con la cantidad mínima de 2000 votos."}
    else:
        return {"mensaje": f"La película {filmacion['title']} fue estrenada en el año {filmacion['release_year']}. La misma cuenta con un total de {int(votos)} votos, con una valoración promedio de {promedio.round(2)}."}

# Función para obtener el éxito de un actor, cantidad de películas y promedio de retorno
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    cast_list = df['cast_name'].explode()
    cast_list = cast_list.str.strip("[]").str.replace("'", "")
    cast_list = cast_list.str.split(", ")
    individual_cast = [actor for sublist in cast_list for actor in sublist]
    peliculas = pd.Series(individual_cast).value_counts().reset_index()
    peliculas.columns = ['actor', 'peliculas']
    peliculas_actor = peliculas[peliculas['actor'].str.contains(nombre_actor, case=False)]['peliculas'].values[0]
    promedio_retorno = df[df['cast_name'].str.contains(nombre_actor, case=False)]['return'].mean()
    retorno_medio_pelicula = round(promedio_retorno / peliculas_actor, 2)
    return {
        "mensaje": f"{nombre_actor.title()} ha participado en {peliculas_actor} películas, consiguiendo un retorno total de {promedio_retorno.round(1)}, con un retorno promedio de {retorno_medio_pelicula} por película."
    }

# Función para obtener el éxito de un director y detalles de sus películas
@app.get("/get_director/{nombre_director}")                               
def get_director(nombre_director:str):
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

# Función para ingresar el título de una película y recibir 5 películas recomendadas
@app.get("/recomendacion/{nombre_pelicula}")                               
def recomendacion(nombre_pelicula:str):
    df = pd.read_csv("movies_MLOps.csv")

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

    titulo_indices = df[df['title'] == nombre_pelicula].index
    if len(titulo_indices) == 0:
        return {'mensaje': 'No se encontró la película especificada'}

    titulo_indice = titulo_indices[0]
    distancias, indices_vecinos = knneighbors.kneighbors(df_transformed[titulo_indice, :].reshape(1, -1))

    peliculas_recomendadas = []
    for indice_vecino in indices_vecinos[0]:
        if df['title'].iloc[indice_vecino] != nombre_pelicula:
            peliculas_recomendadas.append(df['title'].iloc[indice_vecino])

    return {'peliculas_recomendadas': peliculas_recomendadas}
