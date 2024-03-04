import json  # Importa la librería json para manipulación de datos en formato JSON
import pickle  # Importa la librería pickle para trabajar con serialización y deserialización de objetos en Python
import numpy as np  # Importa la librería NumPy para operaciones numéricas eficientes
from fastapi import FastAPI  # Importa la clase FastAPI para crear una aplicación web con FastAPI
from sklearn.linear_model import LinearRegression  # Importa la clase LinearRegression del módulo sklearn.linear_model

app = FastAPI()  # Crea una instancia de la clase FastAPI para manejar la aplicación web

@app.get('/reglineal')  # Define una ruta '/reglineal' para la aplicación web usando el método HTTP GET
async def reglineal_simulacro():
    """ Regresión lineal PRUEBA
    """
    # Cargar modelo previamente entrenado desde un archivo
    with open('modelo_reglineal.pkl', 'rb') as file: 
        modelo_reglineal: LinearRegression = pickle.load(file)
        
    # Cargar input simulado (X_test) desde un archivo
    with open('X_test.pkl', 'rb') as file: 
        X_test = pickle.load(file)

    # Realizar predicciones usando el modelo cargado
    y_pred: np.array = modelo_reglineal.predict(X_test)

    # Convertir las predicciones a formato JSON
    y_pred_json = json.dumps(y_pred.tolist())

    # Devolver las predicciones en formato JSON como respuesta a la solicitud
    return y_pred_json

'''Este código define una aplicación web con FastAPI (FastAPI()) que tiene una ruta /reglineal que, cuando se accede a través de un navegador web o mediante una solicitud HTTP GET, carga un modelo de regresión lineal previamente entrenado y realiza predicciones sobre un conjunto de datos simulado. Luego, las predicciones se convierten a formato JSON y se devuelven como respuesta a la solicitud. '''
