import json
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    feature_values: list

@app.post('/reglineal')  # Cambiado a POST para enviar datos en el cuerpo de la solicitud
async def reglineal_simulacro(item: Item):
    """ Regresión lineal PRUEBA
    """
    # Cargar modelo previamente entrenado desde un archivo
    with open('modelo_reglineal.pkl', 'rb') as file: 
        modelo_reglineal: LinearRegression = pickle.load(file)

    # Obtener valores de características del cuerpo de la solicitud
    feature_values = item.feature_values

    # Realizar predicciones usando el modelo cargado
    y_pred = modelo_reglineal.predict([feature_values])

    # Convertir las predicciones a formato JSON
    result = {'prediction': float(y_pred[0])}

    # Devolver las predicciones en formato JSON como respuesta a la solicitud
    return 'Hellow world'
