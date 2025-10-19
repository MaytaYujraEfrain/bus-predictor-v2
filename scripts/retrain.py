import pandas as pd
import numpy as np
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, ValidationError
from typing import Dict, Any
import pickle
import os

# --- Configuración de Rutas ---
MODEL_PATH = "models/lgbm_regressor.pkl"
ENCODERS_PATH = "models/encoders.pkl"
CARACTERISTICAS_CATEGORICAS = ['borough', 'trip_type', 'route_id', 'period', 'day_of_week']
FEATURE_ORDER = [
    'distance_travelled_meters', 
    'hour', 
    'number_of_customers', 
    'additional_bus_stop_time', 
    'borough', 
    'trip_type', 
    'route_id', 
    'period', 
    'day_of_week',
    'month_num' # El feature creado a partir de 'month'
]

# --- Inicialización del Servidor ---
app = FastAPI(title="MTA Bus Delay Predictor API")

# --- Variables Globales (Modelo y Encoders) ---
model = None
encoders = {}

# --- Funciones de Carga ---

def load_model_assets():
    """Carga el modelo y los encoders necesarios para la predicción."""
    global model, encoders
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError("Assets del modelo no encontrados. Ejecuta 'python scripts/retrain.py' primero.")
            
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        print("Modelos y encoders cargados exitosamente.")
        
    except Exception as e:
        print(f"Error al cargar assets del modelo: {e}")
        # En producción, podrías levantar una excepción para que el servidor falle al iniciar
        # Aquí permitimos que inicie, pero el endpoint fallará si se llama.

# Cargar assets al inicio de la aplicación
load_model_assets()

# --- Esquemas de Datos (Pydantic) ---

class PredictionInput(BaseModel):
    # La validación de tipado de Pydantic
    route_id: str
    day_of_week: str
    hour: int
    distance_travelled_meters: float
    month: str # Lo esperamos como STRING de fecha (ej. "2023-10-01")
    borough: str
    trip_type: str
    period: str
    number_of_customers: int
    additional_bus_stop_time: float

    @field_validator('hour')
    @classmethod
    def check_hour(cls, v: int) -> int:
        if not (0 <= v <= 23):
            raise ValueError('hour debe estar entre 0 y 23')
        return v
    
class PredictionOutput(BaseModel):
    # Este es el formato de salida que tu frontend espera (la clave real)
    prediccion_minutos: float
    variable_objetivo: str = "additional_travel_time"
    descripcion: str = "Tiempo de viaje adicional esperado en minutos (retraso)."

# --- Endpoint de Salud ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de Predicción de Retrasos MTA lista."}

# --- Endpoint de Predicción (¡Aquí está la lógica crítica!) ---

@app.post("/predecir", response_model=PredictionOutput)
def predecir_retraso(data: PredictionInput):
    
    if model is None or not encoders:
        raise HTTPException(status_code=500, detail="Modelo o encoders no cargados.")
    
    try:
        # 1. CONVERTIR INPUT PYDANTIC A DICCIONARIO VARIABLE (¡CORRECCIÓN CLAVE!)
        input_data = data.model_dump() 
        
        # 2. INGENIERÍA DE CARACTERÍSTICAS (¡CORRECCIÓN CLAVE!)
        
        # A. Crear la característica 'month_num' que usa el modelo (ver retrain.py)
        # pd.to_datetime convierte el string de fecha (ej. "2023-10-01") a objeto datetime
        month_obj = pd.to_datetime(input_data['month'])
        input_data['month_num'] = month_obj.month 
        
        # B. Eliminar la característica 'month' ya que el modelo no la necesita
        del input_data['month'] 

        # 3. CREAR DATAFRAME y CODIFICAR
        X_new = pd.DataFrame([input_data])
        
        # Codificación de variables categóricas
        for col in CARACTERISTICAS_CATEGORICAS:
            if col in encoders:
                le = encoders[col]
                # Los nuevos datos deben codificarse con el encoder entrenado
                # Usamos .astype(str) para evitar errores si el valor es numérico pero categorico
                X_new[col] = le.transform(X_new[col].astype(str))
            else:
                # Manejo de error si falta un encoder
                raise ValueError(f"Encoder no encontrado para la columna: {col}")

        # 4. Asegurar el orden de las columnas (¡CRÍTICO para LightGBM!)
        X_final = X_new[FEATURE_ORDER]
        
        # 5. Predicción
        prediction = model.predict(X_final)[0]
        
        # 6. Retorno de Resultado
        return PredictionOutput(prediccion_minutos=float(prediction))

    except Exception as e:
        # Esto capturará errores como la falta de un encoder o un fallo de Pandas/LightGBM
        print(f"Error interno durante la predicción: {e}")
        # Retornamos 500 para indicarle al frontend que es un fallo del servidor
        raise HTTPException(status_code=500, detail=[{"error": f"Error interno del servidor durante la predicción: '{e}'"}])
