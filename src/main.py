import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- 1. Inicialización de la Aplicación ---
app = FastAPI(
    title="MTA Bus Predictor API",
    description="Predice el retraso del bus en minutos utilizando un modelo LightGBM.",
    version="1.0.0"
)

# Configuración de CORS para permitir solicitudes desde tu frontend (o cualquier origen)
# Se asume que el frontend corre localmente o desde cualquier host
origins = ["*"] # Esto permite cualquier origen, lo cual es útil para desarrollo

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Carga del Modelo y Preprocesador ---
# Rutas corregidas para apuntar a la carpeta models/ dentro del contenedor
MODEL_PATH = 'models/lgbm_model.pkl' 
PREPROCESSOR_PATH = 'models/preprocessor.pkl' 

try:
    # Intenta cargar el modelo entrenado
    model = joblib.load(MODEL_PATH)
    # Intenta cargar el transformador de columnas (preprocessor)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model_status = "OK"
except Exception as e:
    model = None
    preprocessor = None
    model_status = f"ERROR: No se pudo cargar el modelo o preprocesador. {e}"

# --- 3. Esquema de Datos (Schema) ---
class PredictionData(BaseModel):
    # Los nombres de los campos DEBEN coincidir con los que espera el modelo (y tu frontend)
    route_id: str
    day_of_week: str
    hour: int # De 0 a 23
    distance_travelled_meters: float

# --- 4. Endpoints de la API ---

@app.get("/")
async def read_root():
    """Endpoint raíz para verificar el estado de la API."""
    return {
        "message": "API de Predicción de Retrasos MTA está funcionando.",
        "status": "OK",
        "model_load_status": model_status,
        "predict_endpoint": "/predict (POST)"
    }

@app.post("/predict")
async def predict_delay(data: PredictionData):
    """
    Realiza la predicción del retraso del bus en minutos.
    Acepta datos en formato JSON y devuelve el retraso predicho.
    """
    if model_status.startswith("ERROR"):
        raise HTTPException(status_code=503, detail=model_status)

    try:
        # Convertir los datos de entrada a un DataFrame
        input_data = pd.DataFrame([data.model_dump()])

        # Aplicar el preprocesador (incluye codificación one-hot, etc.)
        processed_data = preprocessor.transform(input_data)
        
        # Realizar la predicción
        prediction = model.predict(processed_data)[0]

        # Asegurar que el retraso predicho no es negativo
        predicted_delay = max(0, prediction)

        # Devolver el resultado en el formato JSON que espera el frontend
        return {
            "predicted_delay_minutes": float(predicted_delay)
        }

    except Exception as e:
        # Esto captura errores durante la predicción, como problemas con el transformador.
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")

# --- 5. Endpoint de Verificación (Opcional, pero recomendado) ---
@app.get("/")
def read_root():
    return {"message": "API de Predicción de Retrasos MTA está funcionando.",
            "status": "OK",
            "predict_endpoint": "/predict (POST)"}
