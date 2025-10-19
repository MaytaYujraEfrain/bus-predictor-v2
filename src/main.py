import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional

# --- 1. Inicialización de la Aplicación ---
app = FastAPI(
    title="MTA Bus Predictor API",
    description="Predice el retraso del bus en minutos utilizando un modelo LightGBM.",
    version="1.0.0"
)

# Configuración de CORS
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Carga del Modelo y Preprocesador ---
# Rutas corregidas (ajusta si es necesario)
MODEL_PATH = 'models/lgbm_model.pkl' # Asegúrate que tu modelo se llame lgbm_model.pkl
PREPROCESSOR_PATH = 'models/preprocessor.pkl' # Asegúrate que tu preprocesador se llame preprocessor.pkl

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model_status = "OK"
except Exception as e:
    model = None
    preprocessor = None
    model_status = f"ERROR: No se pudo cargar el modelo o preprocesador. {e}. Asegúrate de que los archivos estén en la ruta 'models/'."

# --- 3. Esquema de Datos (Schema) - ¡CORREGIDO CON 10 CAMPOS! ---

class PredictionData(BaseModel):
    # CAMPOS ORIGINALES (4)
    route_id: str
    day_of_week: str
    hour: int 
    distance_travelled_meters: float

    # CAMPOS ADICIONALES REQUERIDOS POR EL MODELO (6)
    month: str # String de fecha "YYYY-MM-DD"
    borough: str
    trip_type: str
    period: str
    number_of_customers: int
    additional_bus_stop_time: float
    
    # Validaciones opcionales (pero recomendadas)
    @field_validator('hour')
    @classmethod
    def check_hour(cls, v: int) -> int:
        if not (0 <= v <= 23):
            raise ValueError('hour debe estar entre 0 y 23')
        return v

# --- 4. Esquema de Salida (Schema) - ¡CORREGIDO PARA EL FRONTEND! ---
class PredictionOutput(BaseModel):
    # La clave que el frontend espera
    prediccion_minutos: float
    variable_objetivo: str = "additional_travel_time"
    descripcion: str = "Tiempo de viaje adicional esperado en minutos (retraso)."

# --- 5. Endpoints de la API ---

@app.get("/")
async def read_root():
    """Endpoint raíz para verificar el estado de la API."""
    return {
        "message": "API de Predicción de Retrasos MTA está funcionando.",
        "status": "OK",
        "model_load_status": model_status,
        "predict_endpoint": "/predecir (POST)" # Actualizado
    }

# Endpoint corregido a /predecir para coincidir con el frontend
@app.post("/predecir", response_model=PredictionOutput) 
async def predict_delay(data: PredictionData):
    """
    Realiza la predicción del retraso del bus en minutos utilizando los 10 campos de entrada.
    """
    if model_status.startswith("ERROR"):
        raise HTTPException(status_code=503, detail=model_status)

    try:
        # 1. Convertir los datos de entrada a un DataFrame
        # data.model_dump() es el método moderno de Pydantic
        input_data = pd.DataFrame([data.model_dump()])

        # 2. Convertir el campo 'month' (string de fecha) a un objeto datetime si el preprocesador lo necesita
        # Si tu preprocesador lo maneja, esta línea puede no ser necesaria, pero es más seguro:
        input_data['month'] = pd.to_datetime(input_data['month'])
        
        # 3. Aplicar el preprocesador (incluye codificación one-hot, etc.)
        processed_data = preprocessor.transform(input_data)
        
        # 4. Realizar la predicción
        prediction = model.predict(processed_data)[0]

        # 5. Asegurar que el retraso predicho no es negativo
        predicted_delay = max(0, prediction)

        # 6. Devolver el resultado en el formato JSON que espera el frontend
        return PredictionOutput(prediccion_minutos=float(predicted_delay))

    except Exception as e:
        # Esto captura errores durante la predicción
        print(f"Error en la predicción del servidor: {e}")
        raise HTTPException(status_code=500, detail=f"Error durante la predicción. Detalle: {str(e)}")

# --- 5. Endpoint de Verificación (Opcional, pero recomendado) ---
@app.get("/")
def read_root():
    return {"message": "API de Predicción de Retrasos MTA está funcionando.",
            "status": "OK",
            "predict_endpoint": "/predict (POST)"}
