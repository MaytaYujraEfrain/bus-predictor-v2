import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os

# --- 1. Configuración de FastAPI ---
app = FastAPI(
    title="Predictor de Retrasos de Autobuses MTA",
    description="API para estimar el tiempo de viaje adicional (retraso) utilizando un modelo LightGBM.",
    version="1.0.0"
)

# Configuración de CORS para permitir peticiones desde cualquier origen (necesario para el frontend local)
origins = ["*"] # Deberías restringir esto a tu frontend URL en un entorno de producción

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Carga del Modelo y Preprocesador ---
# Verifica que los archivos existan antes de cargarlos
MODEL_PATH = 'lgbm_model.pkl' # Asegúrate de que el nombre del archivo sea correcto
PREPROCESSOR_PATH = 'preprocessor.pkl' # Asegúrate de que el nombre del archivo sea correcto

try:
    # Intenta cargar el modelo entrenado
    lgbm_model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo LightGBM cargado desde {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: El archivo del modelo no se encontró en {MODEL_PATH}")
    # Usar un modelo dummy para que la API pueda iniciar (pero las predicciones serán falsas)
    lgbm_model = None
except Exception as e:
    print(f"❌ ERROR al cargar el modelo: {e}")
    lgbm_model = None

try:
    # Intenta cargar el ColumnTransformer usado para codificar datos
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"✅ Preprocesador cargado desde {PREPROCESSOR_PATH}")
except FileNotFoundError:
    print(f"❌ ERROR: El archivo del preprocesador no se encontró en {PREPROCESSOR_PATH}")
    preprocessor = None
except Exception as e:
    print(f"❌ ERROR al cargar el preprocesador: {e}")
    preprocessor = None


# --- 3. Definición del Esquema de Entrada (Pydantic) ---
# Este esquema DEBE coincidir con los datos que envía el frontend corregido
class PredictionRequest(BaseModel):
    Day_of_Week: str
    Time_of_Day: int # 0-23
    Route: str
    Direction: str # 'NORTH', 'SOUTH', 'EAST', 'WEST'

# --- 4. Endpoint de Predicción ---
@app.post("/predict", summary="Realiza una predicción de retraso de autobús (minutos)")
def predict_delay(request: PredictionRequest):
    """
    Toma los parámetros de la ruta y predice el tiempo de viaje adicional (retraso).
    """
    if lgbm_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="El modelo o el preprocesador no se han cargado correctamente en el servidor.")

    try:
        # Convertir la solicitud Pydantic a un DataFrame de pandas
        input_data = pd.DataFrame([request.dict()])
        
        # El preprocesador solo debe aplicarse a las columnas esperadas
        # Asegúrate de que los nombres de las columnas coincidan exactamente con las usadas en el entrenamiento
        input_data = input_data[['Day_of_Week', 'Time_of_Day', 'Route', 'Direction']]

        # Aplicar el transformador de columnas
        processed_data = preprocessor.transform(input_data)
        
        # Realizar la predicción
        prediction = lgbm_model.predict(processed_data)[0]
        
        # Asegurarse de que el retraso predicho no sea negativo (aunque LightGBM lo maneja bien)
        predicted_delay = max(0, prediction)

        # La clave de respuesta debe ser 'predicted_delay_minutes' para que el frontend funcione
        return {"predicted_delay_minutes": float(predicted_delay)}

    except Exception as e:
        # Esto captura errores durante el procesamiento o la predicción
        print(f"Error durante la predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor durante la predicción. Detalle: {e}")

# --- 5. Endpoint de Verificación (Opcional, pero recomendado) ---
@app.get("/")
def read_root():
    return {"message": "API de Predicción de Retrasos MTA está funcionando.",
            "status": "OK",
            "predict_endpoint": "/predict (POST)"}
