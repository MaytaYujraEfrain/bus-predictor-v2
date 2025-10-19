# Importaciones necesarias para FastAPI, pydantic, pandas y el modelo
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuración de FastAPI ---
app = FastAPI(
    title="MTA Bus Delay Predictor API",
    description="API para predecir el retraso de autobuses del MTA (Nueva York) usando LightGBM.",
    version="1.0.1"
)

# Configuración de CORS para permitir peticiones desde cualquier origen (necesario para la UI local)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rutas de Archivos para Docker ---
# Rutas absolutas dentro del contenedor Docker
BASE_DIR = "/app/models"
MODEL_PATH = os.path.join(BASE_DIR, "lgbm_regressor.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# Variables globales para almacenar el modelo y los encoders
model = None
encoders = None

# --- Función de Carga de Artefactos ---
def load_artifacts():
    global model, encoders
    try:
        # Cargar el modelo LightGBM
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Cargar los encoders (LabelEncoders para variables categóricas)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        
        # Este mensaje NO APARECIÓ en los logs anteriores, lo cual indica el fallo.
        print("✅ Modelo y Encoders cargados con éxito.")

    except FileNotFoundError as e:
        print(f"❌ Error al cargar artefactos: No se encontró el archivo {e.filename}. Verifique la ruta en Dockerfile.")
        # Se lanza una excepción para que el servidor falle si los modelos no cargan
        raise RuntimeError(f"Fallo al cargar archivos: {e}")
    except Exception as e:
        print(f"❌ Error inesperado durante la carga de artefactos: {e}")
        raise RuntimeError(f"Error inesperado: {e}")

# Ejecutar la carga al inicio de la aplicación
load_artifacts()

# --- Esquema de Solicitud (Input) ---
# Definición de la estructura de datos que espera el modelo
class PredictionRequest(BaseModel):
    # Usamos Field para añadir validaciones y ejemplos útiles para la documentación de la API
    Day_of_Week: str = Field(..., example="Monday", description="Día de la semana de la predicción.")
    Time_of_Day: int = Field(..., example=8, description="Hora del día (0-23).")
    Route: str = Field(..., example="B1", description="Ruta del autobús (e.g., B1, B6, B44).")
    Direction: str = Field(..., example="NORTH", description="Dirección del viaje (NORTH, SOUTH, EAST, WEST).")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Day_of_Week": "Monday",
                    "Time_of_Day": 8,
                    "Route": "B44",
                    "Direction": "NORTH"
                }
            ]
        }
    }

# --- Endpoint Raíz (Verificación de salud) ---
@app.get("/", summary="Verificación de estado de la API")
def read_root():
    if model is None or encoders is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. El servicio está temporalmente inactivo.")
    # Respuesta modificada para coincidir con la solicitud del usuario
    return {"status": "ok", "message": "API de Predicción de Buses MTA operativa."}

# --- Endpoint de Predicción ---
@app.post("/predict", summary="Realiza una predicción de retraso de autobús (minutos)")
def predict_delay(request: PredictionRequest):
    # 1. Preparar los datos de entrada en un DataFrame
    data = request.model_dump()
    df = pd.DataFrame([data])
    
    # 2. Preprocesamiento (codificación de variables categóricas)
    
    # Las características numéricas (Time_of_Day) se usan directamente.
    
    # Codificar variables categóricas
    try:
        df['Day_of_Week'] = encoders['Day_of_Week'].transform(df['Day_of_Week'])
        df['Route'] = encoders['Route'].transform(df['Route'])
        df['Direction'] = encoders['Direction'].transform(df['Direction'])

        # La columna Time_of_Day ya es numérica, por lo que se mantiene.
        
    except ValueError as e:
        # Esto ocurre si se envía una categoría que el modelo no conoce
        raise HTTPException(
            status_code=400, 
            detail=f"Error de codificación: una o más categorías de entrada no son válidas. Detalle: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la preparación de la predicción: {e}")

    # 3. Ordenar las columnas para que coincidan con el orden de entrenamiento del modelo (CRÍTICO)
    # Se asume que el orden de las features es el mismo que cuando se entrenó el modelo:
    features = ['Day_of_Week', 'Time_of_Day', 'Route', 'Direction']
    df_processed = df[features]
    
    # 4. Realizar la predicción
    try:
        prediction = model.predict(df_processed)[0]
        
        # Asegurarse de que el retraso no sea negativo
        predicted_delay = max(0, float(prediction))
        
        # 5. Devolver el resultado
        return {
            "predicted_delay_minutes": round(predicted_delay, 2),
            "Day_of_Week": data['Day_of_Week'],
            "Time_of_Day": data['Time_of_Day']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción con el modelo: {e}")

# --- Fin del Archivo ---
