# ====================================================================
# src/main.py
# API REST con FastAPI para la predicción de retrasos de bus (MLOps)
# ====================================================================
import uvicorn
import pandas as pd
import pickle
import os
import numpy as np # Necesario para manejar tipos
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 

# --- Configuración de Rutas y Variables ---
MODEL_PATH = "models/lgbm_regressor.pkl"
ENCODERS_PATH = "models/encoders.pkl"
CARACTERISTICAS_CATEGORICAS = ['borough', 'trip_type', 'route_id', 'period']
MODELO_CARGADO = None
ENCODERS_CARGADOS = None

# --- Definición del Esquema de Entrada (Pydantic) ---
class BusFeatures(BaseModel):
    # month es la única fecha, se usará para extraer month_num
    month: str = Field(..., example="2024-01-01")
    borough: str = Field(..., example="Bronx")
    trip_type: str = Field(..., example="LCL/LTD")
    route_id: str = Field(..., example="BX1")
    period: str = Field(..., example="Off-Peak")
    number_of_customers: float = Field(..., example=286804.2)
    additional_bus_stop_time: float = Field(..., example=1.8716421)

# --- Inicialización de la Aplicación y Carga de Modelos ---

app = FastAPI(
    title="MTA Bus Prediction API",
    description="API de MLOps que predice el tiempo de viaje adicional (retraso) en minutos.",
    version="1.0.0"
)

# ====================================================================
# BLOQUE CORS (Necesario para la prueba visual)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
# ====================================================================

def cargar_artefactos():
    """Carga el modelo LightGBM y los encoders."""
    global MODELO_CARGADO, ENCODERS_CARGADOS
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError("Archivos de modelo o encoders no encontrados.")
            
        with open(MODEL_PATH, 'rb') as f:
            MODELO_CARGADO = pickle.load(f)
            
        with open(ENCODERS_PATH, 'rb') as f:
            ENCODERS_CARGADOS = pickle.load(f)
            
        print("✅ Modelo y Encoders cargados con éxito.")
    except Exception as e:
        print(f"❌ Error al cargar artefactos: {e}")
        pass

cargar_artefactos()

# --- Endpoint de Salud (Health Check) ---
@app.get("/", tags=["Health Check"])
def health_check():
    """Verifica si la API está funcionando."""
    return {"status": "ok", "message": "API de Predicción de Buses MTA operativa."}

# --- Endpoint de Predicción ---
@app.post("/predecir", tags=["Predicción"])
def predecir_retraso(features: BusFeatures):
    """
    Realiza una predicción del tiempo de viaje adicional (retraso) en minutos.
    """
    if MODELO_CARGADO is None or ENCODERS_CARGADOS is None:
        return {"error": "Modelo no cargado. Verifique los logs de inicio."}, 500

    try:
        # 1. Convertir datos de entrada a un DataFrame
        data = features.model_dump()
        df = pd.DataFrame([data])
        
        # 2. Ingeniería de Características (Debe coincidir con train_pipeline.py)
        df['month'] = pd.to_datetime(df['month'])
        df['month_num'] = df['month'].dt.month
        
        # 3. Preprocesamiento (Label Encoding)
        for col in CARACTERISTICAS_CATEGORICAS:
            if col in ENCODERS_CARGADOS:
                le = ENCODERS_CARGADOS[col]
                
                # Función para manejar valores no vistos (fallback a 0)
                def transform_with_fallback(value):
                    try:
                        return le.transform([value])[0]
                    except ValueError:
                        return 0 

                df[col] = df[col].apply(transform_with_fallback)
            
        # 4. Preparar el DataFrame final de features
        X = df.drop(columns=['month'])
        
        # 5. Predicción
        prediccion_array = MODELO_CARGADO.predict(X)
        
        # 🔴 CORRECCIÓN: Usar indexación de NumPy ([0]) en lugar de .iloc[0]
        prediccion = prediccion_array[0] 

        return {
            "prediccion_minutos": round(float(prediccion), 4), # Convertir a float estándar
            "variable_objetivo": "additional_travel_time",
            "descripcion": "Tiempo de viaje adicional esperado en minutos (retraso)."
        }

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return {"error": f"Error interno del servidor durante la predicción: '{e}'"}, 500

# --- Bloque de ejecución principal (Uvicorn) ---
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
