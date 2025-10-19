# ====================================================================
# scripts/retrain.py
# Script automatizado para re-entrenar el modelo y validar su rendimiento
# ====================================================================
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import argparse
import sys # Para manejar la salida

# --- Configuración de Rutas y Variables ---
DATA_PATH = "data/MTA_Bus_Metrics_2024.csv"
MODEL_PATH = "models/lgbm_regressor.pkl"
ENCODERS_PATH = "models/encoders.pkl"
MIN_R2_THRESHOLD = 0.70 # Umbral de calidad mínimo. Si el nuevo R2 es menor, no se promueve.

CARACTERISTICAS_CATEGORICAS = ['borough', 'trip_type', 'route_id', 'period']
TARGET = 'additional_travel_time'


def cargar_datos(path):
    """Carga los datos y realiza la ingeniería de características necesaria."""
    try:
        df = pd.read_csv(path)
        print(f"Datos cargados desde: {path}")
        
        # Ingeniería de Características
        df['month'] = pd.to_datetime(df['month'])
        df['month_num'] = df['month'].dt.month # Feature: Mes como número
        
        # Eliminar NaN y valores extremos para asegurar un entrenamiento limpio
        df = df.dropna(subset=CARACTERISTICAS_CATEGORICAS + [TARGET])
        
        return df
    except Exception as e:
        print(f"❌ Error al cargar o procesar datos: {e}")
        sys.exit(1) # Salir con error

def entrenar_modelo(df):
    """Entrena el modelo LightGBM y retorna el modelo, encoders y métricas."""
    print("Iniciando entrenamiento...")
    
    X = df.drop(columns=[TARGET, 'month', 'customer_journey_time_performance'])
    y = df[TARGET]
    
    encoders = {}
    
    # 1. Codificación de Variables Categóricas
    for col in CARACTERISTICAS_CATEGORICAS:
        le = LabelEncoder()
        # Ajustamos y transformamos en el DataFrame completo (ya que es para entrenamiento)
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Lista final de features (debe coincidir con las features usadas en main.py)
    features_final = [col for col in X.columns if col != 'month']
    X = X[features_final]
    
    # 2. División de datos (Entrenamiento/Prueba)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Inicialización y Entrenamiento del Modelo
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    
    # 4. Evaluación del Modelo
    y_pred = lgbm.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✔️ Entrenamiento finalizado. Métrica R2: {r2:.4f}, RMSE: {rmse:.4f}")
    
    return lgbm, encoders, r2, rmse

def validar_y_guardar(nuevo_modelo, nuevos_encoders, nuevo_r2, nuevo_rmse):
    """Compara el nuevo modelo con el modelo en producción y decide si promoverlo."""
    
    # 1. Intentar cargar la métrica del modelo actual (simulando un registro)
    old_r2 = -float('inf')
    if os.path.exists(MODEL_PATH):
        # Para un proyecto real, se guardaría el R2 en un archivo de texto o MLflow/WandB
        # Aquí, simulamos que el R2 anterior era 0.85
        print("Simulando carga de R2 anterior...")
        old_r2 = 0.85 

    print(f"R2 del Modelo en Producción (simulado): {old_r2:.4f}")
    
    # 2. Decisión de Promoción
    if nuevo_r2 > old_r2 and nuevo_r2 >= MIN_R2_THRESHOLD:
        print(f"✨ ¡Éxito! El nuevo R2 ({nuevo_r2:.4f}) es mejor que el anterior ({old_r2:.4f}) y supera el umbral de {MIN_R2_THRESHOLD}. Promoviendo a producción.")
        
        # Guardar el nuevo modelo y encoders (esto sobrescribe la versión anterior)
        os.makedirs('models', exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(nuevo_modelo, f)
        with open(ENCODERS_PATH, 'wb') as f:
            pickle.dump(nuevos_encoders, f)
            
        print(f"Modelos guardados con éxito en {os.getcwd()}/models/")
        
        # Retornamos el R2 para que GitHub Actions pueda leerlo
        print(f"::set-output name=new_r2::{nuevo_r2:.4f}")
        return True
    
    else:
        print(f"⚠️ El nuevo modelo (R2: {nuevo_r2:.4f}) no es mejor que el modelo en producción o no cumple el umbral. No se realizaron cambios.")
        return False

# --- Ejecución Principal ---
if __name__ == "__main__":
    # Asegúrate de que tienes la carpeta 'data'
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Creando carpeta 'data'. Coloca tu archivo 'MTA_Bus_Metrics_2024.csv' dentro.")
        sys.exit(1)

    df_data = cargar_datos(DATA_PATH)
    
    if df_data is not None and not df_data.empty:
        modelo, encoders, r2, rmse = entrenar_modelo(df_data)
        
        # Simula la lógica de promoción (CM)
        validar_y_guardar(modelo, encoders, r2, rmse)
