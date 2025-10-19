# Dockerfile
# Usamos una imagen base de Python ligera (slim)
FROM python:3.10-slim

# Establecer la carpeta de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de requisitos e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copiar Artefactos y Código ---

# 1. Copiar los scripts de la API
COPY src/ /app/src/

# 2. Copiar el dataset (Necesario si el script de entrenamiento se ejecuta en Docker, aunque aquí se usa solo para referencia)
COPY data/ /app/data/

# 3. Copiar los modelos pre-entrenados y encoders
COPY models/ /app/models/

# --- Exponer Puerto y Comando de Inicio ---

# Mantener el puerto 8000 para consistencia con el entorno local
EXPOSE 8000

# Comando para ejecutar la API con Uvicorn
# Usamos el puerto 8000 por defecto de Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
