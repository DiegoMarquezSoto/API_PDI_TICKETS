# Imagen base con Python 3.11 #
FROM python:3.11-slim

# Variables de entorno #
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Dependencias del sistema que necesitan OpenCV y Tesseract #
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo dentro del contenedor #
WORKDIR /app

# Copiar requirements.txt #
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación #
COPY . .

# Directorios para img procesadas #
RUN mkdir -p imagenes_decodificadas_base64 imagenes_ingresadas_como_archivo_upload

# Exponer el puerto 8000 #
EXPOSE 8000

# Iniciar la aplicación #
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]