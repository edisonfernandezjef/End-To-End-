# Imagen base liviana con Python 3.10
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app
COPY . /app

# Instalar librerías del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# --- INSTALACIÓN DE DEPENDENCIAS PYTHON ---

# 1️⃣ Fijamos NumPy < 2.0 para compatibilidad con Torch y Ultralytics
RUN pip install --no-cache-dir numpy==1.26.4

# 2️⃣ Dependencias base de Flask, Gunicorn y OpenCV (headless para servidores)
RUN pip install --no-cache-dir flask gunicorn pillow opencv-python-headless

# 3️⃣ Torch y torchvision (solo CPU, sin CUDA)
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# 4️⃣ Ultralytics (versión estable compatible)
RUN pip install --no-cache-dir ultralytics==8.0.158

# Exponer el puerto para Railway
EXPOSE 8080

# Comando para iniciar la aplicación Flask con Gunicorn
CMD ["gunicorn", "app:app"]
