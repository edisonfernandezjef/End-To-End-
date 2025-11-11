# Imagen base liviana con Python 3.10
FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# --- INSTALACIÓN DE DEPENDENCIAS PYTHON ---

# 1️⃣ Instalar dependencias base
RUN pip install --no-cache-dir flask gunicorn pillow opencv-python-headless

# 2️⃣ Instalar Torch (CPU) y Ultralytics
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics==8.0.158

# 3️⃣ 🔒 Reinstalar NumPy 1.26.4 al final para asegurar compatibilidad
RUN pip install --force-reinstall --no-cache-dir numpy==1.26.4

# Exponer puerto para Railway
EXPOSE 8080

# Comando de inicio
CMD ["gunicorn", "app:app"]

