FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Dependencias Python base
RUN pip install --no-cache-dir flask gunicorn pillow opencv-python-headless

# Torch CPU + torchvision (ligero)
RUN pip install --no-cache-dir torch==2.3.0+cpu torchvision==0.18.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# 🚀 Ultralytics actualizado (compatible con YOLOv11)
RUN pip install --no-cache-dir ultralytics==8.3.227

# Fijar NumPy < 2.0 por compatibilidad
RUN pip install --force-reinstall --no-cache-dir numpy==1.26.4

EXPOSE 8080
CMD ["gunicorn", "app:app"]

