FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Instalar dependencias esenciales y livianas
RUN pip install --no-cache-dir flask gunicorn pillow opencv-python-headless

# Instalar torch CPU y Ultralytics versión compatible
RUN pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics==8.0.158

CMD ["gunicorn", "app:app"]
