from flask import Flask, render_template, request, send_file
from ultralytics import YOLO
import cv2
import os
from PIL import Image

app = Flask(__name__)

# Cargar modelo YOLO (modelo pequeño)
model = YOLO("best.pt")

# Crear carpeta para resultados si no existe
os.makedirs("static", exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No se envió ningún archivo", 400

    file = request.files['file']
    if file.filename == '':
        return "Archivo vacío", 400

    # Guardar imagen temporalmente
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # Ejecutar detección
    results = model.predict(source=img_path, save=False)

    # Obtener la imagen procesada (con bounding boxes)
    for r in results:
        im_array = r.plot()  # dibuja las cajas en un numpy array (BGR)
        im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        output_path = os.path.join('static', 'result_' + file.filename)
        im.save(output_path)

    return render_template('index.html', user_image=output_path)

if __name__ == "__main__":
    app.run(debug=True)
