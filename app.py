from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2, os
from PIL import Image
from pathlib import Path

app = Flask(__name__)
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

    # ✅ Cargar el modelo dentro del endpoint (para liberar RAM)
    model_path = Path(__file__).parent / "best.pt"
    model = YOLO(model_path)

    # Ejecutar predicción
    results = model.predict(source=img_path, save=False, verbose=False)

    # Guardar resultado
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        output_path = os.path.join('static', 'result_' + file.filename)
        im.save(output_path)

    return render_template('index.html', user_image=output_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
