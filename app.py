from flask import Flask, render_template, request
from ultralytics import YOLO
import os, cv2
from PIL import Image

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

    # Guardar imagen original
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # Obtener modelo elegido
    model_choice = request.form.get('model_choice', 'best.pt')

    # Cargar modelo dinámicamente
    model_path = model_choice if os.path.exists(model_choice) else "yolo11n.pt"
    model = YOLO(model_path)

    # Inferencia
    results = model.predict(source=img_path, conf=0.4, device='cpu')

    # Guardar resultado visual
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
        output_path = os.path.join('static', 'result_' + file.filename)
        im.save(output_path)

    del model  # liberar memoria

    return render_template(
        'index.html',
        user_image=output_path,
        original_image=img_path,
        selected_model=model_choice
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




