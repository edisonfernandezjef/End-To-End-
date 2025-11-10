from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from PIL import Image

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# === CARGA DEL MODELO ONNX ===
model_path = "best.onnx"
print("Cargando modelo desde:", model_path)
net = cv2.dnn.readNetFromONNX(model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("✅ Modelo cargado correctamente")

# === CLASES (ajustá con tus clases reales si las tenés en un .txt) ===
CLASSES = ["objeto"]

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

    # === PROCESAMIENTO ===
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Crear blob (ajustar tamaño según entrenamiento)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    # === INFERENCIA ===
    preds = net.forward()
    preds = np.array(preds)

    # Normalizar forma del array (maneja [1,1,N,85], [1,N,85] o [N,85])
    if preds.ndim == 4:
        preds = preds[0][0]
    elif preds.ndim == 3:
        preds = preds[0]

    rows = preds.shape[0]
    boxes, confidences, class_ids = [], [], []

    image_ratio_h, image_ratio_w = h / 640, w / 640

    for i in range(rows):
        data = preds[i]
        if len(data) >= 6:
            conf = float(data[4])
            if conf > 0.4:  # confianza mínima
                scores = data[5:]
                class_id = int(np.argmax(scores))
                if scores[class_id] > 0.4:
                    cx, cy, bw, bh = data[:4]
                    x = int((cx - bw / 2) * image_ratio_w)
                    y = int((cy - bh / 2) * image_ratio_h)
                    w_box = int(bw * image_ratio_w)
                    h_box = int(bh * image_ratio_h)
                    boxes.append([x, y, w_box, h_box])
                    confidences.append(conf)
                    class_ids.append(class_id)

    # Supresión no máxima
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.5)

    # === DIBUJAR CAJAS ===
    for i in idxs:
        i = int(i)
        x, y, w_box, h_box = boxes[i]
        label = f"{CLASSES[class_ids[i]] if class_ids[i] < len(CLASSES) else 'obj'} {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Guardar resultado
    result_path = os.path.join('static', 'result_' + file.filename)
    cv2.imwrite(result_path, image)

    return render_template('index.html', user_image=result_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
