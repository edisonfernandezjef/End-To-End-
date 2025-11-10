from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from PIL import Image
from pathlib import Path

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# === CARGA DEL MODELO ONNX ===
model_path = Path(__file__).parent / "best.onnx"
net = cv2.dnn.readNetFromONNX(str(model_path))

# Si querés forzar CPU explícitamente (Render no tiene GPU):
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# === LISTA DE CLASES (ajustá según tu modelo entrenado) ===
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

    # Guardar la imagen temporalmente
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # === PROCESAMIENTO DE LA IMAGEN ===
    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    # Crear blob (preprocesamiento estándar YOLO)
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    # === INFERENCIA ===
    preds = net.forward()

    # === POSTPROCESAMIENTO (detecciones) ===
    boxes, confidences, class_ids = [], [], []
    rows = preds[0].shape[1]  # filas de detecciones
    image_ratio_h, image_ratio_w = h / 640, w / 640

    for i in range(rows):
        data = preds[0][0][i]
        confidence = data[4]
        if confidence > 0.4:  # umbral de confianza
            scores = data[5:]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.4:
                cx, cy, bw, bh = data[0:4]
                x = int((cx - bw / 2) * image_ratio_w)
                y = int((cy - bh / 2) * image_ratio_h)
                w_box = int(bw * image_ratio_w)
                h_box = int(bh * image_ratio_h)
                boxes.append([x, y, w_box, h_box])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Supresión no máxima (NMS)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.5)

    # === DIBUJAR CAJAS ===
    for i in idxs:
        i = int(i)
        x, y, w_box, h_box = boxes[i]
        label = f"{CLASSES[class_ids[i]] if class_ids[i] < len(CLASSES) else 'obj'} {confidences[i]:.2f}"
        cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Guardar resultado
    result_path = os.path.join('static', 'result_' + file.filename)
    cv2.imwrite(result_path, image)

    return render_template('index.html', user_image=result_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
