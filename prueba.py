import cv2
net = cv2.dnn.readNetFromONNX(r"best.onnx")
print("✅ Modelo cargado correctamente")
