from ultralytics import YOLO

# Ruta absoluta (ajustala a tu PC)
model = YOLO(r"C:\Users\yazed\OneDrive\Documentos\Istea\Segundo Año\Segundo Cuatrimestre\Aprendizaje Automatico II\Proyecto End To End\pruebas\best.pt")

model.export(format="onnx", opset=12, simplify=True, dynamic=False)

