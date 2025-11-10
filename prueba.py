import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("best.onnx")
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
net.setInput(dummy)
print("Forma de salida:", net.forward().shape)

