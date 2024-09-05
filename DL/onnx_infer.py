import onnxruntime
import cv2
import numpy as np
from PIL import Image

ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

x = np.float32(cv2.resize(cv2.imread('../API/static/images/demo1.png'), (256, 256))) / 255.0
x = np.expand_dims(np.transpose(x, (2, 1, 0)), axis=0)

# ONNX 런타임에서 계산된 결과값
ort_inputs = {ort_session.get_inputs()[0].name: x}
ort_outs = ort_session.run(None, ort_inputs)

im = Image.fromarray(np.uint8(np.squeeze(ort_outs) * 255))
im.save("your_file.png")

print(ort_outs)