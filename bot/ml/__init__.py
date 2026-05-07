import onnxruntime as ort
try:
    ort.InferenceSession("bot/ml/onnx_model/model.onnx")
    print("Model loaded successfully")
except Exception as e:
    print(f"Load failed: {e}")
