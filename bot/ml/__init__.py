import os
import onnxruntime as ort

try:
    ort.InferenceSession(os.path.dirname(os.path.abspath(__file__)) + "/model.onnx")
    print("Model loaded successfully")
    with open(os.path.dirname(os.path.abspath(__file__)) + "/words_alpha.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("The file does not exist.")
except PermissionError:
    print("The file exists but you do not have permission to open it.")
except Exception as e:
    print(f"Load failed: {e}")
