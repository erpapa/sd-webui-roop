import os
import glob
from ifnude import detect
from modules import scripts


def detect_image(img):
    shapes = []
    chunks = detect(img)
    for chunk in chunks:
        shapes.append(chunk["score"] > 0.7)
    return any(shapes)


def get_models():
    models_path = os.path.join(scripts.basedir(), "models" + os.path.sep + "roop" + os.path.sep + "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    return models