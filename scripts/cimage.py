import os
import glob
import numpy as np

from PIL import Image
from ifnude import detect
from modules import scripts
from modules import shared
from modules.upscaler import Upscaler, UpscalerData
from modules.face_restoration import FaceRestoration
from modules.api.models import *
from modules.api import api

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


def get_first_model():
    model = "inswapper_128.onnx"
    models = get_models()
    if len(models) > 0:
        model = model[0]
    return model


def find_upscaler(name: str) -> UpscalerData:
    for upscaler in shared.sd_upscalers:
        if upscaler.name == name:
            return upscaler
    return None


def find_face_restorer(name: str) -> FaceRestoration:
    for face_restorer in shared.face_restorers:
        if face_restorer.name() == name:
            return face_restorer
    return None


def decode_to_pil(image):
    if os.path.exists(image):
        return Image.open(image)
    elif type(image) is str:
        return api.decode_base64_to_image(image)
    elif type(image) is Image.Image:
        return image
    elif type(image) is np.ndarray:
        return Image.fromarray(image)
    else:
        Exception("Not an image")


def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return api.encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""


def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


def to_base64_nparray(encoding: str):
    pil = api.decode_base64_to_image(encoding)
    return np.array(pil).astype('uint8')

