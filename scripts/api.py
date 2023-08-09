import os
import numpy as np
import gradio as gr

from PIL import Image
from fastapi import FastAPI, Body
from typing import List

from modules import shared
from modules.upscaler import Upscaler, UpscalerData
from modules.face_restoration import FaceRestoration
from modules.api.models import *
from modules.api import api

from scripts.cimage import get_models
from scripts.swapper import UpscaleOptions, ImageResult, swap_face_v2
from scripts.roop_logging import logger

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

def upscaler() -> UpscalerData:
    for upscaler in shared.sd_upscalers:
        if upscaler.name == "LDSR":
            return upscaler
    return None

def face_restorer() -> FaceRestoration:
    for face_restorer in shared.face_restorers:
        if face_restorer.name() == "CodeFormer":
            return face_restorer
    return None

def roop_api(_: gr.Blocks, app: FastAPI):
    @app.get("/roop/version")
    async def version():
        return {"version": "0.0.2"}

    @app.post("/roop/detect")
    async def detect(
        roop_source_image: str = Body("none", title='Roop Source Image'),
        roop_target_image: str = Body("none", title='Roop Target Image'),
        roop_model: str = Body("none", title='Roop Model Path'),
    ):
        if len(roop_source_image) == 0:
            return {"msg": "No Source Image", "info": "Failed"}
        if len(roop_target_image) == 0:
            return {"msg": "No Target Image", "info": "Failed"}
        if len(roop_model) == 0:
            roop_model = get_models()[0]
        if len(roop_model) == 0:
            return {"msg": "No Model", "info": "Failed"}

        logger.info(f"Roop app.post：/roop/detect，roop_model：%s", roop_model)
        upscale_options = UpscaleOptions(scale=1,
                                         upscaler=upscaler(),
                                         upscale_visibility=1.0,
                                         face_restorer=face_restorer(),
                                         restorer_visibility=1.0)
        source_img: Image.Image = decode_to_pil(roop_source_image)
        target_img: Image.Image = decode_to_pil(roop_target_image)
        result: ImageResult = swap_face_v2(
            source_img,
            target_img,
            model=roop_model,
            faces_index={0},
            upscale_options=upscale_options,
        )
        result_str = encode_to_base64(result.image())
        return {"image": result_str, "info": "Success"}

    @app.post("/roop/batch_detect")
    async def batch_detect(
        roop_source_image: str = Body("none", title='Roop Source Image'),
        roop_target_images: List[str] = Body([], title='Roop Target Images'),
        roop_model: str = Body("none", title='Roop Model Path'),
    ):
        if len(roop_source_image) == 0:
            return {"msg": "No Source Image", "info": "Failed"}
        if len(roop_target_images) == 0:
            return {"msg": "No Target Images", "info": "Failed"}
        if len(roop_model) == 0:
            roop_model = get_models()[0]
        if len(roop_model) == 0:
            return {"msg": "No Model", "info": "Failed"}

        logger.info(f"Roop app.post：/roop/batch_detect，roop_model：%s", roop_model)
        upscale_options = UpscaleOptions(scale=1,
                                         upscaler=upscaler(),
                                         upscale_visibility=1.0,
                                         face_restorer=face_restorer(),
                                         restorer_visibility=1.0)
        source_img: Image.Image = decode_to_pil(roop_source_image)
        result_imgs: List = []
        for img_str in roop_target_images:
            target_img: Image.Image = decode_to_pil(img_str)
            result: ImageResult = swap_face_v2(
                source_img,
                target_img,
                model=roop_model,
                faces_index={0},
                upscale_options=upscale_options,
            )
            result_str = encode_to_base64(result.image())
            result_imgs.append(result_str)
        return {"images": result_imgs, "info": "Success"}

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(roop_api)
except:
    pass