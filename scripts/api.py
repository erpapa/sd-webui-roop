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
from scripts.swapper import UpscaleOptions, ImageResult, swap_face
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


def upscaler(name: str) -> UpscalerData:
    for upscaler in shared.sd_upscalers:
        if upscaler.name == name:
            return upscaler
    return None


def face_restorer(name: str) -> FaceRestoration:
    for face_restorer in shared.face_restorers:
        if face_restorer.name() == name:
            return face_restorer
    return None


def roop_api(_: gr.Blocks, app: FastAPI):
    @app.get("/roop/version")
    async def roop_version():
        return {"version": "0.0.2"}

    @app.post("/roop/swap_face")
    async def roop_swap_face(
        roop_source_image: str = Body("", title='Roop Source Image'),
        roop_target_image: str = Body("", title='Roop Target Image'),
        roop_model: str = Body("", title='Roop Model Path'),
        roop_scale: int = Body(1, title='Roop Image Scale'),
        roop_upscaler: str = Body("None", title='Roop Upscaler Name'),
        roop_face_restorer: str = Body("CodeFormer", title='Roop Face Restorer'),
        roop_restorer_visibility: float = Body(1.0, title='Roop Restorer Visibility'),
        roop_upscale_visibility: float = Body(1.0, title='Roop Upscale Visibility'),
        roop_detect_porn: bool = Body(False, title='Roop Detect Porn'),
    ):
        if len(roop_source_image) == 0:
            return {"msg": "No Source Image", "info": "Failed"}
        if len(roop_target_image) == 0:
            return {"msg": "No Target Image", "info": "Failed"}
        if roop_model is None or len(roop_model) == 0:
            roop_model = get_models()[0]
        if roop_model is None or len(roop_model) == 0:
            return {"msg": "No Model", "info": "Failed"}
        if roop_scale < 1 or roop_scale > 8:
            roop_scale = 1
        if roop_restorer_visibility < 0.0 or roop_restorer_visibility > 1.0:
            roop_restorer_visibility = 1.0
        if roop_upscale_visibility < 0.0 or roop_upscale_visibility > 1.0:
            roop_upscale_visibility = 1.0

        logger.info(f"POST: /roop/swap_face，roop_model: %s", roop_model)
        upscale_options = UpscaleOptions(scale=roop_scale,
                                         upscaler=upscaler(roop_upscaler),
                                         upscale_visibility=roop_upscale_visibility,
                                         face_restorer=face_restorer(roop_face_restorer),
                                         restorer_visibility=roop_restorer_visibility)
        source_img: Image.Image = decode_to_pil(roop_source_image)
        target_img: Image.Image = decode_to_pil(roop_target_image)
        result: ImageResult = swap_face(
            source_img,
            target_img,
            model=roop_model,
            faces_index={0},
            upscale_options=upscale_options,
            detect_porn=roop_detect_porn
        )
        result_str = encode_to_base64(result.image())
        return {"image": result_str, "info": "Success"}

    @app.post("/roop/batch_swap_face")
    async def roop_batch_swap_face(
        roop_source_image: str = Body("", title='Roop Source Image'),
        roop_target_images: List[str] = Body([], title='Roop Target Images'),
        roop_model: str = Body("", title='Roop Model Path'),
        roop_scale: int = Body(1, title='Roop Image Scale'),
        roop_upscaler: str = Body("None", title='Roop Upscaler Name'),
        roop_face_restorer: str = Body("CodeFormer", title='Roop Face Restorer'),
        roop_restorer_visibility: float = Body(1.0, title='Roop Restorer Visibility'),
        roop_upscale_visibility: float = Body(1.0, title='Roop Upscale Visibility'),
        roop_detect_porn: bool = Body(False, title='Roop Detect Porn'),
    ):
        if len(roop_source_image) == 0:
            return {"msg": "No Source Image", "info": "Failed"}
        if len(roop_target_images) == 0:
            return {"msg": "No Target Images", "info": "Failed"}
        if roop_model is None or len(roop_model) == 0:
            roop_model = get_models()[0]
        if roop_model is None or len(roop_model) == 0:
            return {"msg": "No Model", "info": "Failed"}
        if roop_scale < 1 or roop_scale > 8:
            roop_scale = 1
        if roop_restorer_visibility < 0.0 or roop_restorer_visibility > 1.0:
            roop_restorer_visibility = 1.0
        if roop_upscale_visibility < 0.0 or roop_upscale_visibility > 1.0:
            roop_upscale_visibility = 1.0

        logger.info(f"POST: /roop/batch_swap_face，roop_model: %s", roop_model)
        upscale_options = UpscaleOptions(scale=roop_scale,
                                         upscaler=upscaler(roop_upscaler),
                                         upscale_visibility=roop_upscale_visibility,
                                         face_restorer=face_restorer(roop_face_restorer),
                                         restorer_visibility=roop_restorer_visibility)
        source_img: Image.Image = decode_to_pil(roop_source_image)
        result_imgs: List = []
        for img_str in roop_target_images:
            target_img: Image.Image = decode_to_pil(img_str)
            result: ImageResult = swap_face(
                source_img,
                target_img,
                model=roop_model,
                faces_index={0},
                upscale_options=upscale_options,
                detect_porn=roop_detect_porn
            )
            result_str = encode_to_base64(result.image())
            result_imgs.append(result_str)
        return {"images": result_imgs, "info": "Success"}


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(roop_api)
except:
    pass
