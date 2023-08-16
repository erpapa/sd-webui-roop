import cv2
import numpy as np
import gradio as gr

from PIL import Image
from typing import List
from fastapi import FastAPI, Body

from scripts.cimage import get_first_model, find_upscaler, find_face_restorer
from scripts.cimage import decode_to_pil, encode_to_base64
from scripts.swapper import UpscaleOptions, ImageResult, swap_face, get_faces
from scripts.roop_logging import logger
from scripts.roop_version import version_flag


def roop_api(_: gr.Blocks, app: FastAPI):
    @app.get("/roop/version")
    async def roop_version():
        return {"version": f"{version_flag}", "msg": "Success", "state": 200}

    @app.post("/roop/face_detect")
    async def roop_face_detect(
        source_image: str = Body("", title='Roop Source Image'),
        target_image: str = Body("", title='Roop Target Image'),
    ):
        if len(source_image) == 0:
            return {"msg": "No Source Image", "state": 404}
        source_img: Image.Image = decode_to_pil(source_image)
        source_data = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        source_faces = get_faces(source_data)
        if source_faces is None or len(source_faces) == 0:
            return {"msg": "Source Image Detect Failed", "state": 500}
        source_faces_count = len(source_faces)
        target_faces_count = 0

        if len(target_image) > 0:
            target_img: Image.Image = decode_to_pil(target_image)
            target_data = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
            target_faces = get_faces(target_data)
            if target_faces is None or len(target_faces) == 0:
                return {"msg": "Target Image Detect Failed", "state": 500}
            target_faces_count = len(target_faces)

        return {
            "source_faces_count": source_faces_count,
            "target_faces_count": target_faces_count,
            "msg": "Face Detect Success",
            "state": 200
        }

    @app.post("/roop/swap_face")
    async def roop_swap_face(
        source_image: str = Body("", title='Roop Source Image'),
        target_image: str = Body("", title='Roop Target Image'),
        model: str = Body("", title='Roop Model Path'),
        faces_index: str = Body("0", title='Roop Faces Index'),
        face_restorer_name: str = Body("CodeFormer", title='Roop Restorer Name'),
        face_restorer_visibility: float = Body(1.0, title='Roop Restorer Visibility'),
        upscaler_name: str = Body("None", title='Roop Upscaler Name'),
        upscaler_scale: int = Body(1, title='Roop Upscaler Scale'),
        upscaler_visibility: float = Body(1.0, title='Roop Upscaler Visibility'),
        nsfw_filter: bool = Body(False, title='Roop NSFW Filter'),
    ):
        if len(source_image) == 0:
            return {"msg": "No Source Image", "state": 404}
        if len(target_image) == 0:
            return {"msg": "No Target Image", "state": 404}
        if model is None or len(model) == 0:
            model = get_first_model()
        if model is None or len(model) == 0:
            return {"msg": "No Model", "state": 404}
        if upscaler_scale < 1 or upscaler_scale > 8:
            upscaler_scale = 1
        if face_restorer_visibility < 0.0 or face_restorer_visibility > 1.0:
            face_restorer_visibility = 1.0
        if upscaler_visibility < 0.0 or upscaler_visibility > 1.0:
            upscaler_visibility = 1.0

        logger.info(f"POST: /roop/swap_face，model: %s", model)
        faces_index = {
            int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
        }
        if len(faces_index) == 0:
            faces_index = {0}
        upscale_options = UpscaleOptions(scale=upscaler_scale,
                                         upscaler=find_upscaler(upscaler_name),
                                         upscale_visibility=upscaler_visibility,
                                         face_restorer=find_face_restorer(face_restorer_name),
                                         restorer_visibility=face_restorer_visibility)
        source_img: Image.Image = decode_to_pil(source_image)
        target_img: Image.Image = decode_to_pil(target_image)
        result: ImageResult = swap_face(
            source_img=source_img,
            target_img=target_img,
            model=model,
            faces_index=faces_index,
            upscale_options=upscale_options,
            nsfw_filter=nsfw_filter
        )
        result_str = encode_to_base64(result.image())
        return {
            "image": result_str,
            "msg": "Swap Face Success",
            "state": 200
        }

    @app.post("/roop/batch_swap_face")
    async def roop_batch_swap_face(
        source_image: str = Body("", title='Roop Source Image'),
        target_images: List[str] = Body([], title='Roop Target Images'),
        model: str = Body("", title='Roop Model Path'),
        faces_index: str = Body("0", title='Roop Faces Index'),
        face_restorer_name: str = Body("CodeFormer", title='Roop Restorer Name'),
        face_restorer_visibility: float = Body(1.0, title='Roop Restorer Visibility'),
        upscaler_name: str = Body("None", title='Roop Upscaler Name'),
        upscaler_scale: int = Body(1, title='Roop Upscaler Scale'),
        upscaler_visibility: float = Body(1.0, title='Roop Upscaler Visibility'),
        nsfw_filter: bool = Body(False, title='Roop NSFW Filter'),
    ):
        if len(source_image) == 0:
            return {"msg": "No Source Image", "state": 404}
        if len(target_images) == 0:
            return {"msg": "No Target Images", "state": 404}
        if model is None or len(model) == 0:
            model = get_first_model()
        if model is None or len(model) == 0:
            return {"msg": "No Model", "state": 404}
        if upscaler_scale < 1 or upscaler_scale > 8:
            upscaler_scale = 1
        if face_restorer_visibility < 0.0 or face_restorer_visibility > 1.0:
            face_restorer_visibility = 1.0
        if upscaler_visibility < 0.0 or upscaler_visibility > 1.0:
            upscaler_visibility = 1.0

        logger.info(f"POST: /roop/batch_swap_face，model: %s", model)
        faces_index = {
            int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
        }
        if len(faces_index) == 0:
            faces_index = {0}
        upscale_options = UpscaleOptions(scale=upscaler_scale,
                                         upscaler=find_upscaler(upscaler_name),
                                         upscale_visibility=upscaler_visibility,
                                         face_restorer=find_face_restorer(face_restorer_name),
                                         restorer_visibility=face_restorer_visibility)
        source_img: Image.Image = decode_to_pil(source_image)
        result_imgs: List = []
        for img_str in target_images:
            target_img: Image.Image = decode_to_pil(img_str)
            result: ImageResult = swap_face(
                source_img=source_img,
                target_img=target_img,
                model=model,
                faces_index=faces_index,
                upscale_options=upscale_options,
                nsfw_filter=nsfw_filter
            )
            result_str = encode_to_base64(result.image())
            result_imgs.append(result_str)
        return {
            "images": result_imgs,
            "msg": "Swap Face Success",
            "state": 200
        }


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(roop_api)
except:
    pass
