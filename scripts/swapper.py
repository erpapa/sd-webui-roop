import os
import cv2
import insightface
import numpy as np

from PIL import Image
from typing import Union, Dict, Set
from dataclasses import dataclass

from modules.face_restoration import FaceRestoration, restore_faces
from modules.upscaler import Upscaler, UpscalerData

from scripts.cimage import detect_image, decode_to_pil
from scripts.roop_logging import logger

CURRENT_FS_MODEL = None
CURRENT_FS_MODEL_PATH = None
FACE_MODEL_NAME = None
FACE_ANALYSER = None
FACE_DET_SIZE = (0, 0)
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']


@dataclass
class UpscaleOptions:
    scale: int = 1
    upscaler: UpscalerData = None
    upscale_visibility: float = 0.5
    face_restorer: FaceRestoration = None
    restorer_visibility: float = 0.5


def get_face_swap_model(model_path: str):
    global CURRENT_FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        CURRENT_FS_MODEL = insightface.model_zoo.get_model(model_path, providers=PROVIDERS)

    return CURRENT_FS_MODEL


def get_face_analyser(name: str, det_size: tuple):
    global FACE_MODEL_NAME
    global FACE_ANALYSER
    global FACE_DET_SIZE
    if FACE_MODEL_NAME is None or FACE_MODEL_NAME != name:
        FACE_MODEL_NAME = name
        FACE_ANALYSER = insightface.app.FaceAnalysis(name=name, providers=PROVIDERS)
    if FACE_DET_SIZE[0] != det_size[0] or FACE_DET_SIZE[1] != det_size[1]:
        FACE_DET_SIZE = det_size
        FACE_ANALYSER.prepare(ctx_id=0, det_size=det_size)

    return FACE_ANALYSER


def upscale_image(image: Image, upscale_options: UpscaleOptions):
    result_image = image
    if upscale_options.upscaler is not None and upscale_options.upscaler.name != "None":
        original_image = result_image.copy()
        logger.info(
            "Upscale with %s scale = %s",
            upscale_options.upscaler.name,
            upscale_options.scale,
        )
        result_image = upscale_options.upscaler.scaler.upscale(
            image, upscale_options.scale, upscale_options.upscaler.data_path
        )
        if upscale_options.scale == 1:
            result_image = Image.blend(
                original_image, result_image, upscale_options.upscale_visibility
            )

    if upscale_options.face_restorer is not None:
        original_image = result_image.copy()
        logger.info("Restore face with %s", upscale_options.face_restorer.name())
        numpy_image = np.array(result_image)
        numpy_image = upscale_options.face_restorer.restore(numpy_image)
        restored_image = Image.fromarray(numpy_image)
        result_image = Image.blend(
            original_image, restored_image, upscale_options.restorer_visibility
        )

    return result_image


def get_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = get_face_analyser(name="buffalo_l", det_size=det_size)
    faces = face_analyser.get(img_data)

    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_faces(img_data, det_size=det_size_half)

    try:
        return sorted(faces, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = get_face_analyser(name="buffalo_l", det_size=det_size)
    faces = face_analyser.get(img_data)

    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(faces, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


@dataclass
class ImageResult:
    img: Image.Image = None
    path: Union[str, None] = None
    similarity: Union[Dict[int, float], None] = None  # face, 0..1

    def image(self) -> Union[Image.Image, None]:
        if self.img:
            return self.img
        if self.path:
            self.img = Image.open(self.path)
            return self.img
        return None


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    faces_index: Set[int] = {0},
    upscale_options: Union[UpscaleOptions, None] = None,
    nsfw_filter: bool = True,
) -> ImageResult:
    result_image = target_img
    detect_flag = False
    if nsfw_filter:
        detect_flag = detect_image(target_img)
    if detect_flag:
        return ImageResult(img=result_image)

    if model is not None:
        if isinstance(source_img, str):  # source_img is a base64 string
            source_img = decode_to_pil(source_img)
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
            face_swapper = get_face_swap_model(model_path)

            for face_num in faces_index:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                else:
                    logger.info(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            if upscale_options is not None:
                result_image = upscale_image(result_image, upscale_options)
        else:
            logger.info("No source face found")

    return ImageResult(img=result_image)
