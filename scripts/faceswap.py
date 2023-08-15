import gradio as gr

from PIL import Image
from typing import List, Any
from modules.upscaler import Upscaler, UpscalerData
from modules import scripts, shared, images, scripts_postprocessing
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)
from modules.face_restoration import FaceRestoration

from scripts.cimage import get_models, get_first_model, decode_to_pil
from scripts.roop_logging import logger
from scripts.swapper import UpscaleOptions, ImageResult, swap_face
from scripts.roop_version import version_flag


class FaceSwapScript(scripts.Script):
    def title(self):
        return f"roop"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(f"roop {version_flag}", open=False):
            with gr.Column():
                image = gr.inputs.Image(type="pil")
                enable = gr.Checkbox(False, placeholder="enable", label="Enable")
                faces_index = gr.Textbox(
                    value="0",
                    placeholder="Which face to swap (comma separated), start from 0",
                    label="Comma separated face number(s)",
                )
                with gr.Row():
                    face_restorer_name = gr.Radio(
                        label="Restore Face",
                        choices=["None"] + [x.name() for x in shared.face_restorers],
                        value=shared.face_restorers[0].name(),
                        type="value",
                    )
                    face_restorer_visibility = gr.Slider(
                        0, 1, 1, step=0.1, label="Restore visibility"
                    )
                upscaler_name = gr.inputs.Dropdown(
                    choices=[upscaler.name for upscaler in shared.sd_upscalers],
                    label="Upscaler",
                )
                upscaler_scale = gr.Slider(1, 8, 1, step=0.1, label="Upscaler scale")
                upscaler_visibility = gr.Slider(
                    0, 1, 1, step=0.1, label="Upscaler visibility (if scale = 1)"
                )

                models = get_models()
                if len(models) == 0:
                    logger.warning(
                        "You should at least have one model in models directory, please read the doc here : https://github.com/s0md3v/sd-webui-roop/"
                    )
                    model = gr.inputs.Dropdown(
                        choices=models,
                        label="Model not found, please download one and reload automatic 1111",
                    )
                else:
                    model = gr.inputs.Dropdown(
                        choices=models, label="Model", default=models[0]
                    )

                swap_in_source = gr.Checkbox(
                    False,
                    placeholder="Swap face in source image",
                    label="Swap in source image",
                    visible=is_img2img,
                )
                swap_in_generated = gr.Checkbox(
                    True,
                    placeholder="Swap face in generated image",
                    label="Swap in generated image",
                    visible=is_img2img,
                )
                nsfw_filter = gr.Checkbox(
                    True,
                    placeholder="Swap face with nsfw filter",
                    label="NSFW filter",
                    visible=is_img2img,
                )

        return [
            image,
            enable,
            faces_index,
            model,
            face_restorer_name,
            face_restorer_visibility,
            upscaler_name,
            upscaler_scale,
            upscaler_visibility,
            swap_in_source,
            swap_in_generated,
            nsfw_filter,
        ]

    @property
    def upscaler(self) -> UpscalerData:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> FaceRestoration:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None

    @property
    def upscale_options(self) -> UpscaleOptions:
        return UpscaleOptions(
            scale=self.upscaler_scale,
            upscaler=self.upscaler,
            face_restorer=self.face_restorer,
            upscale_visibility=self.upscaler_visibility,
            restorer_visibility=self.face_restorer_visibility,
        )

    def parse_args(self, script_args: List[Any]):
        arg_dict = script_args[0]
        source = script_args[0]
        enable = script_args[1]
        faces_index = script_args[2]
        model = script_args[3]
        face_restorer_name = script_args[4]
        face_restorer_visibility = script_args[5]
        upscaler_name = script_args[6]
        upscaler_scale = script_args[7]
        upscaler_visibility = script_args[8]
        swap_in_source = script_args[9]
        swap_in_generated = script_args[10]
        nsfw_filter = script_args[11]
        if isinstance(arg_dict, dict):
            source = arg_dict.get("image", "")
            enable = arg_dict.get("enable", enable)
            faces_index = arg_dict.get("faces_index", faces_index)
            model = arg_dict.get("model", model)
            face_restorer_name = arg_dict.get("face_restorer_name", face_restorer_name)
            face_restorer_visibility = arg_dict.get("face_restorer_visibility", face_restorer_visibility)
            upscaler_name = arg_dict.get("upscaler_name", upscaler_name)
            upscaler_scale = arg_dict.get("upscaler_scale", upscaler_scale)
            upscaler_visibility = arg_dict.get("upscaler_visibility", upscaler_visibility)
            swap_in_source = arg_dict.get("swap_in_source", swap_in_source)
            swap_in_generated = arg_dict.get("swap_in_generated", swap_in_generated)
            nsfw_filter = arg_dict.get("nsfw_filter", nsfw_filter)

        if isinstance(source, str):
            self.source = decode_to_pil(source)
        else:
            self.source = source

        self.faces_index = {
            int(x) for x in faces_index.strip(",").split(",") if x.isnumeric()
        }
        if len(self.faces_index) == 0:
            self.faces_index = {0}

        if model is None or len(model) == 0:
            model = get_first_model()

        self.enable = enable
        self.model = model
        self.face_restorer_name = face_restorer_name
        self.face_restorer_visibility = face_restorer_visibility
        self.upscaler_name = upscaler_name
        self.upscaler_scale = upscaler_scale
        self.upscaler_visibility = upscaler_visibility
        self.swap_in_source = swap_in_source
        self.swap_in_generated = swap_in_generated
        self.nsfw_filter = nsfw_filter

    def process(self, p: StableDiffusionProcessing, *args):
        self.parse_args(args)
        if self.enable and self.swap_in_source:
            logger.info(f"process: enable=%d, model=%s", self.enable, self.model)
            if self.source is not None and isinstance(p, StableDiffusionProcessingImg2Img):
                for i in range(len(p.init_images)):
                    logger.info(f"Swap in source %s", i)
                    result = swap_face(
                        source_img=self.source,
                        target_img=p.init_images[i],
                        model=self.model,
                        faces_index=self.faces_index,
                        upscale_options=self.upscale_options,
                        nsfw_filter=self.nsfw_filter,
                    )
                    p.init_images[i] = result.image()

    def postprocess_batch(self, *args, **kwargs):
        if self.enable:
            logger.info(f"postprocess_batch: enable=%d, model=%s", self.enable, self.model)
            return images

    def postprocess_image(self, p, script_pp: scripts.PostprocessImageArgs, *args):
        if self.enable and self.swap_in_generated:
            logger.info(f"postprocess_image: enable=%d, model=%s", self.enable, self.model)
            if self.source is not None:
                image: Image.Image = script_pp.image
                result: ImageResult = swap_face(
                    source_img=self.source,
                    target_img=image,
                    model=self.model,
                    faces_index=self.faces_index,
                    upscale_options=self.upscale_options,
                    nsfw_filter=self.nsfw_filter,
                )
                pp = scripts_postprocessing.PostprocessedImage(result.image())
                pp.info = {}
                p.extra_generation_params.update(pp.info)
                script_pp.image = pp.image
