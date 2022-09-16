# -- coding: utf-8 --`
import argparse
import os
# engine
from stable_diffusion_engine import StableDiffusionEngine
# scheduler
from diffusers import LMSDiscreteScheduler, PNDMScheduler
# utils
import cv2
import numpy as np
import time

def current_milli_time():
    return round(time.time() * 1000)

# pipeline configure
# parser.add_argument("--model", type=str, default="bes-dev/stable-diffusion-v1-4-openvino", help="model name")

# scheduler params
# parser.add_argument("--beta-start", type=float, default=0.00085, help="LMSDiscreteScheduler::beta_start")
# parser.add_argument("--beta-end", type=float, default=0.012, help="LMSDiscreteScheduler::beta_end")
# parser.add_argument("--beta-schedule", type=str, default="scaled_linear", help="LMSDiscreteScheduler::beta_schedule")

# diffusion params
# parser.add_argument("--num-inference-steps", type=int, default=32, help="num inference steps")
# parser.add_argument("--guidance-scale", type=float, default=7.5, help="guidance scale")
# parser.add_argument("--eta", type=float, default=0.0, help="eta")

# tokenizer
# parser.add_argument("--tokenizer", type=str, default="openai/clip-vit-large-patch14", help="tokenizer")

# prompt
# parser.add_argument("--prompt", type=str, default="Street-art painting of Emilia Clarke in style of Banksy, photorealism", help="prompt")

# img2img params
# parser.add_argument("--strength", type=float, default=0.5, help="how strong the initial image should be noised [0.0, 1.0]")

# output name
# parser.add_argument("--output", type=str, default="output.png", help="output image name")


def img2img(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", model="bes-dev/stable-diffusion-v1-4-openvino", tokenizer="openai/clip-vit-large-patch14", prompt="Street-art painting of Emilia Clarke in style of Banksy, photorealism", strength=0.5, num_inference_steps=32, guidance_scale=7.5, eta=0.0, output="output.png", init_image=None):
    scheduler = PNDMScheduler(
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        skip_prk_steps = True,
        tensor_format="np"
    )
    engine = StableDiffusionEngine(
        model = model,
        scheduler = scheduler,
        tokenizer = tokenizer
    )
    image = engine(
        prompt = prompt,
        init_image = None if init_image is None else cv2.imread(init_image),
        # mask = None if mask is None else cv2.imread(mask, 0),
        strength = strength,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        eta = eta
    )

    is_outputs_folder_exist = os.path.exists(os.path.join(os.getcwd(), 'outputs'))
    outputs_folder = os.path.join(os.getcwd(), 'outputs')

    if not is_outputs_folder_exist:
        os.mkdir(outputs_folder)

    filename = str(current_milli_time()) + output

    save_path = os.path.join(outputs_folder, filename)

    cv2.imwrite(save_path, image)

    return filename