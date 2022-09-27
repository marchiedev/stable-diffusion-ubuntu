from typing import Union, Optional

from fastapi import FastAPI, Form, UploadFile, File

from fastapi.responses import FileResponse

import translators as ts

#mutations
from resolver.test.query.test import test
from resolver.generate_image.mutation.txt2img import txt2img
from resolver.generate_image.mutation.img2img import img2img

#queries
from resolver.generate_image.query.output_img import output_img

app = FastAPI()

@app.get("/")
def read_root():
    res = test()
    return res

@app.post('/generate-image/img2img')
def generate_image_to_image(strength: Optional[float] = Form(None), prompt: Union[str, None] = Form(), initial_image: UploadFile = File()):
    is_inputs_folder_exist = os.path.exists(os.path.join(os.getcwd(), 'inputs'))
    inputs_folder = os.path.join(os.getcwd(), 'inputs')

    if not is_outputs_folder_exist:
        os.mkdir(inputs_folder)

    file_location = os.path.join(inputs_folder, initial_image.filename)

    with open(file_location, "wb+") as file_object:
        file_object.write(initial_image.file.read())

    translated_prompt = ts.google(prompt)

    generate_filename = txt2img(prompt=translated_prompt, init_image=file_location)

    filename = './outputs/' + generate_filename

    return FileResponse(filename)


@app.post('/generate-image/txt2img')
def generate_text_to_image(strength: Optional[float] = Form(None), prompt: Union[str, None] = Form()):
    translated_prompt = ts.google(prompt)

    print(translated_prompt)

    generate_filename = txt2img(prompt=translated_prompt)

    filename = './outputs/' + generate_filename

    return FileResponse(filename)

@app.get('/generate-image/txt2img/{filename}')
def get_generate_text_to_image_by_filename(filename: str):
    file_location = output_img(filename)

    return FileResponse(file_location)