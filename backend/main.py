from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
import pickle
import re
from pathlib import Path
from PIL import Image
from io import BytesIO
import uvicorn

app = FastAPI()

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# with open(f"{BASE_DIR}/weights/model-{__version__}.pkl", "rb") as f:
#     model = pickle.load(f)

def predict(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`a-z]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    pred = model.predict([text])
    return pred


class TextIn(BaseModel):
    text: str


@app.post("/login")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": __version__ }


@app.post("/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    return {"name": "Update later"}


@app.post("/gen")
async def gen(payload: TextIn):
    # Generate the predicted image
    img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    img = img.save("backend/output/images/output.jpg")
    return {"name":"backend/output/images/output.jpg"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)

