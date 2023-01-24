from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
import pickle
import re
from pathlib import Path
from PIL import Image
import uvicorn
import sys
sys.path.append("../")


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


@app.post("/gen")
async def gen(text):
    # Generate the predicted image
    img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    img = img.save(f"/storage/{text.text}.jpg")
    return {"name":f"/storage/{text.text}.jpg"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)

