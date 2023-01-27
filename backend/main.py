from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
from PIL import Image
import uvicorn
from src.inference import infer
from matplotlib import cm
import numpy as np

app = FastAPI()

__version__ = "0.1.0"

class TextIn(BaseModel):
    text: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": __version__ }


@app.post("/gen")
async def gen(text:TextIn):
    img = np.uint8(infer(int(text.text))*255)
    img = Image.fromarray(img)
    img = img.save(f"/storage/{text.text}.jpg")
    return {"name":f"/storage/{text.text}.jpg"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

