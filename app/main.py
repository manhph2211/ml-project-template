from fastapi import FastAPI, Form
from pydantic import BaseModel
import pickle
import re
from pathlib import Path

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

# with open(f"{BASE_DIR}/weights/model-{__version__}.pkl", "rb") as f:
#     model = pickle.load(f)

def predict(text):
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`a-z]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    pred = model.predict([text])
    return pred


app = FastAPI()


@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": __version__ }


@app.post("/gen", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict(payload.text)
    return {"language": language}