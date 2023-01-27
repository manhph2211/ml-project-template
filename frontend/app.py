
import requests
import streamlit as st
import json
from PIL import Image

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("PLAY WITH JAX - IMAGE GENERATION")

text = st.selectbox("Choose the number", [i for i in range(10)])

if st.button("Generate Image ..."):
    if text is not None:
        files = {"text": str(text)}
        res = requests.post(f"http://backend:8080/gen", data= json.dumps(files))
        img_path = res.json()
        image = Image.open(img_path.get("name"))
        st.image(image, width=500)