
import requests
import streamlit as st
from PIL import Image
import sys
sys.path.append(".")

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("PLAY WITH JAX - IMAGE GENERATION")

text = st.selectbox("Choose the number", [i for i in range(10)])

if st.button("Generate Image ..."):
    if text is not None:
        files = {"text": str(text)}
        res = requests.post("http://backend:8080/gen", json=files)
        img_path = res.json()
        print(img_path, flush=True)
        image = Image.open(img_path.get("name"))
        st.image(image, width=500)