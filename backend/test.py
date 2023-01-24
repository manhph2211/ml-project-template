
from pathlib import Path
from PIL import Image


BASE_DIR = Path(__file__).resolve(strict=True).parent
def get():
    img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    img = img.save("frontend/output/images/output.jpg")
    return {"name":"backend/output/images/output.jpg"}

print(get())



