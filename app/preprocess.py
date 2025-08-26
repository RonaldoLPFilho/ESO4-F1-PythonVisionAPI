from PIL import Image
import io

def load_image_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b)).convert("RGB")

    img = img.resize((224, 244))
    return img

