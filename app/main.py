# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import base64, time, io

from .inference_food import predict_image, MODEL_VERSION

app = FastAPI(title="Vision Core (Food)", version="0.3.0")

class Base64Image(BaseModel):
    image_base64: str

def _bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.get("/metrics")
def metrics():
    # Sem treino aqui: você pode enriquecer com contagem de detecções, etc.
    return {"model_version": MODEL_VERSION, "note": "zero-shot food detector + CLIP classifier"}

@app.post("/predict")
async def predict(image: UploadFile = File(None)):
    if image is None:
        raise HTTPException(status_code=400, detail="Enviar campo multipart 'image'.")
    t0 = time.time()
    content = await image.read()
    img = _bytes_to_image(content)
    label, conf = predict_image(img)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse({
        "label": label,
        "confidence": conf,
        "model_version": MODEL_VERSION,
        "processing_ms": ms
    })

@app.post("/predict/base64")
async def predict_base64(payload: Base64Image):
    try:
        raw = base64.b64decode(payload.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido.")
    t0 = time.time()
    img = _bytes_to_image(raw)
    label, conf = predict_image(img)
    ms = int((time.time() - t0) * 1000)
    return {
        "label": label,
        "confidence": conf,
        "model_version": MODEL_VERSION,
        "processing_ms": ms
    }
