from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64, time, json, os

from .preprocess import load_image_from_bytes
from .inference import predict_image, MODEL_VERSION

app = FastAPI(title="Vision Core", version="0.1.0")

class Base64Image(BaseModel):
    image_base64: str

@app.get("/health")
def health():
    return {"Status": "OK"}


@app.get("/metrics")
def metrics():
    path = os.path.join(os.path.dirname(__file__), "metrics.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return{
            "model_version": MODEL_VERSION,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tested_at": None
        }
    
@app.post("/predict")
async def predict(image: UploadFile = File(None)):
    """
    Aceita multipart `image` 
    """
    if image is None:
        raise HTTPException(status_code=400, detail="Enviar campo multipart 'image'")
    t0 = time.time()
    content = await image.read()
    img = load_image_from_bytes(content)
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
    """
    Alternativa para enviar em abse64
    """
    try:
        raw = base64.b64decode(payload.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido")
    
    t0 = time.time()
    img = load_image_from_bytes(raw)
    label, conf = predict_image(img)
    ms = int((time.time() - t0) * 1000)

    return JSONResponse({
        "label": label,
        "confidence": conf,
        "model_version": MODEL_VERSION,
        "processing_ms": ms
    })
