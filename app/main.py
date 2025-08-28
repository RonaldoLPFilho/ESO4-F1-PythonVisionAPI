from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import base64, time, json, os, io

from .inference_clip import (
    predict_zeroshot, predict_prototype, MODEL_VERSION,
    add_prototype, reset_prototypes, stats_prototypes
)

app = FastAPI(title="Vision Core (CLIP few-shot)", version="0.2.0")

class Base64Image(BaseModel):
    image_base64: str

class LabelledBase64(BaseModel):
    image_base64: str
    label: str  # "saudavel" | "doente"

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.get("/metrics")
def metrics():
    return {"model_version": MODEL_VERSION, "prototypes": stats_prototypes()}

def _bytes_to_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

@app.post("/predict")
async def predict(image: UploadFile = File(None)):
    if image is None:
        raise HTTPException(status_code=400, detail="Enviar campo multipart 'image'.")
    t0 = time.time()
    content = await image.read()
    img = _bytes_to_image(content)

    # Tenta protótipo; se não houver, cai no zero-shot
    label, conf, meta = predict_prototype(img)
    ms = int((time.time() - t0) * 1000)
    return JSONResponse({
        "label": label,
        "confidence": conf,
        "model_version": MODEL_VERSION,
        "processing_ms": ms,
        "mode": meta["mode"]
    })

@app.post("/predict/base64")
async def predict_base64(payload: Base64Image):
    try:
        raw = base64.b64decode(payload.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido.")
    t0 = time.time()
    img = _bytes_to_image(raw)
    label, conf, meta = predict_prototype(img)
    ms = int((time.time() - t0) * 1000)
    return {
        "label": label,
        "confidence": conf,
        "model_version": MODEL_VERSION,
        "processing_ms": ms,
        "mode": meta["mode"]
    }

# ---------- Few-shot maintenance ----------
@app.post("/prototype/add")
async def prototype_add(payload: LabelledBase64):
    try:
        raw = base64.b64decode(payload.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 inválido.")
    img = _bytes_to_image(raw)
    try:
        add_prototype(payload.label)
    except TypeError:
        # se a assinatura da função mudou, tente com img:
        add_prototype(payload.label, img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "prototypes": stats_prototypes()}

@app.post("/prototype/reset")
def prototype_reset():
    reset_prototypes()
    return {"ok": True, "prototypes": stats_prototypes()}

@app.get("/prototype/stats")
def prototype_stats():
    return {"prototypes": stats_prototypes()}
