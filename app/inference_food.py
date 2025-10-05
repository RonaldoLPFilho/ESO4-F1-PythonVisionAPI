import os, io, time
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import open_clip
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_DET_CONF = 0.35 
MIN_CLASS_CONF = float(os.getenv("MIN_CLASS_CONF", "0"))
MODEL_VERSION = "food-v1"

# =========================
# 1) Detector de alimentos
# =========================
# Usamos YOLO-World (open-vocabulary) com um set de "nomes" de alimentos.
# Se não encontrar classe específica, ainda assim tentamos "fruit" / "vegetable".
FOOD_VOCAB = [
    "apple","banana","orange","strawberry","grape","watermelon","pineapple","mango","papaya","avocado",
    "tomato","lettuce","cabbage","broccoli","cauliflower","carrot","cucumber","zucchini","eggplant",
    "bell pepper","onion","garlic","potato","corn","peas","beans",
    "fruit","vegetable","food","dish","meal"
]

# Modelo: YOLO-World;
try:
    _detector = YOLO("yolov8x-world.pt")  # open-vocabulary
    _detector.set_classes(FOOD_VOCAB)     # restringe para o vocabulário FOOD_VOCAB
except Exception:
    _detector = YOLO("yolov8n.pt")        # fallback (COCO)

COCO_FOODS = {"apple","banana","orange","broccoli","carrot"}  # subset do COCO

def _pil_to_cv(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def detect_food_regions(img: Image.Image, conf_thres: float = MIN_DET_CONF):
    """
    Retorna lista de (bbox[x1,y1,x2,y2], nome_classe, score). Coordenadas em pixels.
    """
    frame = _pil_to_cv(img)
    res = _detector.predict(source=frame, verbose=False, conf=conf_thres)[0]

    boxes, names = [], []
    if hasattr(res, "names") and isinstance(res.names, dict):
        names = res.names

    out = []
    for i in range(len(res.boxes)):
        b = res.boxes[i]
        cls_id = int(b.cls.item()) if b.cls is not None else -1
        score = float(b.conf.item()) if b.conf is not None else 0.0
        if hasattr(b, "xyxy"):
            x1,y1,x2,y2 = b.xyxy[0].tolist()
        else:
            continue
        name = names.get(cls_id, "object")
        if _detector.model_name.endswith("yolov8n.pt"):
            if name not in COCO_FOODS:
                continue
        out.append(([x1,y1,x2,y2], name, score))
    out.sort(key=lambda t: t[2], reverse=True)
    return out

def crop_bbox(img: Image.Image, bbox: List[float]) -> Image.Image:
    w,h = img.size
    x1,y1,x2,y2 = bbox
    x1 = int(max(0, min(w-1, x1)))
    x2 = int(max(0, min(w,   x2)))
    y1 = int(max(0, min(h-1, y1)))
    y2 = int(max(0, min(h,   y2)))
    crop = img.crop((x1,y1,x2,y2))
    return crop

# ==========================
# 2) Classificador zero-shot
# ==========================
_CLIP_MODEL, _CLIP_PREP, _CLIP_TOKENIZER = None, None, None

def _load_clip():
    global _CLIP_MODEL, _CLIP_PREP, _CLIP_TOKENIZER
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _, _CLIP_PREP = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=DEVICE
        )
        _CLIP_TOKENIZER = open_clip.get_tokenizer("ViT-B-32")
        _CLIP_MODEL.eval()
_load_clip()

def _embed_image_clip(img: Image.Image):
    x = _CLIP_PREP(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = _CLIP_MODEL.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]

def _embed_text_clip(prompts: List[str]):
    tokens = _CLIP_TOKENIZER(prompts).to(DEVICE)
    with torch.no_grad():
        txt = _CLIP_MODEL.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy()

def _softmax(a: np.ndarray):
    m = a.max()
    e = np.exp(a - m)
    return e / (e.sum() + 1e-9)

def classify_fresh_vs_rotten(img: Image.Image, detected_name: Optional[str]) -> Tuple[str, float]:
    """
    Retorna ("saudavel"|"doente", confidence)
    """
    name = detected_name or "food item"

    prompts = [
        f"a close-up photo of a fresh, clean, ripe {name}",
        f"a close-up photo of a rotten, spoiled, moldy {name}"
    ] + [
        f"uma foto de perto de um(a) {name} fresco, limpo",
        f"uma foto de perto de um(a) {name} podre, estragado, com mofo"
    ]

    im = _embed_image_clip(img)
    txt = _embed_text_clip(prompts)
    sims = im @ txt.T  # (4,)
    s_fresh = max(sims[0], sims[2])
    s_rottn = max(sims[1], sims[3])
    probs = _softmax(np.array([s_fresh, s_rottn]))
    idx = int(np.argmax(probs))
    label = "saudavel" if idx == 0 else "doente"
    conf = float(probs[idx])
    return label, round(conf, 4)

# ================
# API de inferência
# ================
# def predict_image(img: Image.Image):
#     dets = [d for d in detect_food_regions(img, conf_thres=MIN_DET_CONF) if d[2] >= MIN_DET_CONF]

#     if not dets:
#         return "Não reconhecido", 0.0

#     (bbox, name, det_conf) = dets[0]
#     crop = crop_bbox(img, bbox)
#     label, class_conf = classify_fresh_vs_rotten(crop, name)

#     if MIN_CLASS_CONF and class_conf < MIN_CLASS_CONF:
#         return "Não reconhecido", class_conf

#     return label, class_conf

def _food_pt(name_en: Optional[str]) -> Optional[str]:
    if not name_en:
        return None
    m = {
        "apple":"maçã","banana":"banana","orange":"laranja","strawberry":"morango","grape":"uva",
        "watermelon":"melancia","pineapple":"abacaxi","mango":"manga","papaya":"mamão","avocado":"abacate",
        "tomato":"tomate","lettuce":"alface","cabbage":"repolho","broccoli":"brócolis","cauliflower":"couve-flor",
        "carrot":"cenoura","cucumber":"pepino","zucchini":"abobrinha","eggplant":"berinjela",
        "bell pepper":"pimentão","onion":"cebola","garlic":"alho","potato":"batata","corn":"milho",
        "peas":"ervilha","beans":"feijão","fruit":"fruta","vegetable":"legume","food":"alimento","dish":"prato","meal":"refeição"
    }
    return m.get(name_en.lower(), name_en)

def predict_image(img: Image.Image):
    """
    Retorna (label_pt, confidence, meta_dict)
    label: 'saudável' | 'doente' | 'Não reconhecido'
    meta: {'food': <pt|en>, 'food_confidence': float, 'bbox': [x1,y1,x2,y2] | None}
    """
    dets = [d for d in detect_food_regions(img, conf_thres=MIN_DET_CONF) if d[2] >= MIN_DET_CONF]

    if not dets:
        return "Não reconhecido", 0.0, {"food": None, "food_confidence": 0.0, "bbox": None}

    (bbox, name_en, det_conf) = dets[0]
    crop = crop_bbox(img, bbox)
    label, class_conf = classify_fresh_vs_rotten(crop, name_en)

    if MIN_CLASS_CONF and class_conf < MIN_CLASS_CONF:
        return "Não reconhecido", class_conf, {
            "food": _food_pt(name_en),
            "food_confidence": float(det_conf),
            "bbox": [int(b) for b in bbox],
        }

    return label, class_conf, {
        "food": _food_pt(name_en),
        "food_confidence": float(det_conf),
        "bbox": [int(b) for b in bbox],
    }