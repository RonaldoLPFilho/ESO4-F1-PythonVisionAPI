import os, io, time
from typing import List, Tuple, Optional, Dict
from collections import Counter

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import open_clip
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_DET_CONF = 0.40  # Aumentado de 0.35 para reduzir falsos positivos
MIN_CLASS_CONF = float(os.getenv("MIN_CLASS_CONF", "0"))  
MODEL_VERSION = "food-v1.1"  # Versão refinada

TYPE_MIN_CONF = float(os.getenv("TYPE_MIN_CONF", "0.60"))  # Aumentado de 0.45

# =========================
# 1) Detector de alimentos
# =========================
FOOD_TYPES = {
    "tomato": ["tomato", "red tomato", "ripe tomato", "cherry tomato", "roma tomato"],
    "apple": ["apple", "red apple", "green apple", "granny smith apple"],
    "banana": ["banana", "ripe banana", "yellow banana", "plantain", "green banana"],
    "lettuce": ["lettuce", "romaine lettuce", "iceberg lettuce", "leaf lettuce", "green lettuce"],
    "strawberry": ["strawberry", "fresh strawberry", "ripe strawberry"],
    "grape": ["grape", "green grape", "red grape", "grape bunch"],
    "orange": ["orange", "navel orange", "valencia orange"],
    "cucumber": ["cucumber", "fresh cucumber", "green cucumber"],
    "carrot": ["carrot", "orange carrot", "fresh carrot"],
    "broccoli": ["broccoli", "broccoli floret", "green broccoli"],
    "pineapple": ["pineapple", "whole pineapple"],
    "mango": ["mango", "ripe mango", "yellow mango"],
    "papaya": ["papaya", "ripe papaya"],
    "avocado": ["avocado", "ripe avocado", "hass avocado"],
    "pear": ["pear", "green pear", "yellow pear"],
    "kiwi": ["kiwi", "kiwi fruit"],
    "peach": ["peach", "ripe peach"],
    "plum": ["plum", "purple plum"],
    "bell pepper": ["bell pepper", "red bell pepper", "green bell pepper", "yellow bell pepper"],
    "eggplant": ["eggplant", "purple eggplant", "aubergine"],
    "zucchini": ["zucchini", "green zucchini", "courgette"],
    "cabbage": ["cabbage", "green cabbage", "red cabbage"],
    "cauliflower": ["cauliflower", "white cauliflower"],
    "onion": ["onion", "yellow onion", "red onion", "white onion"],
    "garlic": ["garlic", "garlic bulb", "garlic clove"],
    "potato": ["potato", "brown potato", "russet potato"],
    "corn": ["corn", "corn cob", "sweet corn", "yellow corn"],
    "peas": ["peas", "green peas", "garden peas"],
    "beans": ["beans", "green beans", "string beans"],
    "spinach": ["spinach", "fresh spinach", "spinach leaves"],
}

# Vocabulário otimizado - removendo sinônimos muito próximos
FOOD_VOCAB = [
    # frutas
    "apple", "red apple", "green apple",
    "banana", "yellow banana",
    "orange", "tangerine",
    "strawberry", "grape", "watermelon", "pineapple", "mango", "papaya", 
    "avocado", "pear", "kiwi", "peach", "plum",
    # hortaliças
    "tomato", "red tomato", "cherry tomato",
    "lettuce", "romaine lettuce", "iceberg lettuce",
    "spinach", "fresh spinach",
    "cabbage", "broccoli", "cauliflower", "carrot", "cucumber", 
    "zucchini", "eggplant", "bell pepper",
    "onion", "garlic", "potato", "corn", "peas", "green beans"
]

# Modelo: YOLO-World
try:
    _detector = YOLO("yolov8x-world.pt")
    _detector.set_classes(FOOD_VOCAB)
except Exception:
    _detector = YOLO("yolov8n.pt")

COCO_FOODS = {"apple", "banana", "orange", "broccoli", "carrot"}

def _pil_to_cv(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def detect_food_regions(img: Image.Image, conf_thres: float = MIN_DET_CONF):
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
            x1, y1, x2, y2 = b.xyxy[0].tolist()
        else:
            continue
        name = names.get(cls_id, "object")
        if _detector.model_name.endswith("yolov8n.pt"):
            if name not in COCO_FOODS:
                continue
        out.append(([x1, y1, x2, y2], name, score))
    out.sort(key=lambda t: t[2], reverse=True)
    return out

def crop_bbox(img: Image.Image, bbox: List[float]) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w-1, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h-1, y1)))
    y2 = int(max(0, min(h, y2)))
    crop = img.crop((x1, y1, x2, y2))
    return crop

def crop_with_pad(img: Image.Image, bbox: List[float], pad: float = 0.08) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = bbox
    cw, ch = x2 - x1, y2 - y1
    x1 -= cw * pad
    y1 -= ch * pad
    x2 += cw * pad
    y2 += ch * pad
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w, x2))
    y2 = int(min(h, y2))
    return img.crop((x1, y1, x2, y2))

def classify_food_type_enhanced(img: Image.Image) -> Tuple[Optional[str], float, Dict[str, float]]:
    """
    Classifica o TIPO do alimento entre as chaves de FOOD_TYPES via CLIP (zero-shot).
    Retorna (tipo_em_inglês, confiança_softmax, dict_de_scores_top5).
    """

    img_emb = _embed_image_clip(img)
    
    # Templates de prompt
    templates = [
        "a close-up photo of {}",
        "a photo of fresh {}",
        "a high quality photo of {}",
        "{} on a white background",
    ]
    
    best_key, best_sim = None, -1.0
    sims = []
    keys = list(FOOD_TYPES.keys())
    
    for k in keys:
        max_sim = -1.0
        for template in templates:
            prompts = [template.format(p) for p in FOOD_TYPES[k]]
            txt = _embed_text_clip(prompts)
            sim = float((img_emb @ txt.T).max())
            max_sim = max(max_sim, sim)
        
        sims.append(max_sim)
        if max_sim > best_sim:
            best_sim, best_key = max_sim, k
    
    arr = np.array(sims, dtype=np.float32)
    probs = _softmax_temperature(arr, temperature=2.0)
    
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_scores = {keys[i]: float(probs[i]) for i in top5_idx}
    
    conf = float(probs[keys.index(best_key)]) if best_key is not None else 0.0
    
    return best_key, conf, top5_scores

def classify_food_type(img: Image.Image) -> Tuple[Optional[str], float]:
    """Versão simplificada para compatibilidade"""
    tipo, conf, _ = classify_food_type_enhanced(img)
    return tipo, conf

# ==========================
# 2) Classificador zero-shot
# ==========================
_CLIP_MODEL, _CLIP_PREP, _CLIP_TOKENIZER = None, None, None

def _load_clip():
    global _CLIP_MODEL, _CLIP_PREP, _CLIP_TOKENIZER
    if _CLIP_MODEL is None or _CLIP_PREP is None or _CLIP_TOKENIZER is None:
        backbone = os.getenv("CLIP_BACKBONE", "ViT-B-16")
        weights = os.getenv("CLIP_WEIGHTS", "openai")
        _CLIP_MODEL, _, _CLIP_PREP = open_clip.create_model_and_transforms(
            backbone, pretrained=weights, device=DEVICE
        )
        _CLIP_TOKENIZER = open_clip.get_tokenizer(backbone)
        _CLIP_MODEL.eval()

def _ensure_clip_loaded():
    if _CLIP_MODEL is None or _CLIP_PREP is None or _CLIP_TOKENIZER is None:
        _load_clip()

def _embed_image_clip(img: Image.Image):
    _ensure_clip_loaded()
    x = _CLIP_PREP(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = _CLIP_MODEL.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]

def _embed_text_clip(prompts: List[str]):
    _ensure_clip_loaded()
    tokens = _CLIP_TOKENIZER(prompts).to(DEVICE)
    with torch.no_grad():
        txt = _CLIP_MODEL.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy()

def _softmax(a: np.ndarray):
    m = a.max()
    e = np.exp(a - m)
    return e / (e.sum() + 1e-9)

def _softmax_temperature(a: np.ndarray, temperature: float = 1.0):
    #Softmax com temperature scaling para calibração
    a_scaled = a / temperature
    m = a_scaled.max()
    e = np.exp(a_scaled - m)
    return e / (e.sum() + 1e-9)

def classify_fresh_vs_rotten(img: Image.Image, detected_name: Optional[str]) -> Tuple[str, float]:

    #Retorna ("saudavel"|"doente", confidence)
    name = detected_name or "food item"
    
    # Prompts mais específicos para fresh vs rotten
    prompts_fresh = [
        f"a photo of fresh healthy {name}",
        f"a photo of ripe clean {name}",
        f"uma foto de {name} fresco e saudável",
    ]
    
    prompts_rotten = [
        f"a photo of rotten spoiled {name}",
        f"a photo of moldy decayed {name}",
        f"uma foto de {name} podre e estragado",
    ]
    
    im = _embed_image_clip(img)
    
    txt_fresh = _embed_text_clip(prompts_fresh)
    txt_rotten = _embed_text_clip(prompts_rotten)
    
    s_fresh = float((im @ txt_fresh.T).max())
    s_rottn = float((im @ txt_rotten.T).max())
    
    # Temperature scaling para calibração
    probs = _softmax_temperature(np.array([s_fresh, s_rottn]), temperature=1.5)
    
    idx = int(np.argmax(probs))
    label = "saudavel" if idx == 0 else "doente"
    conf = float(probs[idx])
    
    return label, round(conf, 4)

def _food_pt(name_en: Optional[str]) -> Optional[str]:
    if not name_en:
        return None
    m = {
        "apple": "maçã", "banana": "banana", "orange": "laranja", "strawberry": "morango", 
        "grape": "uva", "watermelon": "melancia", "pineapple": "abacaxi", "mango": "manga", 
        "papaya": "mamão", "avocado": "abacate", "pear": "pera", "kiwi": "kiwi", 
        "peach": "pêssego", "plum": "ameixa",
        "tomato": "tomate", "lettuce": "alface", "cabbage": "repolho", "broccoli": "brócolis", 
        "cauliflower": "couve-flor", "carrot": "cenoura", "cucumber": "pepino", 
        "zucchini": "abobrinha", "eggplant": "berinjela", "bell pepper": "pimentão", 
        "onion": "cebola", "garlic": "alho", "potato": "batata", "corn": "milho",
        "peas": "ervilha", "beans": "feijão", "spinach": "espinafre",
        "fruit": "fruta", "vegetable": "legume", "food": "alimento"
    }
    return m.get(name_en.lower(), name_en)

def _normalize_food_name(name: str) -> str:
    for base_name in FOOD_TYPES.keys():
        if base_name in name.lower():
            return base_name
    return name

def predict_image(img: Image.Image):
    """
    Retorna (label_pt, confidence, meta_dict)
    label: 'saudavel' | 'doente' | 'Não reconhecido'
    meta: {'food': <pt|en>, 'food_confidence': float, 'bbox': [x1,y1,x2,y2] | None, 'debug': {...}}
    """

    # 1) Detecção restrita a alimentos
    dets = [d for d in detect_food_regions(img, conf_thres=MIN_DET_CONF) if d[2] >= MIN_DET_CONF]
    if not dets:
        return "Não reconhecido", 0.0, {
            "food": None, 
            "food_confidence": 0.0, 
            "bbox": None,
            "debug": {"reason": "no_detection"}
        }

    (bbox, name_det_en, det_conf) = dets[0]
    name_det_normalized = _normalize_food_name(name_det_en)
    
    crop = crop_with_pad(img, bbox, pad=0.08)

    type_en, type_conf, top5_scores = classify_food_type_enhanced(crop)

    chosen_en = name_det_normalized
    chosen_conf = det_conf
    chosen_source = "yolo"
    
    # Se CLIP está muito confiante E discorda do YOLO, use CLIP
    if type_en and type_conf >= TYPE_MIN_CONF:
        if type_en != name_det_normalized:
            if type_conf > det_conf + 0.15: 
                chosen_en = type_en
                chosen_conf = type_conf
                chosen_source = "clip_override"
            else:
                chosen_source = "yolo_kept"
        else:
            chosen_conf = max(det_conf, type_conf)
            chosen_source = "consensus"

    label, class_conf = classify_fresh_vs_rotten(crop, chosen_en)

    if MIN_CLASS_CONF and class_conf < MIN_CLASS_CONF:
        return "Não reconhecido", class_conf, {
            "food": _food_pt(chosen_en),
            "food_confidence": float(chosen_conf),
            "bbox": [int(b) for b in bbox],
            "debug": {
                "yolo_name": name_det_en,
                "yolo_conf": float(det_conf),
                "clip_name": type_en,
                "clip_conf": float(type_conf) if type_en else 0.0,
                "chosen_source": chosen_source,
                "class_conf_too_low": True,
                "top5_clip": top5_scores
            }
        }

    return label, class_conf, {
        "food": _food_pt(chosen_en),
        "food_confidence": float(chosen_conf),
        "bbox": [int(b) for b in bbox],
        "debug": {
            "yolo_name": name_det_en,
            "yolo_conf": float(det_conf),
            "clip_name": type_en,
            "clip_conf": float(type_conf) if type_en else 0.0,
            "chosen_source": chosen_source,
            "top5_clip": top5_scores
        }
    }