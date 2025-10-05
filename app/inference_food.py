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
MIN_DET_CONF = 0.40
MIN_CLASS_CONF = float(os.getenv("MIN_CLASS_CONF", "0"))  
MODEL_VERSION = "food-v2.0"  # Multi-stage verification

TYPE_MIN_CONF = float(os.getenv("TYPE_MIN_CONF", "0.65"))

# =========================
# 1) Detector de alimentos
# =========================
FOOD_TYPES = {
    "tomato": ["tomato", "red tomato", "ripe tomato", "cherry tomato", "roma tomato", "fresh tomato"],
    "apple": ["apple", "red apple", "green apple", "granny smith apple", "fuji apple"],
    "banana": ["banana", "ripe banana", "yellow banana", "green banana"],
    "lettuce": ["lettuce", "green lettuce", "fresh lettuce", "lettuce leaves", "leafy lettuce", "loose leaf lettuce"],
    "strawberry": ["strawberry", "fresh strawberry", "ripe strawberry"],
    "grape": ["grape", "green grape", "red grape", "grape bunch"],
    "orange": ["orange", "navel orange", "valencia orange", "whole orange"],
    "cucumber": ["cucumber", "fresh cucumber", "green cucumber", "whole cucumber"],
    "carrot": ["carrot", "orange carrot", "fresh carrot", "whole carrot"],
    "broccoli": ["broccoli", "broccoli floret", "broccoli crown", "green broccoli", "fresh broccoli"],
    "pineapple": ["pineapple", "whole pineapple", "fresh pineapple"],
    "mango": ["mango", "ripe mango", "yellow mango", "fresh mango"],
    "papaya": ["papaya", "ripe papaya", "fresh papaya"],
    "avocado": ["avocado", "ripe avocado", "hass avocado", "fresh avocado"],
    "pear": ["pear", "green pear", "yellow pear", "fresh pear"],
    "kiwi": ["kiwi", "kiwi fruit", "fresh kiwi"],
    "peach": ["peach", "ripe peach", "fresh peach"],
    "plum": ["plum", "purple plum", "fresh plum"],
    "bell pepper": ["bell pepper", "red bell pepper", "green bell pepper", "yellow bell pepper", "sweet pepper"],
    "eggplant": ["eggplant", "purple eggplant", "aubergine", "fresh eggplant"],
    "zucchini": ["zucchini", "green zucchini", "courgette", "fresh zucchini"],
    "cabbage": ["cabbage", "green cabbage", "red cabbage", "whole cabbage"],
    "cauliflower": ["cauliflower", "white cauliflower", "cauliflower head"],
    "onion": ["onion", "yellow onion", "red onion", "white onion", "whole onion"],
    "garlic": ["garlic", "garlic bulb", "garlic clove", "fresh garlic"],
    "potato": ["potato", "brown potato", "russet potato", "whole potato"],
    "corn": ["corn", "corn cob", "sweet corn", "yellow corn", "corn on the cob"],
    "peas": ["peas", "green peas", "garden peas", "fresh peas"],
    "beans": ["beans", "green beans", "string beans", "fresh beans"],
    "spinach": ["spinach", "fresh spinach", "spinach leaves", "baby spinach"],
}

# Características visuais para distinção (alface vs brócolis, etc)
VISUAL_DISCRIMINATORS = {
    "lettuce_vs_broccoli": {
        "lettuce": {
            "texture": "smooth and wavy leaves",
            "structure": "loose leafy structure",
            "shape": "wide flat leaves",
        },
        "broccoli": {
            "texture": "tight clustered florets",
            "structure": "compact tree-like head",
            "shape": "rounded dense crown",
        }
    },
    "tomato_vs_apple": {
        "tomato": {
            "texture": "smooth skin with slight shine",
            "features": "often has stem scar",
            "shape": "slightly irregular round",
        },
        "apple": {
            "texture": "waxy smooth skin",
            "features": "often has stem or indent",
            "shape": "uniformly round",
        }
    }
}

FOOD_VOCAB = [
    "apple", "red apple", "green apple",
    "banana", "yellow banana",
    "orange", "tangerine",
    "strawberry", "grape", "watermelon", "pineapple", "mango", "papaya", 
    "avocado", "pear", "kiwi", "peach", "plum",
    "tomato", "red tomato", "cherry tomato",
    "lettuce", "green lettuce", "fresh lettuce",
    "spinach", "fresh spinach",
    "cabbage", "broccoli", "cauliflower", "carrot", "cucumber", 
    "zucchini", "eggplant", "bell pepper",
    "onion", "garlic", "potato", "corn", "peas", "green beans"
]

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

# =========================
# 2) Análise de cor dominante
# =========================
def analyze_color_profile(img: Image.Image) -> Dict[str, float]:
    """
    Analisa perfil de cor para ajudar na distinção
    Retorna percentuais de cores dominantes
    """
    img_small = img.resize((100, 100))
    arr = np.array(img_small.convert("RGB"))
    
    # Calcular médias de canais
    r_mean = arr[:,:,0].mean() / 255.0
    g_mean = arr[:,:,1].mean() / 255.0
    b_mean = arr[:,:,2].mean() / 255.0
    
    # Calcular saturação média
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    saturation = hsv[:,:,1].mean() / 255.0
    value = hsv[:,:,2].mean() / 255.0
    
    return {
        "red_intensity": float(r_mean),
        "green_intensity": float(g_mean),
        "blue_intensity": float(b_mean),
        "saturation": float(saturation),
        "brightness": float(value),
    }

def analyze_texture_complexity(img: Image.Image) -> float:
    """
    Analisa complexidade de textura usando gradientes
    Valores maiores = textura mais complexa (ex: brócolis)
    Valores menores = textura mais lisa (ex: alface)
    """
    img_gray = img.convert("L")
    arr = np.array(img_gray)
    
    # Calcular gradientes
    gx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    complexity = magnitude.mean()
    
    return float(complexity)

# =========================
# 3) Classificador CLIP aprimorado
# =========================
_CLIP_MODEL, _CLIP_PREP, _CLIP_TOKENIZER = None, None, None

def _load_clip():
    global _CLIP_MODEL, _CLIP_PREP, _CLIP_TOKENIZER
    if _CLIP_MODEL is None:
        backbone = os.getenv("CLIP_BACKBONE", "ViT-B-16")
        weights = os.getenv("CLIP_WEIGHTS", "openai")
        _CLIP_MODEL, _, _CLIP_PREP = open_clip.create_model_and_transforms(
            backbone, pretrained=weights, device=DEVICE
        )
        _CLIP_TOKENIZER = open_clip.get_tokenizer(backbone)
        _CLIP_MODEL.eval()

def _ensure_clip_loaded():
    if _CLIP_MODEL is None:
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

def _softmax_temperature(a: np.ndarray, temperature: float = 1.0):
    a_scaled = a / temperature
    m = a_scaled.max()
    e = np.exp(a_scaled - m)
    return e / (e.sum() + 1e-9)

def classify_food_with_visual_verification(
    img: Image.Image, 
    yolo_prediction: str
) -> Tuple[str, float, Dict]:
    templates = [
        "a clear photo of {}",
        "a close-up photo of fresh {}",
        "a high quality photograph of {}",
        "{} on a plate",
        "whole {}",
    ]
    
    best_key, best_sim = None, -1.0
    sims = []
    keys = list(FOOD_TYPES.keys())
    
    for k in keys:
        max_sim = -1.0
        for template in templates:
            prompts = [template.format(p) for p in FOOD_TYPES[k]]
            txt = _embed_text_clip(prompts)
            img_emb = _embed_image_clip(img)
            sim = float((img_emb @ txt.T).max())
            max_sim = max(max_sim, sim)
        sims.append(max_sim)
        if max_sim > best_sim:
            best_sim, best_key = max_sim, k
    
    arr = np.array(sims, dtype=np.float32)
    probs = _softmax_temperature(arr, temperature=2.5)
    
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_scores = {keys[i]: float(probs[i]) for i in top5_idx}
    
    clip_pred = best_key
    clip_conf = float(probs[keys.index(best_key)]) if best_key else 0.0
    
    color_profile = analyze_color_profile(img)
    texture_complexity = analyze_texture_complexity(img)
    
    final_pred = clip_pred
    final_conf = clip_conf
    verification_applied = False
    
    if clip_pred in ["lettuce", "broccoli"] or yolo_prediction in ["lettuce", "broccoli"]:
        verification_applied = True
        
        is_broccoli_texture = texture_complexity > 15.0  # threshold empírico
        
        discriminative_prompts_lettuce = [
            "loose leafy lettuce with smooth wavy leaves",
            "fresh lettuce leaves spread out",
            "green lettuce with wide flat leaves",
            "alface com folhas largas e lisas",
        ]
        discriminative_prompts_broccoli = [
            "broccoli with tight clustered florets",
            "broccoli crown with dense tree-like structure",
            "compact broccoli head with small florets",
            "brócolis com floretes agrupados",
        ]
        
        img_emb = _embed_image_clip(img)
        txt_lettuce = _embed_text_clip(discriminative_prompts_lettuce)
        txt_broccoli = _embed_text_clip(discriminative_prompts_broccoli)
        
        sim_lettuce = float((img_emb @ txt_lettuce.T).max())
        sim_broccoli = float((img_emb @ txt_broccoli.T).max())

        score_lettuce = sim_lettuce * (1.0 if not is_broccoli_texture else 0.7)
        score_broccoli = sim_broccoli * (1.2 if is_broccoli_texture else 0.8)
        
        probs_disc = _softmax_temperature(
            np.array([score_lettuce, score_broccoli]), 
            temperature=1.0
        )
        
        if probs_disc[0] > probs_disc[1]:
            final_pred = "lettuce"
            final_conf = float(probs_disc[0])
        else:
            final_pred = "broccoli"
            final_conf = float(probs_disc[1])
    
    elif clip_pred in ["tomato", "apple"] or yolo_prediction in ["tomato", "apple"]:
        verification_applied = True
        
        red_dominant = color_profile["red_intensity"] > 0.6
        high_saturation = color_profile["saturation"] > 0.5
        
        discriminative_prompts_tomato = [
            "fresh red tomato with smooth skin",
            "ripe tomato with stem scar",
            "round tomato with bright red color",
            "tomate vermelho maduro",
        ]
        discriminative_prompts_apple = [
            "shiny apple with waxy skin",
            "round apple with uniform color",
            "fresh apple fruit",
            "maçã vermelha brilhante",
        ]
        
        img_emb = _embed_image_clip(img)
        txt_tomato = _embed_text_clip(discriminative_prompts_tomato)
        txt_apple = _embed_text_clip(discriminative_prompts_apple)
        
        sim_tomato = float((img_emb @ txt_tomato.T).max())
        sim_apple = float((img_emb @ txt_apple.T).max())
        
        # Boost para tomate se muito vermelho e saturado
        score_tomato = sim_tomato * (1.2 if (red_dominant and high_saturation) else 1.0)
        score_apple = sim_apple
        
        probs_disc = _softmax_temperature(
            np.array([score_tomato, score_apple]), 
            temperature=1.0
        )
        
        if probs_disc[0] > probs_disc[1]:
            final_pred = "tomato"
            final_conf = float(probs_disc[0])
        else:
            final_pred = "apple"
            final_conf = float(probs_disc[1])
    
    return final_pred, final_conf, {
        "top5_clip": top5_scores,
        "color_profile": color_profile,
        "texture_complexity": round(texture_complexity, 2),
        "verification_applied": verification_applied,
        "clip_initial": clip_pred,
        "clip_initial_conf": round(clip_conf, 4),
    }

def _normalize_food_name(name: str) -> str:
    for base_name in FOOD_TYPES.keys():
        if base_name in name.lower():
            return base_name
    return name

def classify_fresh_vs_rotten(img: Image.Image, detected_name: Optional[str]) -> Tuple[str, float]:
    name = detected_name or "food item"
    
    prompts_fresh = [
        f"a photo of fresh healthy {name}",
        f"a photo of ripe vibrant {name}",
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
    }
    return m.get(name_en.lower(), name_en)

def predict_image(img: Image.Image):
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

    type_en, type_conf, visual_analysis = classify_food_with_visual_verification(
        crop, name_det_normalized
    )

    chosen_en = name_det_normalized
    chosen_conf = det_conf
    chosen_source = "yolo"
    
    if visual_analysis.get("verification_applied", False):
        if type_conf >= 0.55:  
            chosen_en = type_en
            chosen_conf = type_conf
            chosen_source = "clip_verified"
    else:
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
                "clip_conf": float(type_conf),
                "chosen_source": chosen_source,
                "class_conf_too_low": True,
                **visual_analysis
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
            "clip_conf": float(type_conf),
            "chosen_source": chosen_source,
            **visual_analysis
        }
    }