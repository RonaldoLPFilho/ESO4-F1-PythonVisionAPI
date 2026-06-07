"""
Vision Core — detecção e classificação de alimentos.

Pipeline (food-v3):
  1. YOLO-World (vocabulário aberto) localiza o alimento na cena -> bounding box.
     É usado APENAS para recortar a região; a identidade vem do CLIP.
  2. CLIP zero-shot com *prompt ensembling* identifica o tipo de alimento.
     - Todos os embeddings de texto são pré-computados UMA vez (cache),
       então cada frame faz apenas 1-2 forward-passes de imagem (rápido).
     - Classes "distractoras" (pessoa, mão, fundo, objeto) permitem rejeitar
       cenas sem alimento em vez de "chutar" uma fruta.
  3. Se um alimento é reconhecido com confiança, CLIP classifica fresco vs. estragado.
     - Só rotula "doente" quando há evidência clara; empate -> "saudavel".

Decisões de projeto importantes:
  - Quando NENHUM alimento é reconhecido com confiança, retornamos
    "Não reconhecido" e NÃO inventamos um veredito de saúde (evita o
    famigerado "Doente 50%").
"""

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
import torch
import open_clip

try:
    from ultralytics import YOLO
    _YOLO_LIB = True
except Exception:
    _YOLO_LIB = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION = "food-v3.0-clip-ensemble"

# ----------------------------------------------------------------------------
# Thresholds (ajustáveis por variável de ambiente)
# ----------------------------------------------------------------------------
# Confiança mínima do detector YOLO-World (baixa de propósito: YOLO-World
# costuma ser subconfiante em enquadramentos incomuns, ex. fruta na mão).
MIN_DET_CONF = float(os.getenv("MIN_DET_CONF", "0.08"))
# Similaridade de cosseno mínima (absoluta) para aceitar um alimento quando o
# YOLO já localizou a região (a caixa do YOLO-World já é um alimento).
# CLIP ViT-B/16 (openai): match correto ~0.25-0.35, errado ~0.15-0.20.
FOOD_ABS_FLOOR = float(os.getenv("FOOD_ABS_FLOOR", "0.195"))
# Piso (mais alto) para o caminho de fallback, quando o YOLO NÃO detectou nada.
# Evita "alucinar" alimento a partir de plantas/fundo verde da cena.
FALLBACK_FLOOR = float(os.getenv("FALLBACK_FLOOR", "0.24"))
# Probabilidade mínima de "podre" para rotular "doente" (caso contrário saudavel).
DISEASE_MIN_CONF = float(os.getenv("DISEASE_MIN_CONF", "0.62"))

# ----------------------------------------------------------------------------
# Vocabulário de alimentos: chave canônica -> sinônimos/descrições em inglês
# (CLIP é treinado majoritariamente em inglês; sinônimos ricos melhoram muito).
# ----------------------------------------------------------------------------
FOOD_TYPES: Dict[str, List[str]] = {
    "tomato": ["tomato", "red tomato", "ripe tomato", "cherry tomato", "fresh tomato"],
    "apple": ["apple", "red apple", "green apple", "fuji apple"],
    "banana": ["banana", "a banana", "ripe yellow banana", "bunch of bananas", "green banana"],
    "lettuce": ["lettuce", "green lettuce", "fresh lettuce leaves", "leafy lettuce", "head of lettuce", "romaine lettuce"],
    "strawberry": ["strawberry", "fresh strawberry", "ripe strawberry"],
    "grape": ["grape", "green grapes", "red grapes", "bunch of grapes"],
    "orange": ["orange", "navel orange", "whole orange", "citrus orange"],
    "cucumber": ["cucumber", "fresh cucumber", "green cucumber"],
    "carrot": ["carrot", "a carrot", "orange carrot", "fresh carrot root", "whole carrot", "carrot vegetable"],
    "broccoli": ["broccoli", "broccoli floret", "broccoli crown", "green broccoli head"],
    "pineapple": ["pineapple", "whole pineapple", "fresh pineapple"],
    "mango": ["mango", "ripe mango", "yellow mango"],
    "papaya": ["papaya", "ripe papaya"],
    "avocado": ["avocado", "ripe avocado", "hass avocado"],
    "pear": ["pear", "green pear", "fresh pear"],
    "kiwi": ["kiwi", "kiwi fruit"],
    "peach": ["peach", "ripe peach"],
    "plum": ["plum", "purple plum"],
    "bell pepper": ["bell pepper", "red bell pepper", "green bell pepper", "sweet pepper"],
    "eggplant": ["eggplant", "purple eggplant", "aubergine"],
    "zucchini": ["zucchini", "green zucchini", "courgette"],
    "cabbage": ["cabbage", "green cabbage", "whole cabbage"],
    "cauliflower": ["cauliflower", "white cauliflower head"],
    "onion": ["onion", "yellow onion", "red onion", "whole onion"],
    "garlic": ["garlic", "garlic bulb", "garlic clove"],
    "potato": ["potato", "brown potato", "whole potato"],
    "corn": ["corn", "corn cob", "corn on the cob"],
    "peas": ["peas", "green peas", "garden peas"],
    "beans": ["green beans", "string beans"],
    "spinach": ["spinach", "fresh spinach leaves"],
}

# Classes "distractoras": NÃO são alimentos. Se uma delas vence o argmax,
# a cena provavelmente não tem um alimento em destaque -> rejeita.
DISTRACTORS: Dict[str, List[str]] = {
    "person": ["a person", "a human face", "a man", "a woman", "people"],
    "hand": ["a hand", "a human hand holding something", "fingers"],
    "background": ["a room interior", "a bookshelf", "furniture", "a wall", "indoor background"],
    "object": ["a random object", "an electronic gadget", "a piece of clothing", "something that is not food"],
}

# Templates aplicados a cada sinônimo (prompt ensembling).
TEMPLATES = [
    "a photo of a {}.",
    "a close-up photo of a {}.",
    "a photo of a single {}.",
    "a hand holding a {}.",
    "a {} on a white background.",
    "a fresh {}.",
    "an image of a {}.",
]

# Tradução para PT-BR (apenas alimentos).
_PT = {
    "apple": "maçã", "banana": "banana", "orange": "laranja", "strawberry": "morango",
    "grape": "uva", "pineapple": "abacaxi", "mango": "manga", "papaya": "mamão",
    "avocado": "abacate", "pear": "pera", "kiwi": "kiwi", "peach": "pêssego", "plum": "ameixa",
    "tomato": "tomate", "lettuce": "alface", "cabbage": "repolho", "broccoli": "brócolis",
    "cauliflower": "couve-flor", "carrot": "cenoura", "cucumber": "pepino",
    "zucchini": "abobrinha", "eggplant": "berinjela", "bell pepper": "pimentão",
    "onion": "cebola", "garlic": "alho", "potato": "batata", "corn": "milho",
    "peas": "ervilha", "beans": "feijão", "spinach": "espinafre",
}


def _food_pt(name_en: Optional[str]) -> Optional[str]:
    if not name_en:
        return None
    return _PT.get(name_en.lower(), name_en)


# ----------------------------------------------------------------------------
# CLIP — carregamento e cache de embeddings de texto
# ----------------------------------------------------------------------------
_CLIP_MODEL = None
_CLIP_PREP = None
_CLIP_TOKENIZER = None
_LOGIT_SCALE = 100.0

# Matrizes de embedding (preenchidas uma única vez).
_FOOD_KEYS: List[str] = list(FOOD_TYPES.keys())
_FOOD_EMB: Optional[np.ndarray] = None        # (num_foods, d)
_DISTRACTOR_KEYS: List[str] = list(DISTRACTORS.keys())
_DISTRACTOR_EMB: Optional[np.ndarray] = None  # (num_distractors, d)


def _load_clip():
    global _CLIP_MODEL, _CLIP_PREP, _CLIP_TOKENIZER, _LOGIT_SCALE
    if _CLIP_MODEL is not None:
        return
    backbone = os.getenv("CLIP_BACKBONE", "ViT-B-16")
    weights = os.getenv("CLIP_WEIGHTS", "openai")
    _CLIP_MODEL, _, _CLIP_PREP = open_clip.create_model_and_transforms(
        backbone, pretrained=weights, device=DEVICE
    )
    _CLIP_TOKENIZER = open_clip.get_tokenizer(backbone)
    _CLIP_MODEL.eval()
    try:
        _LOGIT_SCALE = float(_CLIP_MODEL.logit_scale.exp().item())
    except Exception:
        _LOGIT_SCALE = 100.0


@torch.no_grad()
def _embed_texts(prompts: List[str]) -> np.ndarray:
    tokens = _CLIP_TOKENIZER(prompts).to(DEVICE)
    txt = _CLIP_MODEL.encode_text(tokens)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy()


def _build_class_embedding(synonyms: List[str]) -> np.ndarray:
    """Prompt ensembling: média (normalizada) sobre todos os templates×sinônimos."""
    prompts = [t.format(s) for s in synonyms for t in TEMPLATES]
    embs = _embed_texts(prompts)          # (N, d)
    mean = embs.mean(axis=0)              # (d,)
    mean = mean / (np.linalg.norm(mean) + 1e-9)
    return mean.astype(np.float32)


def _ensure_ready():
    """Carrega CLIP e pré-computa TODOS os embeddings de texto (uma vez)."""
    global _FOOD_EMB, _DISTRACTOR_EMB
    _load_clip()
    if _FOOD_EMB is None:
        _FOOD_EMB = np.stack([_build_class_embedding(FOOD_TYPES[k]) for k in _FOOD_KEYS])
    if _DISTRACTOR_EMB is None:
        _DISTRACTOR_EMB = np.stack([_build_class_embedding(DISTRACTORS[k]) for k in _DISTRACTOR_KEYS])


@torch.no_grad()
def _embed_images(imgs: List[Image.Image]) -> np.ndarray:
    """Retorna embeddings de imagem normalizados (n, d)."""
    batch = torch.stack([_CLIP_PREP(im) for im in imgs]).to(DEVICE)
    feats = _CLIP_MODEL.encode_image(batch)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


# ----------------------------------------------------------------------------
# Detector YOLO-World (apenas localização)
# ----------------------------------------------------------------------------
_FOOD_VOCAB = sorted({s for syn in FOOD_TYPES.values() for s in syn})
_detector = None
_detector_kind = "none"


def _load_detector():
    global _detector, _detector_kind
    if _detector is not None or not _YOLO_LIB:
        return
    try:
        _detector = YOLO("yolov8x-world.pt")
        _detector.set_classes(_FOOD_VOCAB)
        _detector_kind = "world"
    except Exception:
        try:
            _detector = YOLO("yolov8n.pt")
            _detector_kind = "coco"
        except Exception:
            _detector = None
            _detector_kind = "none"


_COCO_FOODS = {"apple", "banana", "orange", "broccoli", "carrot"}


def _detect_box(img: Image.Image) -> Optional[List[float]]:
    """Retorna a melhor bbox [x1,y1,x2,y2] de um alimento, ou None."""
    _load_detector()
    if _detector is None:
        return None
    try:
        res = _detector.predict(source=np.array(img.convert("RGB"))[:, :, ::-1],
                                verbose=False, conf=MIN_DET_CONF)[0]
    except Exception:
        return None

    names = res.names if isinstance(getattr(res, "names", None), dict) else {}
    best, best_score = None, -1.0
    for b in res.boxes:
        score = float(b.conf.item()) if b.conf is not None else 0.0
        cls_id = int(b.cls.item()) if b.cls is not None else -1
        name = names.get(cls_id, "")
        if _detector_kind == "coco" and name not in _COCO_FOODS:
            continue
        if score > best_score and hasattr(b, "xyxy"):
            best_score = score
            best = b.xyxy[0].tolist()
    return best


def _crop_pad(img: Image.Image, bbox: List[float], pad: float = 0.12) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = bbox
    cw, ch = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - cw * pad))
    y1 = int(max(0, y1 - ch * pad))
    x2 = int(min(w, x2 + cw * pad))
    y2 = int(min(h, y2 + ch * pad))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def _center_square(img: Image.Image, frac: float = 0.7) -> Image.Image:
    """Recorte quadrado central — objetos segurados costumam estar no centro."""
    w, h = img.size
    side = int(min(w, h) * frac)
    cx, cy = w // 2, h // 2
    return img.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))


# ----------------------------------------------------------------------------
# Identificação do alimento (CLIP zero-shot multi-crop)
# ----------------------------------------------------------------------------
def _identify_food(views: List[Image.Image]) -> Dict:
    """
    Classifica entre alimentos + distractores usando várias views (TTA).
    Para cada classe usa o MELHOR cosseno entre as views.
    """
    _ensure_ready()
    img_emb = _embed_images(views)                       # (n, d)
    food_sims = (img_emb @ _FOOD_EMB.T).max(axis=0)      # (num_foods,)
    distr_sims = (img_emb @ _DISTRACTOR_EMB.T).max(axis=0)  # (num_distractors,)

    food_idx = int(np.argmax(food_sims))
    best_food = _FOOD_KEYS[food_idx]
    best_food_cos = float(food_sims[food_idx])

    distr_idx = int(np.argmax(distr_sims))
    best_distr = _DISTRACTOR_KEYS[distr_idx]
    best_distr_cos = float(distr_sims[distr_idx])

    # Confiança relativa entre os alimentos (softmax calibrado pelo logit_scale).
    logits = food_sims * _LOGIT_SCALE
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    food_conf = float(probs[food_idx])

    order = np.argsort(food_sims)[-3:][::-1]
    top3 = {_FOOD_KEYS[i]: round(float(food_sims[i]), 4) for i in order}

    return {
        "food": best_food,
        "food_cos": best_food_cos,
        "food_conf": food_conf,
        "distractor": best_distr,
        "distractor_cos": best_distr_cos,
        "top3": top3,
    }


# ----------------------------------------------------------------------------
# Fresco vs. estragado
# ----------------------------------------------------------------------------
def _fresh_vs_rotten(img: Image.Image, food_en: str) -> Tuple[str, float]:
    _ensure_ready()
    n = food_en
    # Conjuntos balanceados; usamos a MÉDIA das similaridades (não o máximo)
    # para reduzir ruído — um único prompt que casa por acaso não decide.
    fresh = [
        f"a photo of a fresh {n}",
        f"a fresh ripe healthy {n}",
        f"a {n} in good condition",
        f"a clean undamaged {n}",
        f"uma foto de {n} fresco e saudável",
    ]
    rotten = [
        f"a photo of a rotten {n}",
        f"a moldy spoiled {n}",
        f"a {n} with brown spots, mold and decay",
        f"a damaged decaying {n}",
        f"uma foto de {n} podre e estragado",
    ]
    im = _embed_images([img])[0]
    s_fresh = float((im @ _embed_texts(fresh).T).mean())
    s_rotten = float((im @ _embed_texts(rotten).T).mean())

    logits = np.array([s_fresh, s_rotten]) * _LOGIT_SCALE
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    p_rotten = float(probs[1])

    # Só acusa "doente" com evidência clara; do contrário, saudável.
    if p_rotten >= DISEASE_MIN_CONF:
        return "doente", round(p_rotten, 4)
    # Confiança do rótulo saudável = certeza de NÃO estar estragado.
    return "saudavel", round(1.0 - p_rotten, 4)


# ----------------------------------------------------------------------------
# API pública
# ----------------------------------------------------------------------------
def _not_recognized(debug: Dict) -> Tuple[str, float, Dict]:
    return "Não reconhecido", 0.0, {
        "food": None, "food_confidence": 0.0, "bbox": None, "debug": debug
    }


def predict_image(img: Image.Image) -> Tuple[str, float, Dict]:
    img = img.convert("RGB")
    bbox = _detect_box(img)

    # Monta as "views" para o CLIP.
    if bbox is not None:
        crop = _crop_pad(img, bbox)
        # Crop justo (sem padding) p/ avaliar saúde: menos fundo = menos falso "podre".
        health_crop = _crop_pad(img, bbox, pad=0.0)
        views = [crop]
        bbox_out = [int(v) for v in bbox]
        from_yolo = True
    else:
        # Sem detecção: imagem inteira + recorte central (ajuda objeto segurado).
        crop = _center_square(img)
        health_crop = crop
        views = [img, crop]
        bbox_out = None
        from_yolo = False

    info = _identify_food(views)

    # Critério de aceitação:
    #  - Com box do YOLO: a região já é um alimento -> basta o piso absoluto.
    #  - Sem box (fallback): exige piso mais alto E que o alimento supere a
    #    melhor classe distractora (pessoa/mão/fundo) — senão é cena sem comida.
    if from_yolo:
        passes_floor = info["food_cos"] >= FOOD_ABS_FLOOR
        beats_distractor = True
    else:
        passes_floor = info["food_cos"] >= FALLBACK_FLOOR
        beats_distractor = info["food_cos"] >= info["distractor_cos"]

    debug = {
        "detector": _detector_kind,
        "from_yolo": from_yolo,
        "food_cos": round(info["food_cos"], 4),
        "distractor": info["distractor"],
        "distractor_cos": round(info["distractor_cos"], 4),
        "top3": info["top3"],
    }

    if not (passes_floor and beats_distractor):
        debug["reason"] = "no_confident_food"
        return _not_recognized(debug)

    food_en = info["food"]
    label, class_conf = _fresh_vs_rotten(health_crop, food_en)

    return label, class_conf, {
        "food": _food_pt(food_en),
        "food_confidence": round(info["food_conf"], 4),
        "bbox": bbox_out,
        "debug": {**debug, "food_en": food_en},
    }
