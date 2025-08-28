import os, json, math, io
import torch, numpy as np
from PIL import Image
import open_clip

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL, _PREPROCESS, _TOKENIZER = None, None, None
MODEL_VERSION = "clip-vitb32-v1"

def _load_model():
    global _MODEL, _PREPROCESS, _TOKENIZER
    if _MODEL is None:
        _MODEL, _, _PREPROCESS = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai', device=DEVICE
        )
        _TOKENIZER = open_clip.get_tokenizer('ViT-B-32')
        _MODEL.eval()
_load_model()

STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
os.makedirs(STATE_DIR, exist_ok=True)
PROTOS_PATH = os.path.join(STATE_DIR, "prototypes.json")

# Estrutura
def _load_prototypes():
    if os.path.exists(PROTOS_PATH):
        with open(PROTOS_PATH, "r") as f:
            return json.load(f)
    return {"saudavel": [], "doente": []}

def _save_prototypes(data):
    with open(PROTOS_PATH, "w") as f:
        json.dump(data, f)

_PROTOTYPES = _load_prototypes()

def _img_to_tensor(img: Image.Image):
    return _PREPROCESS(img).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def _embed_image(img: Image.Image):
    x = _img_to_tensor(img)
    feats = _MODEL.encode_image(x)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()[0]  # (d,)

@torch.no_grad()
def _embed_text(prompts):
    tokens = _TOKENIZER(prompts).to(DEVICE)
    txt = _MODEL.encode_text(tokens)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy()  # (n, d)

def _softmax(xs):
    m = np.max(xs)
    e = np.exp(xs - m)
    return e / np.sum(e)

# --------- ZERO-SHOT ---------
_ZS_PROMPTS_EN = [
    "a photo of a healthy object",
    "a photo of a diseased object"
]
_ZS_PROMPTS_PT = [
    "uma imagem de algo saudável",
    "uma imagem de algo doente"
]
_ZS_TEXT_EMB = None

def _get_zeroshot_text_emb():
    global _ZS_TEXT_EMB
    if _ZS_TEXT_EMB is None:
        emb_en = _embed_text(_ZS_PROMPTS_EN)
        emb_pt = _embed_text(_ZS_PROMPTS_PT)
        _ZS_TEXT_EMB = (emb_en + emb_pt) / 2.0
    return _ZS_TEXT_EMB

def predict_zeroshot(img: Image.Image):
    img_emb = _embed_image(img)
    text_emb = _get_zeroshot_text_emb()  
    sims = (img_emb @ text_emb.T) 
    probs = _softmax(sims)
    idx = int(np.argmax(probs))
    label = "saudavel" if idx == 0 else "doente"
    conf = float(probs[idx])
    return label, round(conf, 4), {"mode": "zeroshot", "sims": sims.tolist()}

# --------- FEW-SHOT (protótipos) ---------
def add_prototype(label: str, img: Image.Image):
    label = label.lower().strip()
    if label not in ("saudavel", "doente"):
        raise ValueError("label deve ser 'saudavel' ou 'doente'")
    vec = _embed_image(img).tolist()
    _PROTOTYPES[label].append(vec)
    _save_prototypes(_PROTOTYPES)

def reset_prototypes():
    global _PROTOTYPES
    _PROTOTYPES = {"saudavel": [], "doente": []}
    _save_prototypes(_PROTOTYPES)

def stats_prototypes():
    return {k: len(v) for k, v in _PROTOTYPES.items()}

def _centroid(vecs):
    if not vecs:
        return None
    arr = np.array(vecs, dtype=np.float32)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    c = arr.mean(axis=0)
    c = c / np.linalg.norm(c)
    return c

def _prototype_centroids():
    c_s = _centroid(_PROTOTYPES["saudavel"])
    c_d = _centroid(_PROTOTYPES["doente"])
    return c_s, c_d

def predict_prototype(img: Image.Image):
    c_s, c_d = _prototype_centroids()
    if c_s is None or c_d is None:
        # fallback para zero-shot
        return predict_zeroshot(img)
    emb = _embed_image(img)
    sims = np.array([emb @ c_s, emb @ c_d]) 
    probs = _softmax(sims)
    idx = int(np.argmax(probs))
    label = "saudavel" if idx == 0 else "doente"
    conf = float(probs[idx])
    return label, round(conf, 4), {"mode": "fewshot", "sims": sims.tolist()}
