import random
from PIL import Image

LABELS = ["saudavel", "doente"]
MODEL_VERSION = "v0-mock"

def predict_image(img: Image.Image):
    # MOCKADISSIMO DOG
    confidence = round(random.uniform(0.70, 0.99), 4)
    label = random.choice(LABELS)

    return label, confidence

