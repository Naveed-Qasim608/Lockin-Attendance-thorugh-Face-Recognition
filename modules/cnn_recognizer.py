"""
cnn_recognizer.py
------------------
Transfer-learning face classifier (ResNet34, ImageNet-pretrained backbone,
fine-tuned classification head) — trained in model-training-facersys.ipynb.

This is used as the FIRST-PASS recognizer in the hybrid pipeline:
- If the CNN is confident AND the predicted person is enrolled locally,
  we trust it (this is the part that proves "we trained a model, not just
  used a pretrained one off the shelf").
- Otherwise we fall back to the existing dlib 128-D embedding matcher,
  so newly-enrolled people (who the CNN was never trained on) still work.

Required files (place after exporting from the Kaggle notebook):
    lockin/models/face_recognition_resnet34.pth
    lockin/models/class_names.json     -> {"0": "Ahmed_Khan", "1": "...", ...}
"""

import os
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet34
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'face_recognition_resnet34.pth')
CLASSES_PATH = os.path.join(MODELS_DIR, 'class_names.json')

# Minimum softmax confidence before we trust the CNN's prediction.
# Tune this after testing — start strict, loosen if it under-fires.
CNN_THRESHOLD = 0.60

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Must exactly match the preprocessing used during training in the notebook
# (Resize 224x224 + ToTensor, no normalization was used there — keep it
# identical or accuracy will silently degrade).
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

_model = None
_idx_to_class = None
_available = False


def _load():
    """Lazy-load the model + class mapping once. Safe to call repeatedly."""
    global _model, _idx_to_class, _available

    if _model is not None or _available is False and _idx_to_class is not None:
        return

    if not (os.path.exists(WEIGHTS_PATH) and os.path.exists(CLASSES_PATH)):
        logger.warning(
            "CNN model files not found in lockin/models/ — "
            "hybrid recognizer will run in dlib-only mode until you add "
            "face_recognition_resnet34.pth and class_names.json."
        )
        _available = False
        return

    try:
        with open(CLASSES_PATH, 'r') as f:
            _idx_to_class = json.load(f)

        num_classes = len(_idx_to_class)

        model = resnet34(weights=None)  # architecture only — real weights loaded next
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        _model = model
        _available = True
        logger.info(f"CNN transfer-learning model loaded ({num_classes} classes, device={DEVICE})")
    except Exception as e:
        logger.error(f"Failed to load CNN model: {e}")
        _available = False


def is_available() -> bool:
    _load()
    return _available


def crop_face(img_array: np.ndarray, bbox, padding: float = 0.25) -> np.ndarray:
    """
    Crop a face region from the full frame using a dlib-style bbox
    (top, right, bottom, left), with a bit of padding so the CNN sees
    a similar framing to the enrollment captures it was trained on.
    """
    top, right, bottom, left = bbox
    h, w = bottom - top, right - left
    pad_h, pad_w = int(h * padding), int(w * padding)

    top    = max(0, top - pad_h)
    left   = max(0, left - pad_w)
    bottom = min(img_array.shape[0], bottom + pad_h)
    right  = min(img_array.shape[1], right + pad_w)

    return img_array[top:bottom, left:right]


def cnn_predict(face_crop_rgb: np.ndarray):
    """
    Run the trained ResNet34 on a cropped RGB face image.

    Returns (name, confidence) or (None, 0.0) if unavailable / failed.
    """
    _load()
    if not _available or face_crop_rgb.size == 0:
        return None, 0.0

    try:
        img = Image.fromarray(face_crop_rgb)
        tensor = _TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = _model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)

        name = _idx_to_class[str(idx.item())]
        return name, float(conf.item())
    except Exception as e:
        logger.warning(f"CNN prediction failed: {e}")
        return None, 0.0
