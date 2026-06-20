# models/

Place the files exported from `model-training-facersys.ipynb` here:

- `face_recognition_resnet34.pth`   — the trained ResNet34 state_dict
- `class_names.json`                — index→name mapping, e.g. {"0": "Ahmed_Khan", "1": "Sara_Ali", ...}

The app works fine without these files (it just falls back to the
existing dlib-only recognizer), but you need both files present for the
CNN transfer-learning path to activate.
