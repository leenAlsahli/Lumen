
# LUMEN — Natural Scene Classification

An AI-powered computer vision system that classifies natural scenes using deep learning.
A CNN trained from scratch on the Intel Image Classification dataset (6 classes, **89.43 % test accuracy**), wrapped in a FastAPI backend and a hand-built HTML/CSS/JS interface.

**Features:** drag-and-drop / paste upload · real predictions with top-3 probabilities · Grad-CAM attention map · model architecture and training details · accuracy / loss curves · confusion matrix.

## Project structure

```
lumen/
├── app.py                 # FastAPI backend: loads the model, /api/predict, serves the UI
├── requirements.txt
├── Dockerfile             # for Hugging Face Spaces / any container host
├── model/                 # ← PUT YOUR MODEL FILE HERE (.keras or .h5)
└── static/
    ├── index.html  style.css  app.js
    ├── img/               # logo files
    └── data/metrics.json  # history, confusion matrix, scores (from your notebook)
```

## 1. Model and evaluation files

The trained model is already in `model/lumen_model.keras`. To replace it, save from Colab and overwrite the file:

```python
model.save("lumen_model.keras")   # `model` holds the best epoch weights after EarlyStopping
```

**Confusion matrix and per-class scores** are loaded from `static/data/evaluation.json`. Generate it in Colab (after the test generator cells) and put the file in `static/data/`:

```python
import json, numpy as np
from sklearn.metrics import confusion_matrix, classification_report
test_gen_final = test_datagen.flow_from_directory(TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
y_pred = np.argmax(model.predict(test_gen_final, verbose=0), axis=1)
y_true = test_gen_final.classes
rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
json.dump({"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
           "report": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                          "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])} for c in class_names},
           "macro_f1": rep["macro avg"]["f1-score"]}, open("evaluation.json", "w"))
from google.colab import files; files.download("evaluation.json")
```

Until this file exists, the confusion-matrix block is hidden.

## 2. Run locally

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open http://localhost:8000. `GET /api/health` tells you whether the model was loaded.

### Preview the interface without a model

Only the web dependencies are needed (no TensorFlow):

```bash
pip install fastapi "uvicorn[standard]" python-multipart pillow numpy
LUMEN_PREVIEW=1 uvicorn app:app --port 8000        # Windows PowerShell: $env:LUMEN_PREVIEW=1; uvicorn app:app --port 8000
```

Uploads then return **simulated** results, clearly labelled "Preview mode". Never deploy with this variable set.

## 3. Deploy online

FastAPI serves your custom HTML, so use a container host (Streamlit Cloud only runs Streamlit apps).

**Hugging Face Spaces (free, recommended)**
1. Create a new Space → SDK **Docker** (blank).
2. Add this header at the very top of `README.md` in the Space repo:
   ```
   ---
   title: Lumen
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push the project including `model/lumen_model.keras` (~8 MB, no Git LFS needed):
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/lumen
   git push space main
   ```
The `Dockerfile` already listens on port 7860.

**Render / Railway / Fly.io** — same repo, start command:
`uvicorn app:app --host 0.0.0.0 --port $PORT`

## Notes

- **Preprocessing matches training:** RGB → shrink to ≤150 px (dataset native size) → 128×128 (nearest-neighbour, the Keras `flow_from_directory` default) → ÷255. Class order is alphabetical: buildings, forest, glacier, mountain, sea, street.
- Update the figures on the page by editing `static/data/metrics.json`.
- The "25K+ images" figure is the size of the full Intel dataset; your notebook trained on 11,932 images, validated on 2,102 and tested on 3,000 (shown in *About the Model*).
