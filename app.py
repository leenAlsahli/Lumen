"""LUMEN — FastAPI backend serving the trained CNN and the web interface."""

import base64, hashlib, io, os, threading, time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = 128
MAX_BYTES = 10 * 1024 * 1024
PREVIEW = os.getenv("LUMEN_PREVIEW") == "1"

state = {"model": None, "cam": None, "name": None, "error": None}
lock = threading.Lock()


def find_model_file():
    for pattern in ("*.keras", "*.h5"):
        files = sorted(MODEL_DIR.glob(pattern))
        if files:
            return files[0]
    return None


def load_model():
    path = find_model_file()

    if path is None:
        state["error"] = (
            "No model found. Place your .keras or .h5 file inside the 'model/' folder."
        )
        return

    try:
        import tensorflow as tf
    except ImportError:
        state["error"] = "TensorFlow is not installed. Run: pip install tensorflow"
        return

    try:
        model = tf.keras.models.load_model(path, compile=False)

        model(
            np.zeros(
                (1, IMG_SIZE, IMG_SIZE, 3),
                dtype="float32"
            )
        )

        state.update(
            model=model,
            name=path.name,
            error=None
        )

        try:
            inp = tf.keras.Input(
                (IMG_SIZE, IMG_SIZE, 3)
            )

            last_conv = [
                l
                for l in model.layers
                if isinstance(l, tf.keras.layers.Conv2D)
            ][-1]

            x = inp
            conv = None

            for layer in model.layers:
                x = layer(x)

                if layer is last_conv:
                    conv = x

            state["cam"] = tf.keras.Model(
                inp,
                [conv, x]
            )

        except Exception:
            state["cam"] = None

    except Exception as exc:
        state["error"] = (
            f"Could not load '{path.name}': {exc}"
        )


@asynccontextmanager
async def lifespan(app):
    load_model()
    yield


app = FastAPI(
    title="LUMEN",
    version="1.0.0",
    lifespan=lifespan
)

app.mount(
    "/static",
    StaticFiles(directory=BASE / "static"),
    name="static"
)


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Mimic training:
    dataset images are ~150 px,
    so big photos are first shrunk
    with a proper filter,
    then resized to 128x128
    and scaled to 0-1.
    """

    img = img.copy()

    img.thumbnail(
        (150, 150),
        Image.LANCZOS
    )

    arr = np.asarray(
        img.resize(
            (IMG_SIZE, IMG_SIZE),
            Image.NEAREST
        ),
        dtype="float32"
    ) / 255.0

    return arr[None, ...]


def grad_cam(
    x: np.ndarray,
    class_idx: int,
    img: Image.Image
):
    """
    Return a data-URL PNG
    with the Grad-CAM heat-map
    blended over the photo.
    """

    import tensorflow as tf

    if state["cam"] is None:
        return None

    try:

        with tf.GradientTape() as tape:

            conv, preds = state["cam"](
                x,
                training=False
            )

            score = preds[:, class_idx]

        grads = tape.gradient(
            score,
            conv
        )

        weights = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        cam = tf.nn.relu(
            tf.reduce_sum(
                conv * weights[:, None, None, :],
                axis=-1
            )
        )[0].numpy()

        cam = cam / (
            cam.max() + 1e-8
        )

        base = img.copy()

        base.thumbnail(
            (640, 640)
        )

        heat = Image.fromarray(
            (cam * 255).astype("uint8")
        ).resize(
            base.size,
            Image.BICUBIC
        )

        h = (
            np.asarray(
                heat,
                dtype="float32"
            )[..., None]
            / 255.0
        )

        low = np.array(
            [24, 54, 50],
            "float32"
        )

        high = np.array(
            [255, 196, 96],
            "float32"
        )

        colour = (
            low
            + (high - low) * h
        )

        out = (
            np.asarray(
                base,
                dtype="float32"
            )
            * (1 - 0.62 * h)
            + colour
            * (0.62 * h)
        )

        buf = io.BytesIO()

        Image.fromarray(
            out.clip(
                0,
                255
            ).astype("uint8")
        ).save(
            buf,
            "PNG"
        )

        return (
            "data:image/png;base64,"
            + base64.b64encode(
                buf.getvalue()
            ).decode()
        )

    except Exception:
        return None


@app.get("/")
def index():
    return FileResponse(
        BASE / "static" / "index.html"
    )


@app.head("/")
def head_index():
    return Response(
        status_code=200
    )


@app.get("/api/health")
def health():

    status = (
        "ok"
        if state["model"]
        else (
            "preview"
            if PREVIEW
            else "no_model"
        )
    )

    return {
        "status": status,
        "model": state["name"],
        "error": state["error"],
        "classes": CLASSES
    }


@app.head("/api/health")
def head_health():
    return Response(
        status_code=200
    )


@app.post("/api/predict")
def predict(
    file: UploadFile = File(...)
):

    if (
        state["model"] is None
        and not PREVIEW
    ):
        raise HTTPException(
            503,
            state["error"]
            or "Model is not loaded."
        )

    raw = file.file.read(
        MAX_BYTES + 1
    )

    if len(raw) > MAX_BYTES:
        raise HTTPException(
            413,
            "Image is larger than 10 MB."
        )

    try:

        img = ImageOps.exif_transpose(
            Image.open(
                io.BytesIO(raw)
            )
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError
    ):

        raise HTTPException(
            400,
            "This file is not a valid image."
        )


    if state["model"] is None:

        rng = np.random.default_rng(
            int(
                hashlib.md5(
                    raw
                ).hexdigest()[:8],
                16
            )
        )

        probs = rng.dirichlet(
            np.ones(
                len(CLASSES)
            ) * .25
        )

        order = np.argsort(
            probs
        )[::-1]

        return {

            "label":
                CLASSES[
                    int(order[0])
                ],

            "confidence":
                round(
                    float(
                        probs[
                            order[0]
                        ]
                    ) * 100,
                    1
                ),

            "top3": [
                {
                    "label":
                        CLASSES[i],

                    "probability":
                        round(
                            float(
                                probs[i]
                            ) * 100,
                            1
                        )
                }
                for i in order[:3]
            ],

            "all": {
                CLASSES[i]:
                    round(
                        float(
                            probs[i]
                        ) * 100,
                        2
                    )
                for i
                in range(
                    len(CLASSES)
                )
            },

            "gradcam": None,
            "inference_ms": 0,
            "preview": True
        }


    x = preprocess(img)

    t0 = time.perf_counter()


    with lock:

        probs = state["model"](
            x,
            training=False
        ).numpy()[0]

        top = int(
            np.argmax(
                probs
            )
        )

        cam = grad_cam(
            x,
            top,
            img
        )


    ms = round(
        (
            time.perf_counter()
            - t0
        )
        * 1000
    )


    order = np.argsort(
        probs
    )[::-1]


    return {

        "label":
            CLASSES[top],

        "confidence":
            round(
                float(
                    probs[top]
                ) * 100,
                1
            ),

        "top3": [
            {
                "label":
                    CLASSES[i],

                "probability":
                    round(
                        float(
                            probs[i]
                        ) * 100,
                        1
                    )
            }
            for i
            in order[:3]
        ],

        "all": {
            CLASSES[i]:
                round(
                    float(
                        probs[i]
                    ) * 100,
                    2
                )
            for i
            in range(
                len(CLASSES)
            )
        },

        "gradcam":
            cam,

        "inference_ms":
            ms,

        "model":
            state["name"]
    }
