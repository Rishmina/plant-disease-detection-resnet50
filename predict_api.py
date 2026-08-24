"""
Plant Disease Detection - FastAPI Backend
--------------------------------------------------
Serves predictions from your trained ONNX model over a REST API.

FOLDER SETUP REQUIRED (on your PC):
plant-disease-detection/
├── predict_api.py          <- this file
├── models/
│   └── plant_disease_model.onnx   <- the file you downloaded from Colab
└── ... (your other existing files)

HOW TO RUN (from your project folder, with venv activated):
    venv\\Scripts\\activate
    pip install fastapi uvicorn onnxruntime pillow numpy python-multipart
    uvicorn predict_api:app --reload

Then open http://127.0.0.1:8000/docs in your browser to test it
interactively — FastAPI auto-generates a UI where you can upload
an image and see the prediction, no extra code needed.
"""

import io
import time
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "models/plant_disease_model_v2.onnx"
IMG_SIZE = 224

# Must match the exact order printed during training/export
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]

# ImageNet normalization stats - must match training preprocessing exactly
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ============================================================
# APP + MODEL LOADING (runs once, when the server starts)
# ============================================================
app = FastAPI(
    title="Plant Disease Detection API",
    description="Upload a leaf image to get a disease prediction with confidence score.",
    version="1.0.0",
)

# Allow the React dashboard (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    session = None
    print(f"WARNING: Could not load model at startup: {e}")
    print("The API will start, but /predict will fail until the model file is in place.")


# ============================================================
# RESPONSE SCHEMA
# ============================================================
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    inference_time_ms: float
    all_class_probabilities: dict


# ============================================================
# HELPERS
# ============================================================
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw uploaded image bytes into the exact tensor format the model expects."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    array = np.array(image, dtype=np.float32) / 255.0       # scale to 0-1
    array = (array - MEAN) / STD                              # normalize like training
    array = array.transpose(2, 0, 1)                          # HWC -> CHW
    array = np.expand_dims(array, axis=0)                     # add batch dimension
    return array.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


# ============================================================
# ROUTES
# ============================================================
@app.get("/")
def root():
    return {
        "message": "Plant Disease Detection API is running.",
        "docs": "Visit /docs for interactive testing.",
        "model_loaded": session is not None,
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": session is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Make sure {MODEL_PATH} exists.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        input_tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    start_time = time.time()
    outputs = session.run(None, {input_name: input_tensor})
    inference_time_ms = (time.time() - start_time) * 1000

    logits = outputs[0][0]
    probabilities = softmax(logits)

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    all_probs = {
        CLASS_NAMES[i]: round(float(probabilities[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        inference_time_ms=round(inference_time_ms, 2),
        all_class_probabilities=all_probs,
    )
