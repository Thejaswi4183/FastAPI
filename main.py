from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from datetime import datetime
import os
import gdown
from dotenv import load_dotenv

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

KERAS_FILE_ID = os.getenv("KERAS_FILE_ID")
PKL_FILE_ID = os.getenv("PKL_FILE_ID")

if not KERAS_FILE_ID or not PKL_FILE_ID:
    raise ValueError("Missing Google Drive File IDs in environment variables")

# Globals
model = None
tokenizer = None
max_len = 100

# Download model/tokenizer if missing
def download_if_missing(file_id, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)

# Lazy load model
def load_assets():
    global model, tokenizer
    if model is None:
        download_if_missing(KERAS_FILE_ID, "model.keras")
        download_if_missing(PKL_FILE_ID, "tokenizer.pkl")

        print("Loading model...")
        model = tf.keras.models.load_model("model.keras", compile=False)

        print("Loading tokenizer...")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        print("Model and tokenizer loaded!")

# Image preprocessing
def preprocess_image(contents):
    img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0)

# Text preprocessing
def preprocess_text(text: str):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=max_len, padding="post")

# Root endpoint
@app.get("/")
async def root():
    return {"status": "FastAPI is running!"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile, text: str = Form(...)):
    try:
        # Load model lazily
        if model is None or tokenizer is None:
            load_assets()

        # Validate file type
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})

        # Read file safely
        contents = await file.read()

        # Preprocess
        img = preprocess_image(contents)
        txt = preprocess_text(text)

        # Predict
        prediction = model.predict([img, txt])[0][0]
        age = datetime.now().year - prediction

        return {"age": f"{float(age):.2f}"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})