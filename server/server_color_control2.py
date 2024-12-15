from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import cv2
import dlib
import numpy as np
from io import BytesIO
import uuid
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Store session images
session_images = {}
session_images_original = {}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Invalid image file"}

    session_id = str(uuid.uuid4())
    session_images[session_id] = image.copy()
    session_images_original[session_id] = image.copy()
    logger.info(f"Image uploaded successfully. Session ID: {session_id}")
    return {"session_id": session_id}

@app.post("/adjust-color/")
async def adjust_color(session_id: str = Form(...), feature: str = Form(...),
                       brightness: int = Form(...), color_r: int = Form(...),
                       color_g: int = Form(...), color_b: int = Form(...)):
    if session_id not in session_images:
        return {"error": "Invalid session ID"}

    image = session_images[session_id]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return {"error": "No face detected"}

    landmarks = predictor(gray, faces[0])
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    # Select the region
    region_points = list(range(48, 61)) if feature == "lip" else list(range(36, 48))
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([points[i] for i in region_points], np.int32)], (255, 255, 255))

    # Apply color and brightness
    region = cv2.bitwise_and(image, mask)
    color_layer = np.full_like(image, (color_b, color_g, color_r), dtype=np.uint8)
    adjusted = cv2.addWeighted(region, 0.7, color_layer, 0.3, 0)
    combined = cv2.add(adjusted, cv2.bitwise_and(image, cv2.bitwise_not(mask)))

    session_images[session_id] = combined
    _, buffer = cv2.imencode(".jpg", combined)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")

@app.post("/reset/")
async def reset_image(session_id: str = Form(...)):
    if session_id not in session_images_original:
        return {"error": "Invalid session ID"}
    session_images[session_id] = session_images_original[session_id].copy()
    _, buffer = cv2.imencode(".jpg", session_images[session_id])
    logger.info(f"Image reset for session ID: {session_id}")
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
