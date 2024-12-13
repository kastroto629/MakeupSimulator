from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import cv2
import dlib
import numpy as np
from io import BytesIO

app = FastAPI()

# Load dlib's pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def apply_adjustments(image, landmarks, region_points, brightness, color):
    mask = np.zeros_like(image, dtype=np.uint8)

    # Create mask for the specific region
    points = np.array([landmarks[i] for i in region_points], dtype=np.int32)
    cv2.fillPoly(mask, [points], (255, 255, 255))

    # Apply brightness adjustment
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 2] = np.clip(image_hsv[:, :, 2] + brightness, 0, 255)

    # Apply color adjustment
    color_layer = np.full_like(image, color, dtype=np.uint8)
    image_color = cv2.addWeighted(image, 0.7, color_layer, 0.3, 0)

    # Combine adjusted region with original
    output = cv2.bitwise_and(image_color, mask) + cv2.bitwise_and(image, cv2.bitwise_not(mask))
    return output

@app.post("/adjust-color/")
async def adjust_color(
    feature: str = Form(...),
    brightness: int = Form(...),
    color_r: int = Form(...),
    color_g: int = Form(...),
    color_b: int = Form(...),
    file: UploadFile = File(...)
):
    # Read and decode the image
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Detect face and landmarks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        return {"error": "No face detected in the image"}

    face = faces[0]
    landmarks = predictor(gray, face)
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    # Define regions based on feature
    if feature == "lip":
        region_points = list(range(48, 61))  # Lip landmarks
    elif feature == "hair":
        # Approximate top of the face for hair
        region_points = list(range(17))  # Jawline for now
    else:
        return {"error": "Invalid feature"}

    # Apply adjustments
    color = (color_b, color_g, color_r)  # OpenCV uses BGR format
    adjusted_image = apply_adjustments(image, points, region_points, brightness, color)

    # Encode and send adjusted image
    _, buffer = cv2.imencode(".jpg", adjusted_image)
    return StreamingResponse(BytesIO(buffer.tobytes()), media_type="image/jpeg")
