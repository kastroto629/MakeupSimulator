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
    if session_id not in session_images_original:
        return {"error": "Invalid session ID"}

    # 원본 이미지에서 시작
    original_image = session_images_original[session_id]
    image = original_image.copy()

    # 얼굴 검출 및 랜드마크 추출
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        return {"error": "No face detected"}

    landmarks = predictor(gray, faces[0])
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    # 영역 선택: 입술 또는 눈
    region_points = list(range(48, 61)) if feature == "lip" else list(range(36, 48))
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array([points[i] for i in region_points], np.int32)], (255, 255, 255))

    if brightness == 0 and color_r == 0 and color_g == 0 and color_b == 0:
        # 모든 값이 0이면 원본 이미지의 해당 영역만 유지
        combined = cv2.bitwise_and(original_image, mask) + cv2.bitwise_and(image, cv2.bitwise_not(mask))
    else:
        # 색상 레이어 생성 및 적용
        color_layer = np.full_like(image, (color_b, color_g, color_r), dtype=np.uint8)

        # 밝기 및 색상 조정
        adjusted = cv2.addWeighted(image, 1 - (brightness / 100), color_layer, (brightness / 100), 0)

        # 선택된 영역에만 조정된 색상 반영
        combined = cv2.bitwise_and(adjusted, mask) + cv2.bitwise_and(image, cv2.bitwise_not(mask))

    # 클라이언트에 이미지 반환
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
