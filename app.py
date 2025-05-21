from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import subprocess
import cv2
import numpy as np
import base64
from facelib.utils.color_utils import extract_face_colors

app = FastAPI()

# 👉 CORS 설정 (React 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 👉 경로 설정
INPUT_DIR = 'inputs/TestWhole'
RESULT_DIR = 'results/TestWhole_0.7/restored_faces'

# 👉 폴더 초기화
@app.post("/reset")
def reset_folders():
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    shutil.rmtree(RESULT_DIR, ignore_errors=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    return {"message": "✅ 입력 및 출력 폴더 초기화 완료"}

# 👉 파일 업로드
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    os.makedirs(INPUT_DIR, exist_ok=True)
    dst_path = os.path.join(INPUT_DIR, file.filename)
    with open(dst_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename}

# 👉 CodeFormer subprocess 실행
@app.post("/enhance")
def enhance():
    result = subprocess.run(
        "python inference_codeformer.py -w 0.7 --input_path inputs/TestWhole",
        shell=True, capture_output=True, text=True
    )

    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR ===")
    print(result.stderr)

    if result.returncode != 0:
        return JSONResponse(status_code=500, content={
            "error": result.stderr or "복원 중 알 수 없는 오류가 발생했습니다.",
            "stdout": result.stdout
        })

    return {
        "message": "복원 완료",
        "stdout": result.stdout
    }


# 👉 결과 이미지 보기
@app.get("/result/{filename}")
def get_result(filename: str):
    result_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(result_path):
        return FileResponse(result_path, media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "복원된 파일이 없습니다."})

# 👉 다운로드 엔드포인트
@app.get("/download")
def download_result(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(
        path=file_path,
        media_type='image/png',
        filename='restored_image.png',
        headers={"Content-Disposition": "attachment; filename=restored_image.png"}
    )
    else:
        return {"error": "파일이 존재하지 않습니다."}

# 👉 분석 기능 추가
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    print(f"📥 분석 요청 받음: {file.filename}")
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "이미지를 읽을 수 없습니다."})

    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lip_hex, lip_html, iris_hex, iris_html, brow_hex, brow_html = extract_face_colors(rgb_image)

    return {
        "lip_hex": lip_hex, "lip_html": lip_html,
        "iris_hex": iris_hex, "iris_html": iris_html,
        "brow_hex": brow_hex, "brow_html": brow_html
    }
