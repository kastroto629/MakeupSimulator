from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import os
import cv2
import numpy as np
import torch
import base64

from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.download_util import load_file_from_url

from facelib.detection import init_detection_model
from facelib.utils.face_restoration_helper import get_largest_face
from facelib.utils.face_utils import align_crop_face_landmarks, paste_face_back
from facelib.utils.color_utils import extract_face_colors

router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CodeFormer 모델 초기화
net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                      connect_list=["32", "64", "128", "256"]).to(device)
ckpt_path = load_file_from_url(
    url="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    model_dir="weights/CodeFormer", progress=True)
net.load_state_dict(torch.load(ckpt_path)["params_ema"])
net.eval()


@router.post("/enhance_and_analyze")
async def enhance_and_analyze(file: UploadFile = File(...)):
    print(f"📥 분석 요청 받음: {file.filename}")
    contents = await file.read()
    print(f"📦 파일 크기: {len(contents)} bytes")
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print("❌ 이미지 디코딩 실패")
        return JSONResponse(status_code=400, content={"error": "이미지를 읽을 수 없습니다."})

    h, w = img.shape[:2]
    scale = max(h / 800, w / 800)
    img_scale = cv2.resize(img, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR) if scale > 1 else img

    # 얼굴 검출 및 landmark
    det_net = init_detection_model("retinaface_resnet50", half=False)
    bboxes = det_net.detect_faces(img_scale, 0.97)
    if scale > 1:
        bboxes *= scale
    if len(bboxes) == 0:
        return JSONResponse(status_code=404, content={"error": "얼굴을 찾을 수 없습니다."})
    bboxes = get_largest_face(bboxes, h, w)[0]
    landmarks = np.array([[bboxes[i], bboxes[i + 1]] for i in range(5, 15, 2)])

    # 얼굴 정렬
    aligned_face, inverse_affine = align_crop_face_landmarks(
        img, landmarks, output_size=512, return_inverse_affine=True)

    # CodeFormer 복원
    face_tensor = img2tensor(aligned_face / 255., bgr2rgb=True, float32=True).unsqueeze(0).to(device)
    normalize(face_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    with torch.no_grad():
        output = net(face_tensor, w=0.7, adain=True)[0]
        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")

    # 얼굴 원본 이미지에 다시 붙이기
    restored_full_img = paste_face_back(img.astype(np.float32), restored_face, inverse_affine).astype("uint8")

    # 색상 추출 (입술, 홍채, 눈썹)
    rgb_image = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)
    lip_hex, lip_html, iris_hex, iris_html, brow_hex, brow_html = extract_face_colors(rgb_image)

    # 이미지 base64 인코딩
    _, buffer = cv2.imencode(".png", restored_full_img)
    result_image_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "restored_image": result_image_base64,
        "lip_hex": lip_hex, "lip_html": lip_html,
        "iris_hex": iris_hex, "iris_html": iris_html,
        "brow_hex": brow_hex, "brow_html": brow_html
    }
