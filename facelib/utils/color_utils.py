import cv2
import mediapipe as mp
import numpy as np

# MediaPipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 랜드마크 인덱스 정의
LIPS_UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER_OUTER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_UPPER_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_LOWER_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LIPS_OUTER_INDEXES = list(set(LIPS_UPPER_OUTER + LIPS_LOWER_OUTER))
LIPS_INNER_INDEXES = list(set(LIPS_UPPER_INNER + LIPS_LOWER_INNER))

RIGHT_EYE_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_IRIS = [468, 469, 470, 471, 472]
IRIS_GROUPS = [RIGHT_EYE_IRIS, LEFT_EYE_IRIS]

RIGHT_EYEBROW_UPPER = [156, 70, 63, 105, 66, 107, 55, 193]
RIGHT_EYEBROW_LOWER = [35, 124, 46, 53, 52, 65]
LEFT_EYEBROW_UPPER = [383, 300, 293, 334, 296, 336, 285, 417]
LEFT_EYEBROW_LOWER = [265, 353, 276, 283, 282, 295]
EYEBROW_GROUPS = [
    RIGHT_EYEBROW_UPPER, RIGHT_EYEBROW_LOWER,
    LEFT_EYEBROW_UPPER, LEFT_EYEBROW_LOWER
]


def get_hex_and_html(r, g, b):
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    html = f"""<div style="
      width:120px; height:50px;
      background-color:{hex_color};
      border:1px solid #000; border-radius:4px;
      margin-bottom:8px;"></div>"""
    return hex_color, html


def extract_region_color_groups(image_rgb, add_groups, sub_groups=None):
    h, w, _ = image_rgb.shape
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return get_hex_and_html(0, 0, 0)
    lm = results.multi_face_landmarks[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    for grp in add_groups:
        pts = np.array([[int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)]
                        for i in grp], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    if sub_groups:
        mask_sub = np.zeros((h, w), dtype=np.uint8)
        for grp in sub_groups:
            pts = np.array([[int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)]
                            for i in grp], np.int32)
            cv2.fillPoly(mask_sub, [pts], 255)
        mask[mask_sub == 255] = 0
    pixels = image_rgb[mask == 255]
    if pixels.size == 0:
        return get_hex_and_html(0, 0, 0)
    uniq, counts = np.unique(pixels.reshape(-1, 3), axis=0, return_counts=True)
    r, g, b = map(int, uniq[counts.argmax()])
    return get_hex_and_html(r, g, b)


def extract_iris_color_kmeans(image_rgb, iris_groups, k=3):
    h, w, _ = image_rgb.shape
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return get_hex_and_html(0, 0, 0)
    lm = results.multi_face_landmarks[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    for grp in iris_groups:
        pts = np.array([[int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)]
                        for i in grp], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    region_rgb = image_rgb[mask == 255]
    region_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)[mask == 255]
    if region_rgb.size == 0:
        return get_hex_and_html(0, 0, 0)
    Z = region_rgb.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    centers = centers.astype(np.uint8)
    best_score, best_center = -1, centers[0]
    for i in range(k):
        idxs = np.where(labels == i)[0]
        if idxs.size == 0:
            continue
        s_vals = region_hsv[idxs, 1].astype(float) / 255
        score = idxs.size * (s_vals.mean() if s_vals.size > 0 else 0)
        if score > best_score:
            best_score = score
            best_center = centers[i]
    r, g, b = map(int, best_center)
    return get_hex_and_html(r, g, b)


def extract_face_colors(image):
    if image is None:
        return ("없음", "") * 3
    rgb = image
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return ("얼굴 미검출", "") * 3
    lip_hex, lip_html = extract_region_color_groups(rgb, add_groups=[LIPS_OUTER_INDEXES], sub_groups=[LIPS_INNER_INDEXES])
    iris_hex, iris_html = extract_iris_color_kmeans(rgb, IRIS_GROUPS)
    brow_hex, brow_html = extract_region_color_groups(rgb, add_groups=EYEBROW_GROUPS)
    return (
        lip_hex or "없음", lip_html,
        iris_hex or "없음", iris_html,
        brow_hex or "없음", brow_html
    )
