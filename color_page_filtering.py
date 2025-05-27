import sys
import traceback
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import gradio as gr
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

# ─── 0) CSV→SQLite 초기화 ───────────────────────────────
def init_cosmetic_db(csv_path="cosmetics.csv", db_path="cosmetics.db"):
    df = pd.read_csv(csv_path)
    # price: 숫자 이외 모두 제거 → int
    df["price"] = (
        df["price"].astype(str)
          .str.replace(r"[^\d]", "", regex=True)
          .replace("", "0")
          .astype(int)
    )
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS products")
    cur.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            section TEXT,
            category TEXT,
            type TEXT,
            brand TEXT,
            product_series TEXT,
            product_name TEXT,
            hex_color TEXT,
            price INTEGER
        )
    """)
    df.to_sql("products", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

# ─── 1) RGB→Lab 벡터, HEX→스와치 HTML ───────────────────
def rgb_to_lab_vector(r, g, b):
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    lab = convert_color(rgb, LabColor)
    return np.array([[lab.lab_l, lab.lab_a, lab.lab_b]])

def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def get_html_swatch(hex_color):
    return (
        f"<div style='width:120px; height:50px; "
        f"background-color:{hex_color}; border:1px solid #000; "
        f"border-radius:4px; margin-bottom:8px;'></div>"
    )

# ─── 2) DB 로드 + Lab 벡터화 ───────────────────────────
def load_cosmetics(db_path="cosmetics.db"):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT id, section, category, type, brand,
               product_series, product_name, hex_color, price
        FROM products
    """).fetchall()
    conn.close()
    records, vectors = [], []
    for row in rows:
        hex_c = row[7]
        if not isinstance(hex_c, str): continue
        try:
            r, g, b = [int(hex_c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)]
        except:
            continue
        records.append(row)
        vectors.append(rgb_to_lab_vector(r, g, b)[0])
    return records, np.array(vectors)

# 초기화 및 로드
try:
    init_cosmetic_db()
    cosmetic_records, cosmetic_vectors = load_cosmetics()
except Exception as e:
    print("DB 초기화/로드 실패:", e)
    traceback.print_exc()
    sys.exit(1)

# 필터용 목록
ALL_SECTIONS  = sorted({r[1] for r in cosmetic_records})
ALL_CATEGORIES= sorted({r[2] for r in cosmetic_records})
ALL_BRANDS    = sorted({r[4] for r in cosmetic_records})
ALL_TYPES     = sorted({r[3] for r in cosmetic_records})
PRICES        = [r[8] for r in cosmetic_records if isinstance(r[8], int)]
MIN_PRICE, MAX_PRICE = (min(PRICES), max(PRICES)) if PRICES else (0, 0)

# 필터링 초기화 함수
def reset_filters_and_results():
    return (
        [],  # section_f
        [],  # category_f
        [],  # brand_f
        [],  # type_f
        (MIN_PRICE, MAX_PRICE),  # price_f
        "#000000",  # hex_in
        "",  # rec_html
    )

# ─── 3) Cosine 유사도 추천 (필터링 포함) ───────────────
def recommend_with_filters(hex_code, sections, categories, brands, types, price_range, top_n=5):
    print(f"입력 HEX: {hex_code}, 필터: {sections}, {categories}, {brands}, {types}, {price_range}, top_n: {top_n}")
    # HEX 검사
    if not hex_code or not hex_code.startswith("#") or len(hex_code) != 7:
        return "❌ 유효한 HEX 코드", ""
    try:
        r, g, b = [int(hex_code[i:i+2], 16) for i in (1, 3, 5)]
    except ValueError as ve:
        return f"HEX 파싱 오류: {ve}", ""
    # 가격 범위 처리
    if isinstance(price_range, (list, tuple)) and len(price_range) == 2:
        pmin, pmax = price_range
    elif isinstance(price_range, (int, float)):
        pmin, pmax = MIN_PRICE, int(price_range)  # 단일 값이면 최대값으로 간주
    else:
        return "가격 범위가 유효하지 않습니다", ""
    print(f"가격 범위: {pmin} ~ {pmax}")
    # 필터링
    idxs = [
        i for i, rec in enumerate(cosmetic_records)
        if (not sections or rec[1] in sections)
        and (not categories or rec[2] in categories)
        and (not types or rec[3] in types)
        and (not brands or rec[4] in brands)
        and (pmin <= rec[8] <= pmax)
    ]
    print(f"필터링된 제품 수: {len(idxs)}")
    if not idxs:
        print(f"조건: section={sections}, category={categories}, brand={brands}, type={types}, price={pmin}~{pmax}")
        return "필터 조건에 맞는 제품이 없습니다. 조건을 완화해 보세요.", ""
    fvecs = cosmetic_vectors[idxs]
    qv = rgb_to_lab_vector(r, g, b)
    sims = cosine_similarity(qv, fvecs)[0]
    top = sims.argsort()[::-1][:top_n]
    # HTML 생성
    html = "<ul style='list-style:none;padding:0;'>"
    for rank in top:
        sim = sims[rank]
        rec = cosmetic_records[idxs[rank]]
        _, sec, cat, typ, br, ser, nm, hc, pr = rec
        sw = get_html_swatch(hc)
        html += (
            f"<li style='margin-bottom:8px;'>"
            f"{sw}<b>{br}</b> {ser} {nm}<br>"
            f"<code>{hc}</code> | {sec}/{cat}/{typ} | {pr}원 "
            f"(유사도: {sim:.2f})"
            "</li>"
        )
    html += "</ul>"
    return f"{len(top)}개 추천", html

# ─── 4) MediaPipe FaceMesh 세팅 ────────────────────────
mpfm = mp.solutions.face_mesh
face_mesh = mpfm.FaceMesh(static_image_mode=True, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.5)
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
IRIS_GROUPS = [[473, 474, 475, 476, 477], [468, 469, 470, 471, 472]]
EYEBROW_GROUPS = [[156, 70, 63, 105, 66, 107, 55, 193],
                  [383, 300, 293, 334, 296, 336, 285, 417]]

# ─── 5) 자동/수동 색상 추출 ─────────────────────────────
def extract_region_color_histogram(img, add, sub=None, margin=15):
    if img is None:
        return "#000000", get_html_swatch("#000000")
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    h, w = img.shape[:2]
    # MediaPipe는 BGR 형식을 기대
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not res.multi_face_landmarks:
        return "#000000", get_html_swatch("#000000")
    lm = res.multi_face_landmarks[0]
    mask = np.zeros((h, w), np.uint8)
    for grp in add:
        pts = np.array([[int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)]
                        for i in grp if i < len(lm.landmark)], np.int32)
        if pts.size:
            cv2.fillPoly(mask, [pts], 255)
    if sub:
        m2 = np.zeros((h, w), np.uint8)
        for grp in sub:
            pts = np.array([[int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)]
                            for i in grp if i < len(lm.landmark)], np.int32)
            if pts.size:
                cv2.fillPoly(m2, [pts], 255)
        mask[m2 == 255] = 0
    pix = img[mask == 255]
    if pix.size == 0:
        return "#000000", get_html_swatch("#000000")
    m = pix.mean(axis=0)
    low, high = np.clip(m - margin, 0, 255), np.clip(m + margin, 0, 255)
    filt = pix[np.all((pix >= low) & (pix <= high), axis=1)]
    if filt.size == 0:
        filt = pix
    r, g, b = map(int, filt.mean(axis=0))
    hc = rgb_to_hex(r, g, b)
    return hc, get_html_swatch(hc)

def extract_face_colors(img):
    if img is None:
        return ("#000000", get_html_swatch("#000000")) * 3
    lh, lsw = extract_region_color_histogram(img, [LIPS_OUTER], [LIPS_INNER])
    ih, isw = extract_region_color_histogram(img, IRIS_GROUPS)
    bh, bsw = extract_region_color_histogram(img, EYEBROW_GROUPS)
    return lh, lsw, ih, isw, bh, bsw

def manual_spoid(img, x, y):
    if img is None:
        return "", "", "이미지 먼저 업로드", None
    try:
        x, y = int(x), int(y)
        h, w, _ = img.shape
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("좌표 범위 벗어남")
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        r, g, b = img[y, x]
        hc = rgb_to_hex(r, g, b)
        sw = get_html_swatch(hc)
        mk = img.copy()
        cv2.circle(mk, (x, y), 7, (255, 0, 0), -1)
        cv2.putText(mk, f"({x},{y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return hc, sw, "성공", mk
    except Exception as e:
        return "", "", f"에러: {e}", None

# ─── 6) Gradio UI ──────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("## 🎨 얼굴 색상 추출 & 필터링 가능한 화장품 추천")

    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="이미지 업로드")
            size_tb = gr.Textbox(label="이미지 크기")
            with gr.Row():
                gr.Markdown("**입술**");   lip_hex, lip_sw   = gr.Textbox(), gr.HTML()
                gr.Markdown("**홍채**");   iris_hex, iris_sw = gr.Textbox(), gr.HTML()
                gr.Markdown("**눈썹**"); brow_hex, brow_sw = gr.Textbox(), gr.HTML()
            x_tb, y_tb = gr.Textbox(label="X"), gr.Textbox(label="Y")
            btn_manual = gr.Button("수동 추출")
            out_hex_m, out_sw_m, out_stat, out_img = (
                gr.Textbox(label="HEX"), gr.HTML(), gr.Textbox(label="상태"), gr.Image()
            )

    with gr.Accordion("🔍 추천 필터 설정", open=False):
        section_f = gr.CheckboxGroup(choices=ALL_SECTIONS, label="Section")
        category_f = gr.CheckboxGroup(choices=ALL_CATEGORIES, label="Category")
        brand_f = gr.CheckboxGroup(choices=ALL_BRANDS, label="Brand")
        type_f = gr.CheckboxGroup(choices=ALL_TYPES, label="Type")
        price_f = gr.Slider(
            minimum=MIN_PRICE,
            maximum=MAX_PRICE,
            value=(MIN_PRICE, MAX_PRICE),
            step=1000,
            label="Price Range",
            interactive=True,
            type="range"  # 범위 슬라이더 명시 (Gradio 4.x 호환)
        )
        btn_reset = gr.Button("필터 및 결과 리셋")

    with gr.Row():
        gr.Markdown("### 💄 제품 추천")
        hex_in = gr.Textbox(label="추천할 HEX 코드 입력", value="#000000")
        btn_rec = gr.Button("추천 시작")
        rec_cnt = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="추천 수")
        rec_cnt_out = gr.Textbox(label="추천된 제품 수")
        rec_html = gr.HTML(label="추천 결과")

    # 이벤트 연결
    inp.change(extract_face_colors, inputs=inp,
               outputs=[lip_hex, lip_sw, iris_hex, iris_sw, brow_hex, brow_sw])
    inp.change(lambda img: f"{img.shape[1]}×{img.shape[0]}" if img is not None else "",
               inputs=inp, outputs=size_tb)
    btn_manual.click(manual_spoid, inputs=[inp, x_tb, y_tb],
                     outputs=[out_hex_m, out_sw_m, out_stat, out_img])
    lip_hex.change(lambda h: str(h) if h.startswith("#") else "#000000",
                   inputs=lip_hex, outputs=hex_in)
    btn_rec.click(recommend_with_filters,
                  inputs=[hex_in, section_f, category_f, brand_f, type_f, price_f, rec_cnt],
                  outputs=[rec_cnt_out, rec_html])
    btn_reset.click(reset_filters_and_results,
                    inputs=None,
                    outputs=[section_f, category_f, brand_f, type_f, price_f, hex_in, rec_html])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7863, debug=True)