import os
import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
import re
import json

# ✅ Google Cloud Vision API認証設定
client = vision.ImageAnnotatorClient.from_service_account_info(st.secrets["google_credentials"])

# ✅ 保存ファイル名
MAPPINGS_FILE = "mappings.json"

# ✅ UI初期設定
st.set_page_config(layout="wide", page_title="🎰 パチスログラフ解析アプリ")
st.title("🎰 解析アプリ")

threshold = st.number_input("出玉枚数のしきい値（以上）", value=2000, step=1000, key="threshold_input")

uploaded_files = st.file_uploader(
    "📷 グラフ画像をアップロード（複数可）",
    type=None,
    accept_multiple_files=True
)

if 'ocr_cache' not in st.session_state:
    st.session_state.ocr_cache = {}
if 'manual_corrections' not in st.session_state:
    st.session_state.manual_corrections = {}

# ✅ 名称マッピング保存＆ロード関数
def load_mappings():
    if os.path.exists(MAPPINGS_FILE):
        try:
            with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_mappings(mappings):
    with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

if 'name_mappings' not in st.session_state:
    st.session_state.name_mappings = load_mappings()

# ✅ サイドバー（名称変更＋⬇️ボタン シンプル反映版）
st.sidebar.title("🛠 名称変更設定")
for i, mapping in enumerate(st.session_state.name_mappings):
    cols = st.sidebar.columns([5, 1])
    with cols[0]:
        updated_name_b = st.text_input(
            f"{mapping['name_a']}", value=mapping["name_b"], key=f"name_b_{i}"
        )
        if updated_name_b != mapping["name_b"]:
            st.session_state.name_mappings[i]["name_b"] = updated_name_b
            save_mappings(st.session_state.name_mappings)
    with cols[1]:
        if i < len(st.session_state.name_mappings) - 1:
            if st.button("⬇️", key=f"down_{i}"):
                st.session_state.name_mappings[i + 1], st.session_state.name_mappings[i] = (
                    st.session_state.name_mappings[i],
                    st.session_state.name_mappings[i + 1],
                )
                save_mappings(st.session_state.name_mappings)

# ✅ OCR実施
def run_ocr_once(img_cv):
    pil_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_content = buffered.getvalue()
    image = vision.Image(content=image_content)
    return client.text_detection(image=image)

# ✅ 出玉OCR
def extract_samai_by_fixed_coords(ocr_results, coords, img_width, img_height):
    results = []
    for idx, (x1, y1, x2, y2) in enumerate(coords):
        if x2 > img_width or y2 > img_height:
            results.append((idx, None, "座標外"))
            continue
        matched = []
        for text in ocr_results.text_annotations[1:]:
            vertices = text.bounding_poly.vertices
            xs = [v.x for v in vertices]
            ys = [v.y for v in vertices]
            box_x1 = min(xs)
            box_y1 = min(ys)
            box_x2 = max(xs)
            box_y2 = max(ys)
            if (x1 <= box_x1 <= x2) and (y1 <= box_y1 <= y2):
                matched.append(text.description)
        if matched:
            samai_text = " ".join(matched)
            samai_match = re.search(r'\d{3,5}', samai_text.replace(",", ""))
            if samai_match:
                samai = int(samai_match.group())
                results.append((idx, samai, samai_text))
            else:
                results.append((idx, None, samai_text))
        else:
            results.append((idx, None, "なし"))
    return results

# ✅ 赤色検出
def has_red_area(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    red_count = cv2.countNonZero(red_mask)
    return red_count >= 50

# ✅ テキスト描画
def draw_text_on_pil_image(pil_img, machine_name, ocr_text):
    draw = ImageDraw.Draw(pil_img)
    try:
        font_path = os.path.join(os.path.dirname(__file__), "NotoSansJP-Medium.ttf")
        font = ImageFont.truetype(font_path, size=24)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 5), f"{machine_name}", fill="white", font=font)
    draw.text((10, 35), f"{ocr_text}", fill="white", font=font)
    return pil_img

# ✅ メイン処理
machine_results = []

if uploaded_files:
    for file_idx, uploaded_file in enumerate(uploaded_files):
        try:
            image = Image.open(uploaded_file)
            base_width = 780
            w_percent = (base_width / float(image.size[0]))
            h_size = int((float(image.size[1]) * float(w_percent)))
            image_resized = image.resize((base_width, h_size), Image.LANCZOS)
            img_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)

            if uploaded_file.name not in st.session_state.ocr_cache:
                ocr_results = run_ocr_once(img_cv)
                st.session_state.ocr_cache[uploaded_file.name] = ocr_results
            else:
                ocr_results = st.session_state.ocr_cache[uploaded_file.name]

            # 機種名をOCR結果の最初の行から仮で取る（必要に応じて改善）
            machine_name = ocr_results.text_annotations[0].description.strip().split('\n')[0]
            existing_names = [m["name_a"] for m in st.session_state.name_mappings]
            if machine_name not in existing_names:
                st.session_state.name_mappings.append({"name_a": machine_name, "name_b": ""})
                save_mappings(st.session_state.name_mappings)

            display_name = next(
                (m["name_b"] for m in st.session_state.name_mappings if m["name_a"] == machine_name and m["name_b"]),
                machine_name
            )

            red_detected = has_red_area(img_cv)
            red_status = "〇赤あり" if red_detected else "×赤なし"

            samai_results = extract_samai_by_fixed_coords(
                ocr_results, [(230, 1010, 370, 1040)], img_cv.shape[1], img_cv.shape[0]
            )
            samai_value = samai_results[0][1] if samai_results else None
            samai_text = samai_results[0][2] if samai_results else "不明"

            machine_results.append({
                "machine": display_name,
                "graph_number": file_idx + 1,
                "image": image_resized,
                "samai_value": samai_value,
                "samai_text": samai_text,
                "red_status": red_status,
            })

        except Exception as e:
            st.error(f"エラー発生: {e}")

# ✅ 出力＆画像表示
if machine_results:
    st.subheader("📊 出力結果")
    cols = st.columns(4)
    for item in machine_results:
        idx = (item["graph_number"] - 1) % 4
        with cols[idx]:
            annotated_img = draw_text_on_pil_image(
                item["image"].copy(),
                f"{item['machine']} グラフ {item['graph_number']}",
                f"OCR結果: {item['samai_text']} / {item['red_status']}"
            )
            st.image(annotated_img, use_container_width=True)