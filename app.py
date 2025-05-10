import os
import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
import re
from collections import defaultdict
import json

# ✅ Google Cloud Vision API認証設定（secretsから認証情報を読み込み）
client = vision.ImageAnnotatorClient.from_service_account_info(st.secrets["google_credentials"])

# ✅ 保存ファイル名（機種名マッピングを保存するローカルJSONファイル）
MAPPINGS_FILE = "mappings.json"

# ✅ UI初期設定
st.set_page_config(layout="wide", page_title="🎰 パチスログラフ解析アプリ")
st.title("🎰 解析アプリ")

# ✅ 出玉枚数のしきい値
threshold = st.number_input("出玉枚数のしきい値（以上）", value=2000, step=1000, key="threshold_input")

# ✅ 画像アップローダー
uploaded_files = st.file_uploader(
    "📷 グラフ画像をアップロード（複数可）",
    type=None,
    accept_multiple_files=True
)

# ✅ OCRキャッシュ初期化
if 'ocr_cache' not in st.session_state:
    st.session_state.ocr_cache = {}
if 'manual_corrections' not in st.session_state:
    st.session_state.manual_corrections = {}

# ✅ 名称マッピング保存＆ロード
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

# ✅ グラフ検出
def detect_graph_rectangles(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 200 < w < 800 and 200 < h < 800:
            rects.append((x, y, w, h))
    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    return rects

# ✅ OCR
def run_ocr_once(img_cv):
    pil_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_content = buffered.getvalue()
    image = vision.Image(content=image_content)
    return client.text_detection(image=image)

# ✅ 機種名抽出
def extract_machine_name_by_lines(ocr_results):
    lines = ocr_results.full_text_annotation.text.split("\n")[:15]
    for i, line in enumerate(lines):
        if re.search(r"\d+台", line):
            if i > 0:
                return lines[i - 1].strip()
    return "不明"

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

# ✅ サイドバー（名称変更＋⬇️ボタンを右側に配置）
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
                st.rerun()

# ✅ メイン処理
machine_results = defaultdict(list)

if uploaded_files:
    for uploaded_file in uploaded_files:
        filename_lower = uploaded_file.name.lower()
        if not (filename_lower.endswith('.jpg') or filename_lower.endswith('.jpeg') or filename_lower.endswith('.png')):
            st.warning(f"スキップ: {uploaded_file.name} はサポート外です")
            continue

        try:
            image = Image.open(uploaded_file)
            base_width = 780
            w_percent = (base_width / float(image.size[0]))
            h_size = int((float(image.size[1]) * float(w_percent)))
            image_resized = image.resize((base_width, h_size), Image.LANCZOS)
            img_cv = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            if uploaded_file.name not in st.session_state.ocr_cache:
                ocr_results = run_ocr_once(img_cv)
                st.session_state.ocr_cache[uploaded_file.name] = ocr_results
            else:
                ocr_results = st.session_state.ocr_cache[uploaded_file.name]

            machine_name = extract_machine_name_by_lines(ocr_results)
            existing_names = [m["name_a"] for m in st.session_state.name_mappings]
            if machine_name not in existing_names:
                st.session_state.name_mappings.append({"name_a": machine_name, "name_b": ""})
                save_mappings(st.session_state.name_mappings)
                st.rerun()

            display_name = next(
                (m["name_b"] for m in st.session_state.name_mappings if m["name_a"] == machine_name and m["name_b"]),
                machine_name
            )

            rects = detect_graph_rectangles(img_gray)
            for idx, (x, y, w, h) in enumerate(rects):
                crop = img_cv[y:y + h, x:x + w]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                red_detected = has_red_area(crop)
                red_status = "〇赤あり" if red_detected else "×赤なし"

                default_key = f"{display_name}_graph_{idx + 1}"
                manual_value = st.session_state.manual_corrections.get(default_key, "")

                samai_input = st.text_input(
                    f"{display_name} グラフ {idx + 1} 出玉修正",
                    value=manual_value,
                    key=f"manual_{default_key}"
                )
                if samai_input != manual_value:
                    st.session_state.manual_corrections[default_key] = samai_input

                # OCR結果を保持（空欄ならOCR値を使う）
                ocr_text = f"OCR結果: {samai_input if samai_input else '不明'} / {red_status}"
                annotated_img = draw_text_on_pil_image(
                    pil_crop.copy(),
                    f"{display_name} グラフ {idx + 1}",
                    ocr_text
                )

                machine_results[display_name].append({
                    "index": idx + 1,
                    "image": annotated_img,
                    "samai_value": samai_input.strip() if samai_input.strip() else None,
                    "red_status": red_status
                })

        except Exception as e:
            st.error(f"エラー発生: {e}")

# ✅ 出力結果（上に戻す、ソートも適用）
if machine_results:
    st.subheader("📊 出力結果")
    output_texts = []
    for machine in sorted(machine_results.keys()):
        results = sorted(machine_results[machine], key=lambda x: x["index"])
        filtered = []
        for result in results:
            try:
                val = int(result["samai_value"]) if result["samai_value"] else None
            except:
                val = None
            if val and val >= threshold:
                filtered.append(val)
        header = f"▼{machine} ({len(filtered)}/{len(results)})"
        output_texts.append(header)
        for val in sorted(filtered, reverse=True):
            if val >= 19000:
                output_texts.append(f"㊗️{val}枚 コンプ！")
            elif val >= 10000:
                output_texts.append(f"🎉{val}枚")
            elif val >= 8000:
                output_texts.append(f"🚨{val}枚")
            elif val >= 5000:
                output_texts.append(f"✨{val}枚")
            else:
                output_texts.append(f"・{val}枚")
        output_texts.append("")
    st.code("\n".join(output_texts), language="")

# ✅ 検出したグラフ画像を4列で表示（ソート付き）
cols = st.columns(4)
for machine in sorted(machine_results.keys()):
    for item in sorted(machine_results[machine], key=lambda x: x["index"]):
        col = cols[(item["index"] - 1) % 4]
        with col:
            col.image(item["image"], use_container_width=True)