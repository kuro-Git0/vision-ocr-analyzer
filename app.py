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

# ✅ Google Cloud Vision API認証
client = vision.ImageAnnotatorClient.from_service_account_info(st.secrets["google_credentials"])

MAPPINGS_FILE = "mappings.json"

st.set_page_config(layout="wide", page_title="🎰 パチスログラフ解析アプリ")
st.title("🎰 解析アプリ")

threshold = st.number_input("出玉枚数のしきい値（以上）", value=2000, step=1000)

uploaded_files = st.file_uploader(
    "📷 グラフ画像をアップロード（複数可）",
    type=None,
    accept_multiple_files=True
)

if 'ocr_cache' not in st.session_state:
    st.session_state.ocr_cache = {}
if 'manual_corrections' not in st.session_state:
    st.session_state.manual_corrections = {}

def load_mappings():
    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_mappings(mappings):
    with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

if 'name_mappings' not in st.session_state:
    st.session_state.name_mappings = load_mappings()

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

def run_ocr_once(img_cv):
    pil_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_content = buffered.getvalue()
    image = vision.Image(content=image_content)
    return client.text_detection(image=image)

def extract_machine_name_by_lines(ocr_results):
    lines = ocr_results.full_text_annotation.text.split("\n")[:15]
    for i, line in enumerate(lines):
        if re.search(r"\d+台", line):
            if i > 0:
                return lines[i - 1].strip()
    return "不明"

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

# ✅ 名称変更用
st.sidebar.title("🛠 名称変更設定")
for i, mapping in enumerate(st.session_state.name_mappings):
    col1, col2 = st.sidebar.columns([1, 5])
    with col1:
        if i < len(st.session_state.name_mappings) - 1:
            if st.button("⬇️", key=f"down_{i}"):
                st.session_state.name_mappings[i + 1], st.session_state.name_mappings[i] = (
                    st.session_state.name_mappings[i],
                    st.session_state.name_mappings[i + 1],
                )
                save_mappings(st.session_state.name_mappings)
                st.rerun()
    with col2:
        updated_name_b = st.text_input(
            f"{mapping['name_a']}", value=mapping["name_b"], key=f"name_b_{i}"
        )
        if updated_name_b != mapping["name_b"]:
            st.session_state.name_mappings[i]["name_b"] = updated_name_b
            save_mappings(st.session_state.name_mappings)

graph_data_list = []

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

                # OCRから数値を取得
                full_text = ocr_results.full_text_annotation.text
                found_match = re.search(r'\d{3,5}', full_text.replace(",", ""))
                ocr_value = found_match.group() if found_match else "0"

                # manual_correctionsに初期登録（空の場合）
                if default_key not in st.session_state.manual_corrections:
                    st.session_state.manual_corrections[default_key] = ""

                graph_data_list.append({
                    "machine": display_name,
                    "index": idx + 1,
                    "image": pil_crop,
                    "red_status": red_status,
                    "manual_key": default_key,
                    "ocr_value": ocr_value
                })

        except Exception as e:
            st.error(f"エラー発生: {e}")

if graph_data_list:
    sorted_graphs = sorted(graph_data_list, key=lambda x: (x["machine"], x["index"]))
    cols = st.columns(4)
    for i, data in enumerate(sorted_graphs):
        col = cols[i % 4]
        with col:
            # 表示用：OCR値 or 修正値
            current_val = st.session_state.manual_corrections.get(data["manual_key"], "").strip()
            show_val = current_val if current_val else data["ocr_value"]
            annotated_img = data["image"].copy()
            draw = ImageDraw.Draw(annotated_img)
            draw.text((10, 5), f"{data['machine']} グラフ {data['index']}", fill="white")
            draw.text((10, 35), f"OCR結果: {show_val} / {data['red_status']}", fill="white")
            col.image(annotated_img, use_container_width=True)

            corrected = st.text_input(
                f"{data['machine']} グラフ {data['index']} 出玉修正",
                value=current_val,
                key=f"correct_{data['manual_key']}"
            )
            st.session_state.manual_corrections[data["manual_key"]] = corrected

    # ✅ 出力結果
    st.subheader("📊 出力結果")
    output_texts = []
    machine_group = defaultdict(list)
    for data in sorted_graphs:
        val_text = st.session_state.manual_corrections.get(data["manual_key"], "").strip()
        val_to_use = val_text if val_text else data["ocr_value"]
        try:
            val = int(val_to_use)
            machine_group[data["machine"]].append(val)
        except:
            continue

    for mapping in st.session_state.name_mappings:
        machine = mapping["name_b"] if mapping["name_b"] else mapping["name_a"]
        vals = [v for v in machine_group.get(machine, []) if v >= threshold]
        header = f"▼{machine} ({len(vals)}/{len(machine_group.get(machine, []))})"
        output_texts.append(header)
        for val in sorted(vals, reverse=True):
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