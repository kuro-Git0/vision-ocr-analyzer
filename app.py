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

# 認証と初期設定
client = vision.ImageAnnotatorClient.from_service_account_info(st.secrets["google_credentials"])
MAPPINGS_FILE = "mappings.json"
st.set_page_config(layout="wide", page_title="🎰 パチスログラフ解析アプリ")
st.title("🎰 解析アプリ")
threshold = st.number_input("出玉枚数のしきい値（以上）", value=2000, step=1000, key="threshold_input")
uploaded_files = st.file_uploader("📷 グラフ画像をアップロード（複数可）", accept_multiple_files=True)

# セッション初期化
if "ocr_cache" not in st.session_state:
    st.session_state.ocr_cache = {}
if "manual_corrections" not in st.session_state:
    st.session_state.manual_corrections = {}
if "name_mappings" not in st.session_state:
    st.session_state.name_mappings = []
if "rerun_output" not in st.session_state:
    st.session_state.rerun_output = False

# マッピング保存・ロード
def load_mappings():
    if os.path.exists(MAPPINGS_FILE):
        try:
            with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_mappings(mappings):
    with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)

if not st.session_state.name_mappings:
    st.session_state.name_mappings = load_mappings()

# 処理系関数群
def detect_graph_rectangles(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 200 < w < 800 and 200 < h < 800:
            rects.append((x, y, w, h))
    return sorted(rects, key=lambda r: (r[1], r[0]))

def run_ocr_once(img_cv):
    pil_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_content = buffered.getvalue()
    return client.text_detection(image=vision.Image(content=image_content))

def extract_machine_name_by_lines(ocr_results):
    lines = ocr_results.full_text_annotation.text.split("\n")[:15]
    for i, line in enumerate(lines):
        if re.search(r"\d+台", line) and i > 0:
            return lines[i - 1].strip()
    return "不明"

def get_fixed_coords():
    coords = []
    for row in range(10):
        y1 = 1010 + row * 375
        y2 = 1040 + row * 375
        coords += [(230, y1, 370, y2), (600, y1, 740, y2)]
    return coords

def extract_samai_by_fixed_coords(ocr_results, coords, img_width, img_height):
    results = []
    for idx, (x1, y1, x2, y2) in enumerate(coords):
        if x2 > img_width or y2 > img_height:
            results.append((idx, None, "座標外"))
            continue
        matched = []
        for text in ocr_results.text_annotations[1:]:
            xs = [v.x for v in text.bounding_poly.vertices]
            ys = [v.y for v in text.bounding_poly.vertices]
            if x1 <= min(xs) <= x2 and y1 <= min(ys) <= y2:
                matched.append(text.description)
        if matched:
            joined = " ".join(matched)
            m = re.search(r"\d{3,5}", joined.replace(",", ""))
            results.append((idx, int(m.group()) if m else None, joined))
        else:
            results.append((idx, None, "なし"))
    return results

def has_red_area(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
    return cv2.countNonZero(mask1 + mask2) >= 50

def draw_text_on_pil_image(pil_img, machine_name, ocr_text):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("NotoSansJP-Medium.ttf", 24)
    except:
        font = ImageFont.load_default()
    draw.text((10, 5), machine_name, fill="white", font=font)
    draw.text((10, 35), ocr_text, fill="white", font=font)
    return pil_img

# サイドバー（名称変更と並び替え）
st.sidebar.title("🛠 名称変更設定")
for i, mapping in enumerate(st.session_state.name_mappings):
    cols = st.sidebar.columns([5, 1])
    with cols[0]:
        updated = st.text_input(f"{mapping['name_a']}", value=mapping["name_b"], key=f"name_b_{i}")
        if updated != mapping["name_b"]:
            st.session_state.name_mappings[i]["name_b"] = updated
            save_mappings(st.session_state.name_mappings)
            st.session_state.rerun_output = True
    with cols[1]:
        if i < len(st.session_state.name_mappings) - 1:
            if st.button("⬇️", key=f"down_{i}"):
                st.session_state.name_mappings[i], st.session_state.name_mappings[i + 1] = (
                    st.session_state.name_mappings[i + 1],
                    st.session_state.name_mappings[i],
                )
                save_mappings(st.session_state.name_mappings)
                st.rerun()

# メイン解析
machine_results = []
if uploaded_files:
    coords_list = get_fixed_coords()
    st.session_state.rerun_output = True
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        if not filename.endswith((".jpg", ".jpeg", ".png")):
            st.warning(f"{filename} は非対応です")
            continue
        try:
            image = Image.open(uploaded_file)
            image = image.resize((780, int(image.size[1] * 780 / image.size[0])), Image.LANCZOS)
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            if filename not in st.session_state.ocr_cache:
                st.session_state.ocr_cache[filename] = run_ocr_once(img_cv)

            ocr = st.session_state.ocr_cache[filename]
            machine = extract_machine_name_by_lines(ocr)
            if machine not in [m["name_a"] for m in st.session_state.name_mappings]:
                st.session_state.name_mappings.append({"name_a": machine, "name_b": ""})
                save_mappings(st.session_state.name_mappings)

            display = next((m["name_b"] for m in st.session_state.name_mappings if m["name_a"] == machine and m["name_b"]), machine)
            samai = extract_samai_by_fixed_coords(ocr, coords_list, *img_cv.shape[1::-1])
            rects = detect_graph_rectangles(img_gray)

            for idx, (x, y, w, h) in enumerate(rects):
                crop = img_cv[y:y+h, x:x+w]
                key = f"{machine}_graph_{idx + 1}"
                if key not in st.session_state.manual_corrections:
                    st.session_state.manual_corrections[key] = ""
                machine_results.append({
                    "machine": display,
                    "graph_number": idx + 1,
                    "image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                    "samai_value": samai[idx][1] if idx < len(samai) else None,
                    "samai_text": samai[idx][2] if idx < len(samai) else "不明",
                    "red_status": "〇赤あり" if has_red_area(crop) else "×赤なし",
                    "manual_key": key
                })
        except Exception as e:
            st.error(f"{filename} 処理失敗: {e}")

# 出力ボタン
st.button("🔄 出力を更新する", on_click=lambda: setattr(st.session_state, "rerun_output", True))

# 出力結果
if machine_results and st.session_state.rerun_output:
    st.subheader("📊 出力結果")
    out = []
    grouped = defaultdict(list)
    for item in machine_results:
        grouped[item["machine"]].append(item)

    for mapping in st.session_state.name_mappings:
        name = mapping["name_b"] if mapping["name_b"] else mapping["name_a"]
        if name not in grouped:
            continue
        items = sorted(grouped[name], key=lambda x: x["graph_number"])
        valid = []
        for i in items:
            val = st.session_state.manual_corrections.get(i["manual_key"], "").strip()
            v = int(val) if val.isdigit() else i["samai_value"]
            if v and v >= threshold and i["red_status"] == "〇赤あり":
                valid.append(v)
        out.append(f"▼{name} ({len(valid)}/{len(items)})")
        for v in sorted(valid, reverse=True):
            if v >= 19000:
                out.append(f"㊗️{v}枚 コンプ！")
            elif v >= 10000:
                out.append(f"🎉{v}枚")
            elif v >= 8000:
                out.append(f"🚨{v}枚")
            elif v >= 5000:
                out.append(f"✨{v}枚")
            else:
                out.append(f"・{v}枚")
        out.append("")
    st.code("\n".join(out), language="")

# グラフ＋修正欄
cols = st.columns(4)
for mapping in st.session_state.name_mappings:
    name = mapping["name_b"] if mapping["name_b"] else mapping["name_a"]
    items = [m for m in machine_results if m["machine"] == name]
    for item in sorted(items, key=lambda x: x["graph_number"]):
        col = cols[(item["graph_number"] - 1) % 4]
        with col:
            img = draw_text_on_pil_image(item["image"].copy(), f"{item['machine']} グラフ {item['graph_number']}", f"OCR結果: {item['samai_text']} / {item['red_status']}")
            st.image(img, use_container_width=True)
            default_val = st.session_state.manual_corrections.get(item["manual_key"], "")
            val = st.text_input("⬇️最大枚数の修正", value=default_val, key=f"manual_{item['manual_key']}", label_visibility="collapsed", placeholder="▼最大枚数の修正")
            if val != "":
                st.session_state.manual_corrections[item["manual_key"]] = val