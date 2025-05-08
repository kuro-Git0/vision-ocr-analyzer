import os
import io
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
import re
from collections import defaultdict

# ✅ Google Cloud Vision API認証設定
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "pachislot-ocr-458910-c0b5dda31bd7.json"
client = vision.ImageAnnotatorClient()

st.set_page_config(layout="wide", page_title="🎰 パチスログラフ解析アプリ")
st.title("🎰 パチスログラフ解析アプリ（グラフ自動検出＋最大枚数を座標指定で安定抽出＋赤色検出＋日本語フォント対応）")

threshold = st.number_input("出玉枚数のしきい値（以上）", value=2000, step=1000)

uploaded_files = st.file_uploader(
    "📷 グラフ画像をアップロード（複数可）",
    type=None,
    accept_multiple_files=True
)

if 'ocr_cache' not in st.session_state:
    st.session_state.ocr_cache = {}

# ✅ グラフ検出ロジック
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

# ✅ OCR実施
def run_ocr_once(img_cv):
    pil_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    image_content = buffered.getvalue()
    image = vision.Image(content=image_content)
    return client.text_detection(image=image)

# ✅ 機種名を抽出（以前の方式）
def extract_machine_name_by_lines(ocr_results):
    lines = ocr_results.full_text_annotation.text.split("\n")[:15]
    for i, line in enumerate(lines):
        if re.search(r"\d+台", line):
            if i > 0:
                return lines[i - 1].strip()
    return "不明"

# ✅ 固定座標リスト（例：2列×10行 = 20個）
def get_fixed_coords():
    coords = []
    for row in range(10):
        y1 = 1010 + row * 375
        y2 = 1040 + row * 375
        coords.append((230, y1, 370, y2))  # 1列目
        coords.append((600, y1, 740, y2))  # 2列目
    return coords

# ✅ 出玉OCR（絶対座標式）
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

# ✅ 赤色検出ロジック（50ピクセル以上で赤判定）
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

# ✅ テキスト描画（PILで直接描き込み：日本語対応フォント）
def draw_text_on_pil_image(pil_img, machine_name, ocr_text):
    draw = ImageDraw.Draw(pil_img)
    try:
        font_path = r"M:\マイドライブ\くろいけもみみ\開発\vision-ocr-analyzer\NotoSansJP-Medium.ttf"
        font = ImageFont.truetype(font_path, size=24)
    except IOError:
        font = ImageFont.load_default()
    # 上部に描画
    draw.text((10, 5), f"{machine_name}", fill="white", font=font)
    draw.text((10, 35), f"{ocr_text}", fill="white", font=font)
    return pil_img

# -----------------------------------

machine_results = defaultdict(lambda: {"entries": [], "total_count": 0})
all_extracted = []

if uploaded_files:
    coords_list = get_fixed_coords()

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
            img_height, img_width = img_cv.shape[:2]

            # OCR実行
            if uploaded_file.name not in st.session_state.ocr_cache:
                ocr_results = run_ocr_once(img_cv)
                st.session_state.ocr_cache[uploaded_file.name] = ocr_results
            else:
                ocr_results = st.session_state.ocr_cache[uploaded_file.name]

            rects = detect_graph_rectangles(img_gray)
            st.markdown(f"検出グラフ数：{len(rects)}個")
            machine_name = extract_machine_name_by_lines(ocr_results)
            samai_results = extract_samai_by_fixed_coords(ocr_results, coords_list, img_width, img_height)

            machine_results[machine_name]["total_count"] += len(rects)

            for idx, (x, y, w, h) in enumerate(rects):
                crop = img_cv[y:y + h, x:x + w]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)

                if idx < len(samai_results):
                    samai_value = samai_results[idx][1]
                    samai_text = samai_results[idx][2]
                else:
                    samai_value = None
                    samai_text = "不明"

                red_detected = has_red_area(crop)
                red_status = "〇赤あり" if red_detected else "×赤なし"

                # 出力結果フィルタ用
                if red_detected and samai_value is not None:
                    machine_results[machine_name]["entries"].append(samai_value)

                # PILで直接描画
                annotated_img = draw_text_on_pil_image(
                    pil_crop.copy(),
                    f"{machine_name} グラフ {idx + 1}",
                    f"OCR結果: {samai_text} / {red_status}"
                )

                all_extracted.append({
                    "image": annotated_img,
                    "samai_value": samai_value
                })

        except Exception as e:
            st.error(f"エラー発生: {e}")

# ✅ 出力結果
if machine_results:
    st.subheader("📊 出力結果")
    output_texts = []
    for machine, data in machine_results.items():
        filtered_samai = [samai for samai in data["entries"] if samai >= threshold]
        filtered_samai.sort(reverse=True)
        header = f"▼{machine} ({len(filtered_samai)}/{data['total_count']})"
        output_texts.append(header)
        for samai in filtered_samai:
            if samai >= 19000:
                output_texts.append(f"㊗️{samai}枚 コンプ！")
            elif samai >= 10000:
                output_texts.append(f"🎉{samai}枚")
            elif samai >= 8000:
                output_texts.append(f"🚨{samai}枚")
            elif samai >= 5000:
                output_texts.append(f"✨{samai}枚")
            else:
                output_texts.append(f"・{samai}枚")
        output_texts.append("")
    st.code("\n".join(output_texts), language="")

# ✅ グラフ結果を4列で表示
cols = st.columns(4)
for idx, item in enumerate(all_extracted):
    col = cols[idx % 4]
    with col:
        col.image(item["image"], use_container_width=True)
