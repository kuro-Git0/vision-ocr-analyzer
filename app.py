import streamlit as st
st.set_page_config(layout="wide", page_title="🎰 パチスログラフ解析アプリ")

import os
import io
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

# 画像処理関数
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