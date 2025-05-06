import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
from PIL import Image
import toml

# --- API キー読み込み ---
config = toml.load("config.toml")
API_KEY = config["google"]["api_key"]

# --- Google Cloud Vision API リクエスト関数 ---
def call_vision_api(image_content):
    vision_url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    body = {
        "requests": [{
            "image": {
                "content": image_content
            },
            "features": [{
                "type": "TEXT_DETECTION"
            }]
        }]
    }
    response = requests.post(vision_url, headers=headers, data=json.dumps(body))
    return response.json()

# --- Streamlit UI ---
st.title("Google Cloud Vision OCR")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_container_width=True)

    # 画像のBase64エンコード
    buffered = st.file_uploader("再アップロード", type=["png", "jpg", "jpeg"])
    if buffered:
        img = Image.open(buffered)
        img_bytes = buffered.read()
    else:
        buffered = uploaded_file
        img_bytes = buffered.read()

    encoded_image = base64.b64encode(img_bytes).decode()

    # Vision API 実行
    st.info("OCR を実行中...")
    result = call_vision_api(encoded_image)

    try:
        text_annotations = result["responses"][0]["textAnnotations"]
        full_text = text_annotations[0]["description"]
        st.success("OCR 結果")
        st.text_area("認識されたテキスト", full_text, height=200)
    except Exception as e:
        st.error("OCR結果の取得に失敗しました。")
        st.write(result)