# 補正キーは常に name_a（OCR抽出の元名称）を使う
key_name = f"{machine}_graph_{idx + 1}"

# 初期化（変更後も維持される）
if key_name not in st.session_state.manual_corrections:
    st.session_state.manual_corrections[key_name] = ""

# 保存
machine_results.append({
    "machine": display,  # 表示には display_name
    "graph_number": idx + 1,
    "image": Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
    "samai_value": samai[idx][1] if idx < len(samai) else None,
    "samai_text": samai[idx][2] if idx < len(samai) else "不明",
    "red_status": "〇赤あり" if has_red_area(crop) else "×赤なし",
    "manual_key": key_name  # 補正には machine_name
})