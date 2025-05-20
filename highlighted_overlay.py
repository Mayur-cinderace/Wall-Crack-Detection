import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.badges import badge
import matplotlib.pyplot as plt

SERVER_URL = 'http://127.0.0.1:11434/v1'

def detect_cracks(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    total_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h + 1)
            if aspect_ratio < 4:
                cv2.drawContours(output, [cnt], -1, (0, 0, 255), 2)
                total_area += area

    return output, total_area

def classify_severity(area: float):
    if area < 1000:
        return 'Low (Hairline)', 'Monitor periodically; cosmetic treatment if desired.'
    elif area < 5000:
        return 'Medium (Moderate)', 'Apply crack filler and sealant; monitor for progression.'
    else:
        return 'High (Severe)', 'Structural repair required: routing and sealing, injection, or reinforcement.'

def get_repair_suggestions(severity: str):
    model_name = 'llama3.2:latest'
    endpoint = f"{SERVER_URL}/chat/completions"
    prompt_text = (
        f"You are a civil engineer specializing in structural maintenance. A wall crack has been detected and "
        f"classified as '{severity}' severity. Based on Indian civil infrastructure codes—specifically IRC "
        f"(Indian Roads Congress) and IS codes by BIS (Bureau of Indian Standards)—provide a detailed, code-compliant "
        f"repair recommendation. Include:\n"
        "- Appropriate IRC and IS code references (like IRC:SP:62, IS 456:2000, IS 3370, etc. if applicable)\n"
        "- Suitable repair methods and their rationale\n"
        "- List of tools and materials required\n"
        "- Step-by-step repair process\n"
        "- Safety precautions\n"
        "- Monitoring or inspection advice post-repair\n\n"
        "Be concise, technically accurate, and strictly aligned with Indian engineering standards."
    )
    payload = {
        'model': model_name,
        'messages': [{'role': 'user', 'content': prompt_text}],
        'temperature': 0.7,
        'max_tokens': 600
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        choice = data['choices'][0]
        if 'message' in choice:
            return choice['message'].get('content', '').strip()
        return choice.get('text', '').strip()
    except requests.RequestException as e:
        return f"Error generating suggestions: {e}"

def main():
    st.set_page_config(page_title="Crack Detector", layout="wide")
    st.markdown("""
        <style>
            .main-title {
                font-size: 2.5rem;
                font-weight: bold;
                color: #00416A;
            }
            .sub-text {
                font-size: 1.2rem;
                color: #333;
            }
            .metric-box {
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                background-color: #f9f9f9;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>🧱 Wall Crack Detection & Repair Advisory (IRC + IS Codes)</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>Upload an image of a wall to detect cracks, evaluate severity, and get BIS/IRC-compliant repair recommendations.</div>", unsafe_allow_html=True)

    add_vertical_space(1)
    uploaded_file = st.file_uploader("📄 Upload JPEG/PNG Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(img)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption='📷 Original Image', use_column_width=True)

        processed, area = detect_cracks(image_np)

        with col2:
            st.image(processed, caption='🧠 Detected Cracks', use_column_width=True)

        severity, brief = classify_severity(area)

        st.markdown("""
            <div class='metric-box'>
                <strong>Total Crack Area:</strong> {:.2f} pixels<br>
                <strong>Severity Level:</strong> {}<br>
                <strong>Brief Recommendation:</strong> {}
            </div>
        """.format(area, severity, brief), unsafe_allow_html=True)

        add_vertical_space(1)
        if st.button("🔍 Get Detailed Repair Suggestions (IRC/IS)"):
            with st.spinner('🛠️ Generating repair recommendations...'):
                suggestions = get_repair_suggestions(severity)
            st.subheader("📁 IRC/IS Based Repair Suggestions")
            st.write(suggestions)

    add_vertical_space(2)
    st.markdown("---")
    st.markdown("Built with Wall Profiling")

if __name__ == '__main__':
    main()
