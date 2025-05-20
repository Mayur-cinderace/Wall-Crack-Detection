import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import requests
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.badges import badge

SERVER_URL = 'http://127.0.0.1:11434/v1'

def calibrate_scale(px_length: float, real_mm: float) -> float:
    if px_length <= 0:
        st.error("Calibration pixel length must be > 0.")
        return 1.0
    return real_mm / px_length


def segment_cracks(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    sigma = 0.33
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lo, hi)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def compute_area_px(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(cv2.contourArea(cnt) for cnt in contours)


def classify_severity(area_mm2: float) -> tuple[str, str]:
    if area_mm2 < 50:
        return 'Low (Hairline)', 'Monitor periodically; cosmetic treatment if desired.'
    elif area_mm2 < 250:
        return 'Medium (Moderate)', 'Apply crack filler and sealant; monitor for progression.'
    else:
        return 'High (Severe)', 'Structural repair required: routing and sealing, injection, or reinforcement.'


def get_repair_suggestions(severity: str) -> str:
    prompt = (
        f"You are a civil engineer specializing in structural maintenance. "
        f"A wall crack has been detected and classified as '{severity}'. "
        "Based on IRC/SP:62 and IS codes, provide a detailed repair plan."
    )
    payload = {
        'model': 'llama3.2:latest',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
        'max_tokens': 800
    }
    try:
        resp = requests.post(f"{SERVER_URL}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        msg = resp.json()['choices'][0].get('message', resp.json()['choices'][0])
        return msg.get('content', msg.get('text', '')).strip()
    except Exception as e:
        return f"Error: {e}"

# Streamlit App
def main():
    st.set_page_config(page_title="🧱 Crack Detector with Video", layout="wide")
    st.markdown(
        """
        <style>
          .main-title {font-size:2.5rem; font-weight:bold; color:#00416A;}
          .sub-text {font-size:1.2rem; color:#333;}
          .metric-box {border:1px solid #ddd; border-radius:10px; padding:10px; background:#f9f9f9;}
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<div class='main-title'>🧱 Wall Crack Detection & Measurement</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>Choose image or video mode, calibrate scale, and analyze cracks.</div>", unsafe_allow_html=True)
    add_vertical_space(1)

    mode = st.sidebar.radio("Mode", ['Image', 'Video'])
    st.sidebar.header("📐 Calibration")
    real_mm = st.sidebar.number_input("Real-world length (mm)", min_value=1.0, value=100.0)
    px_len = st.sidebar.number_input("Measured pixel length", min_value=1.0, value=200.0)
    display = st.sidebar.radio("Display Mode", ['Grayscale Mask', 'Highlighted Overlay'])
    mm_per_px = calibrate_scale(px_len, real_mm)

    if mode == 'Image':
        uploaded = st.file_uploader("📤 Upload JPEG/PNG", type=['jpg','jpeg','png'])
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            image_np = np.array(img)
            st.image(img, caption='Original', use_container_width=True)
            mask = segment_cracks(image_np)
            area_mm2 = compute_area_px(mask) * (mm_per_px ** 2)
            sev, note = classify_severity(area_mm2)
            col1, col2 = st.columns(2)
            with col1:
                if display == 'Grayscale Mask':
                    st.image(mask, caption='Mask', channels='GRAY', use_container_width=True)
                else:
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    ov = image_np.copy()
                    cv2.drawContours(ov, cnts, -1, (255, 0, 0), 2)
                    st.image(ov, caption='Overlay', use_container_width=True)
            with col2:
                st.markdown(
                    f"""
                    <div class='metric-box'>
                      <strong>Area:</strong> {area_mm2:.1f} mm²<br>
                      <strong>Severity:</strong> {sev}<br>
                      <strong>Note:</strong> {note}
                    </div>
                    """, unsafe_allow_html=True
                )
            if st.button("🔍 Repair Suggestions"):
                recs = get_repair_suggestions(sev)
                st.write(recs)
    else:
        uploaded_vid = st.file_uploader("📤 Upload Video (mp4/avi/mov)", type=['mp4','avi','mov'])
        if uploaded_vid:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.avi')
            tfile.write(uploaded_vid.read())
            if st.button("▶ Process Video"):
                cap = cv2.VideoCapture(tfile.name)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.avi').name
                out = cv2.VideoWriter(out_path, fourcc, 20.0, (640, 480), True)
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (640, 480))
                    mask = segment_cracks(frame)
                    if display == 'Grayscale Mask':
                        disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    else:
                        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        disp = frame.copy()
                        cv2.drawContours(disp, cnts, -1, (255, 0, 0), 1)
                    out.write(disp)
                    stframe.image(disp, use_container_width=True)
                cap.release()
                out.release()
                st.success("Processing complete. Playing processed video below.")
                st.video(out_path)

    add_vertical_space(2)
    badge(type="github", name="Your GitHub")
    st.markdown("---")
    st.markdown("Built by You")

if __name__ == '__main__':
    main()
