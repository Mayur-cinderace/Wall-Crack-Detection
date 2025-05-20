import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.badges import badge

SERVER_URL = 'http://127.0.0.1:11434/v1'

def calibrate_scale(px_length: float, real_mm: float) -> float:
    """
    Compute mm-per-pixel from a known real-world length.
    """
    if px_length <= 0:
        st.error("Calibration pixel length must be > 0.")
        return 1.0
    return real_mm / px_length


def segment_cracks(image: np.ndarray) -> np.ndarray:
    """
    Adaptive Canny-based segmentation to extract crack edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    sigma = 0.33
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lo, hi)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def compute_area_px(mask: np.ndarray) -> float:
    """
    Total crack pixel area via contour fill.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(cv2.contourArea(cnt) for cnt in contours)


def classify_severity(area_mm2: float) -> tuple[str, str]:
    if area_mm2 < 50:
        return 'Low (Hairline)', 'Monitor periodically; cosmetic treatment if desired.'
    elif area_mm2 < 250:
        return 'Medium (Moderate)', 'Apply crack filler and sealant; monitor for progression.'
    else:
        return 'High (Severe)', 'Structural repair required: routing and sealing, injection, or reinforcement.'


def get_repair_suggestions(severity: str):
    prompt = (
        f"You are a civil engineer specializing in structural maintenance. A wall crack has been detected and "
        f"classified as '{severity}' severity. Based on Indian civil infrastructure codes—specifically IRC "
        f"(Indian Roads Congress) and IS codes by BIS (Bureau of Indian Standards)—provide a detailed, code-compliant "
        "repair recommendation. Include IRC/IS references, methods, tools, materials, step-by-step process, safety "
        "precautions, and monitoring advice."
    )
    payload = {
        'model': 'llama3.2:latest',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
        'max_tokens': 600
    }
    try:
        resp = requests.post(f"{SERVER_URL}/chat/completions", json=payload, timeout=60)
        resp.raise_for_status()
        msg = resp.json()['choices'][0].get('message', resp.json()['choices'][0])
        return msg.get('content', msg.get('text', '')).strip()
    except Exception as e:
        return f"Error: {e}"

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="🧱 Crack Detector with Scale", layout="wide")

    st.markdown("""
    <style>
      .main-title {font-size:2.5rem; font-weight:bold; color:#00416A;}
      .sub-text {font-size:1.2rem; color:#333;}
      .metric-box {border:1px solid #ddd; border-radius:10px; padding:10px; background:#f9f9f9;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-title'>🧱 Wall Crack Detection & Measurement</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-text'>Upload an image, calibrate scale, detect cracks, and get area in mm² & severity.</div>", unsafe_allow_html=True)
    add_vertical_space(1)

    uploaded_file = st.file_uploader("📤 Upload JPEG/PNG Image", type=["jpg","jpeg","png"])
    if not uploaded_file:
        return

    img = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(img)
    st.image(img, caption='Original Image', use_column_width=True)

    # Sidebar calibration
    st.sidebar.header("📐 Calibration")
    real_mm = st.sidebar.number_input("Real-world length (mm)", min_value=1.0, value=100.0)
    px_len = st.sidebar.number_input("Measured pixel length", min_value=1.0, value=200.0)

    mm_per_px = calibrate_scale(px_len, real_mm)
    mask = segment_cracks(image_np)
    area_px = compute_area_px(mask)
    area_mm2 = area_px * (mm_per_px ** 2)
    severity, brief = classify_severity(area_mm2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(mask, caption='Detected Crack Mask', channels='GRAY', use_column_width=True)
    with col2:
        st.markdown(f"""
        <div class='metric-box'>
          <strong>Area:</strong> {area_mm2:.1f} mm²<br>
          <strong>Severity:</strong> {severity}<br>
          <strong>Note:</strong> {brief}
        </div>
        """, unsafe_allow_html=True)

    add_vertical_space(1)
    if st.button("🔍 Get Detailed Repair Suggestions (IRC/IS)"):
        with st.spinner('🛠️ Generating repair recommendations...'):
            suggestions = get_repair_suggestions(severity)
        st.subheader("📑 IRC/IS Based Repair Suggestions")
        st.write(suggestions)

    add_vertical_space(2)
    st.markdown("---")
    st.markdown("wall crack detection with Profiling.")

if __name__ == '__main__':
    main()
