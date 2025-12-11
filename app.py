import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json
import argparse

# -----------------------
# FLASK IMPORTS
# -----------------------
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
#  PART 1 — YOUR ORIGINAL ACD MEASUREMENT LOGIC (UNTOUCHED)
# ============================================================

def process_acd(image_bytes):
    try:
        # Read image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Blur
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Edge detection
        edges = cv2.Canny(blur, 40, 100)

        # Cornea detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"error": "No edges detected (cornea not found)"}

        cornea_contour = min(contours, key=lambda c: cv2.boundingRect(c)[0])
        x_cornea, y, w, h = cv2.boundingRect(cornea_contour)
        cornea_pos = x_cornea + w // 2

        # Pupil detection
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        pupil_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not pupil_contours:
            return {"error": "Pupil not detected"}

        pupil_contour = max(pupil_contours, key=cv2.contourArea)
        M = cv2.moments(pupil_contour)
        if M['m00'] == 0:
            return {"error": "Pupil center calculation failed"}

        pupil_x = int(M['m10'] / M['m00'])
        pupil_y = int(M['m01'] / M['m00'])

        # Measurement
        pixel_distance = pupil_x - cornea_pos
        scaling_factor = 0.01
        acd_estimate = (pixel_distance * scaling_factor) + 0.8

        # Visualization Image
        debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.line(debug_img, (cornea_pos, pupil_y), (pupil_x, pupil_y), (0,255,0), 2)
        cv2.circle(debug_img, (cornea_pos, pupil_y), 5, (255,0,0), -1)
        cv2.circle(debug_img, (pupil_x, pupil_y), 5, (0,0,255), -1)

        _, buffer = cv2.imencode(".png", debug_img)
        encoded_debug = base64.b64encode(buffer).decode("utf-8")

        return {
            "pixel_distance": int(pixel_distance),
            "acd_mm": float(round(acd_estimate, 3)),
            "cornea_x": int(cornea_pos),
            "pupil_x": int(pupil_x),
            "pupil_y": int(pupil_y),
            "debug_image": encoded_debug
        }

    except Exception as e:
        return {"error": str(e)}

# ============================================================
#  PART 2 — STREAMLIT UI (UNCHANGED FROM YOUR ORIGINAL CODE)
# ============================================================

def run_streamlit():
    st.set_page_config(page_title="ACD Estimator", layout="wide")
    st.title("🩺 Anterior Chamber Depth (ACD) Measurement")
    st.markdown("""
    Upload a **side-view eye image** taken from **6 cm distance** to estimate ACD.
    The app detects corneal edge and pupil position to calculate the distance.
    """)

    uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        result = process_acd(uploaded_file.read())

        if "error" in result:
            st.error(result["error"])
            return

        st.subheader("📐 Measurement Results")
        st.metric("Estimated ACD", f"{result['acd_mm']:.2f} mm")

        debug_bytes = base64.b64decode(result["debug_image"])
        st.image(debug_bytes, caption="Detection Visualization", use_column_width=True)

    else:
        st.info("ℹ️ Please upload an eye image to begin analysis")

# ============================================================
#  PART 3 — FLASK API BACKEND
# ============================================================

def create_api_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/predict_acd", methods=["POST"])
    def predict_acd():
        if "file" not in request.files:
            return jsonify({"error": "Missing file parameter"}), 400

        img_bytes = request.files["file"].read()
        result = process_acd(img_bytes)

        return jsonify(result)

    return app

# ============================================================
#  PART 4 — MAIN ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Run Flask API instead of Streamlit UI")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.api:
        app = create_api_app()
        print(f"🚀 Flask API running on http://0.0.0.0:{args.port}")
        app.run(host="0.0.0.0", port=args.port)
    else:
        run_streamlit()
