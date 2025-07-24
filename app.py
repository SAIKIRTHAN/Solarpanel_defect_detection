import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO
import tensorflow as tf

# ---- Function: Predict potential issue from real-time input ----
def get_potential_issue(temp, voltage, current, irradiance):
    if temp > 60:
        return "⚠️ Overheating risk! May reduce panel efficiency."
    elif voltage < 15 and irradiance > 600:
        return "⚠️ Possible electrical fault or shading issue."
    elif current < 3 and irradiance > 700:
        return "⚠️ Low current output — potential cell damage."
    else:
        return "✅ Sensor data looks normal."

# ---- Load Models ----
classifier = tf.keras.models.load_model("solar_panel_classifier.h5", compile=False)
detector = YOLO("runs/detect/train3/weights/best.pt")

# ---- Class Labels and Recommendations ----
class_names = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-covered"]
recommendations = {
    "Clean": "✅ Panel is clean. No maintenance needed.",
    "Dusty": "🧽 Recommend soft water cleaning within 2 days to restore efficiency.",
    "Bird-drop": "🧼 Remove bird droppings gently. Use water and soft cloth to avoid scratches.",
    "Snow-covered": "❄️ Clear snow using soft brushes or warm air to avoid prolonged energy loss.",
    "Electrical-damage": "⚠️ Inspect electrical connections and replace faulty components. Involve technician.",
    "Physical-Damage": "🔧 Replace or repair the damaged module. Avoid using cracked panels to prevent hazards."
}

# ---- Streamlit UI ----
st.set_page_config(page_title="Solar Panel Inspection", layout="wide")
st.title("🔆 Solar Panel Defect Detection & Monitoring")

# ---- Image Upload ----
uploaded = st.file_uploader("📤 Upload a solar panel image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    # Convert for classifier
    img_array = np.array(image)
    resized = cv2.resize(img_array, (224, 224)) / 255.0
    expanded = np.expand_dims(resized, axis=0)

    # Predict class
    prediction = classifier.predict(expanded).argmax()
    predicted_class = class_names[prediction]

    # Save image temporarily for YOLOv8
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        results = detector(temp.name)[0].plot()

    # ---- Show Images Side by Side ----
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("### 📍 YOLOv8 Detected Defects")
        st.image(results, use_container_width=True)

    # ---- Show Predictions and Maintenance Recommendation ----
    st.markdown(f"### 🧠 Predicted Class: **{predicted_class}**")
    st.info(recommendations[predicted_class])

    # ---- Real-time Data Input ----
    st.markdown("## 🧾 Real-time Sensor Data (Optional)")
    col_t, col_v, col_c, col_i = st.columns(4)
    with col_t:
        temp = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=100.0, value=35.0)
    with col_v:
        voltage = st.number_input("🔌 Voltage (V)", min_value=0.0, max_value=40.0, value=18.0)
    with col_c:
        current = st.number_input("🔋 Current (A)", min_value=0.0, max_value=20.0, value=5.0)
    with col_i:
        irradiance = st.number_input("🔆 Irradiance (W/m²)", min_value=0.0, max_value=1200.0, value=800.0)

    # ---- Predict Issue from Real-time Data ----
    issue = get_potential_issue(temp, voltage, current, irradiance)
    st.markdown("### 🔍 Real-time Sensor Diagnosis:")
    st.warning(issue if "⚠️" in issue else issue)
