# 🔆 Solar Panel Defect Detection Using Deep Learning

This project focuses on improving solar panel efficiency and maintenance by detecting and classifying defects through deep learning. It uses a combination of **image classification**, **object detection (YOLOv8)**, **real-time sensor input**, and an interactive **Streamlit web app**.

---

## 📌 Features

- ✅ **Defect Classification** using a CNN model (`solar_panel_classifier.h5`)
- 🔍 **Object Detection** using YOLOv8 (`best.pt`)
- 📤 Upload images via Streamlit app
- 🧠 Real-time sensor input (temperature, voltage, current, irradiance)
- ⚠️ Potential issue prediction from sensor data
- 🧾 Actionable maintenance recommendations
- 📸 Displays annotated image with detected defects

---

## 🧠 Defect Classes

| Class ID | Label               |
|----------|---------------------|
| 0        | Bird-drop           |
| 1        | Clean               |
| 2        | Dusty               |
| 3        | Electrical-damage   |
| 4        | Physical-Damage     |
| 5        | Snow-covered        |


