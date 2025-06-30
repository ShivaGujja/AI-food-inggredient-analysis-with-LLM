import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import ollama

# --- SETUP ---
MODEL_PATH = r"C:\Minor_Git\Minor-Project\runscopy\detect\train4\weights\best.pt"  # Update with your trained model path
model = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(["en"], gpu=False)

st.set_page_config(page_title="Ingredient Analyzer", layout="centered")
st.title("🧪 Ingredient Health Analyzer")

st.markdown("""
1. Upload a **food package image**  
2. The model detects and extracts the **ingredient list**  
3. The local LLM analyzes health impacts and returns suggestions  
""")

# --- Local LLM helper ---
def call_local_llm(prompt):
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a food package image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path, caption="Uploaded Image", use_column_width=True)

    st.info("🔍 Detecting ingredient box...")
    results = model.predict(source=image_path, conf=0.3, verbose=False)
    r = results[0]

    if len(r.boxes) == 0:
        st.warning("No ingredient box detected.")
    else:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            img = cv2.imread(image_path)
            crop = img[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            st.image(crop_rgb, caption="Ingredient Region", use_column_width=True)

            st.info("🧾 Extracting text...")
            ocr_result = ocr_reader.readtext(crop_rgb, detail=0)
            extracted_text = " ".join(ocr_result)
            st.text_area("Extracted Ingredients", extracted_text, height=150)

            if extracted_text.strip():
                st.info("💡 Analyzing ingredients with local LLM...")

                prompt = f"""
                Here is a list of food ingredients extracted from a package:

                \"\"\"{extracted_text}\"\"\"

                Analyze this list and tell if the product is healthy or not. Mention any harmful ingredients, possible long-term health impacts, and suggest healthier alternatives if needed. Your response should be clear and easy to understand for general users. Rate the overall healthiness of the product on a scale of 1 to 10, where 1 is very unhealthy and 10 is very healthy.
                """
                analysis = call_local_llm(prompt)
                st.markdown("### 🧠 Health Analysis")
                st.write(analysis)

            else:
                st.warning("No text found in the detected region.")
