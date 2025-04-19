import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import os
import io
from dotenv import load_dotenv
import google.generativeai as genai

# === Load environment variables ===
load_dotenv()  # Load environment variables from a .env file

# === Retrieve API Key from Environment Variable ===
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Fetch the API key securely

if GEMINI_API_KEY is None:
    st.error("API Key for Gemini not found! Please set the GEMINI_API_KEY environment variable.")
    st.stop()

# === Gemini API Key Setup ===
genai.configure(api_key=GEMINI_API_KEY)

# === Constants ===
MODEL_PATH = "skin_cancer_model.keras"
CLASS_INDICES_PATH = "class_indices.json"
CANCEROUS_CONDITIONS = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]
SAMPLE_IMAGES_DIR = "sample_images"

@st.cache_resource
def load_cancer_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

cancer_model = load_cancer_model()

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
CLASS_NAMES = list(class_indices.keys())

def check_image_validity_with_gemini(img):
    img_bytes_io = io.BytesIO()
    img.save(img_bytes_io, format='PNG')
    img_bytes = img_bytes_io.getvalue()

    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    response = gemini_model.generate_content([
        "You are a dermatologist AI assistant. Analyze the following image. If the image contains a visible skin lesion, say 'valid skin lesion image'. If it does not contain a skin lesion or is unclear, say 'invalid'. Do not be overly cautious. If the image is of a skin lesion, allow classification.",
        {
            "mime_type": "image/png",
            "data": img_bytes
        }
    ])
    verdict = response.text.lower()
    return "valid skin lesion image" in verdict


# === Preprocess and Predict ===
def preprocess_image(image):
    image = image.resize((300, 300))
    array = img_to_array(image)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return np.expand_dims(array, axis=0)

def predict_skin_cancer(image):
    processed = preprocess_image(image)
    predictions = cancer_model.predict(processed, verbose=0)[0]
    sorted_indices = np.argsort(predictions)[::-1]
    top3 = sorted_indices[:3]
    return {
        "primary": {
            "class": CLASS_NAMES[top3[0]],
            "confidence": float(predictions[top3[0]])
        },
        "secondary": [
            {"class": CLASS_NAMES[top3[1]], "confidence": float(predictions[top3[1]])},
            {"class": CLASS_NAMES[top3[2]], "confidence": float(predictions[top3[2]])}
        ]
    }

def display_results(image, prediction):
    primary = prediction['primary']
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(image, use_column_width=True, caption="Uploaded Image")
    with col2:
        if any(cancer_type in primary['class'] for cancer_type in CANCEROUS_CONDITIONS):
            st.error("⚠️ **Potential Cancer Detected!**\n\nUrgent medical consultation recommended!")
        else:
            st.success("✅ **Likely Benign**\n\nRegular monitoring still advised")
        st.metric("Primary Prediction", f"{primary['class']}", f"{primary['confidence']*100:.1f}% confidence")
        with st.expander("View Detailed Analysis"):
            st.write("**Prediction Breakdown:**")
            for i, pred in enumerate([primary] + prediction['secondary']):
                label = "🥇 Primary" if i == 0 else f"🥈 Secondary {i}"
                st.write(f"{label}: **{pred['class']}** ({pred['confidence']*100:.1f}%)")

def load_sample_images():
    samples = []
    if os.path.exists(SAMPLE_IMAGES_DIR):
        for file in sorted(os.listdir(SAMPLE_IMAGES_DIR)):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    path = os.path.join(SAMPLE_IMAGES_DIR, file)
                    with Image.open(path) as img:
                        samples.append({'path': path, 'name': " ".join(os.path.splitext(file)[0].split("_")).title()})
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
    return samples

# === Streamlit UI ===
st.title("🔍 Skin Cancer ISIC Detection Assistant")
st.markdown("""
**Madhav Institute of Technology & Science, Gwalior**  
*Computer Science & Engineering Department*
""")

with st.expander("ℹ️ How to Use", expanded=True):
    st.markdown("""
    1. **Test Samples**: Try pre-loaded examples in 🖼️ Sample Images tab  
    2. **Upload**: Use 📁 Upload Image for existing photos  
    3. **Review**: Get instant analysis with medical guidance  
    """)

upload_tab, sample_tab = st.tabs(["📁 Upload Image", "🖼️ Sample Images"])

if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# === Sample Tab ===
with sample_tab:
    st.subheader("Test with Sample Images")
    samples = load_sample_images()
    if samples:
        cols = st.columns(3)
        for idx, sample in enumerate(samples):
            with cols[idx % 3]:
                try:
                    img = Image.open(sample['path'])
                    st.image(img, use_column_width=True, caption=sample['name'])
                    if st.button(f"Test {sample['name']}", key=f"sample_{idx}"):
                        st.session_state.current_image = img
                        st.session_state.prediction = None
                except Exception as e:
                    st.error(f"Error loading sample: {str(e)}")
    else:
        st.warning("No sample images found in 'sample_images' directory")

# === Upload Tab ===
with upload_tab:
    uploaded_file = st.file_uploader("Choose skin image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.current_image = Image.open(uploaded_file)

# === Prediction Logic ===
if st.session_state.current_image:
    image = st.session_state.current_image
    with st.spinner("🔍 Validating image..."):
        valid = check_image_validity_with_gemini(image)

    if not valid:
        st.warning("🚫 No valid skin lesion detected. Please upload a clear or valid image.")
    else:
        if st.session_state.prediction is None:
            with st.spinner("🔬 Analyzing image..."):
                st.session_state.prediction = predict_skin_cancer(image)
        display_results(image, st.session_state.prediction)

        if st.button("🧹 Clear Current Analysis"):
            st.session_state.current_image = None
            st.session_state.prediction = None
            st.rerun()

# === Sidebar ===
st.sidebar.header("Clinical Notes")
st.sidebar.markdown("""
- **Model Accuracy**: 81.3% (ISIC validation set)  
- **Coverage**: 9 lesion types  
- **Sensitivity**: 92.8% (Malignant)  
- **Specificity**: 95.1% (Benign)
""")

st.sidebar.header("Development Team")
st.sidebar.markdown("""
**B.Tech CSE 2022-2026**  
*(Machine Learning Group)*  

👨‍💻 **Amul Agrawal** - `0901CS233D03`  
👨‍💻 **Harshit Varshney** - `0901CS233D07`  
👨‍💻 **Lokendra Sharma** - `0901CS233D08`  
👨‍💻 **Tanmay Sawnkar** - `0901CS221139`  
""")

st.sidebar.markdown("""
**Clinical Validation**  
Dr. Rahul Dubey  
Professor, CSE Department  
📧 22cs10ta64@mitsgwl.ac.in  
📞 +91 97139 99175  
""")

st.markdown("---")
st.caption("""
🛠️ Skin Cancer Classification System v1.2 | MITS Gwalior  
🔗 [ISIC Dataset Source](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)  
⚠️ **Disclaimer**: This tool provides preliminary analysis only. Always consult a dermatologist for diagnosis.
""")
