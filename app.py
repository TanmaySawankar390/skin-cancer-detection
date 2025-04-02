import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import os

# Configuration
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

# Load model and class info
cancer_model = load_cancer_model()

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
CLASS_NAMES = list(class_indices.keys())

def preprocess_image(image):
    """Process image for model prediction"""
    image = image.resize((300, 300))
    array = img_to_array(image)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return np.expand_dims(array, axis=0)

def predict_skin_cancer(image):
    """Make prediction and return formatted results"""
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
    """Display prediction results with visual feedback"""
    primary = prediction['primary']
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(image, use_column_width=True, caption="Uploaded Image")
        
    with col2:
        if any(cancer_type in primary['class'] for cancer_type in CANCEROUS_CONDITIONS):
            st.error("""
            ⚠️ **Potential Cancer Detected!**
            Urgent medical consultation recommended!
            """)
        else:
            st.success("""
            ✅ **Likely Benign**
            Regular monitoring still advised
            """)
        
        st.metric(
            label="Primary Prediction", 
            value=f"{primary['class']}",
            delta=f"{primary['confidence']*100:.1f}% confidence"
        )
        
        with st.expander("View Detailed Analysis"):
            st.write("**Prediction Breakdown:**")
            for i, pred in enumerate([primary] + prediction['secondary']):
                label = "🥇 Primary" if i == 0 else f"🥈 Secondary {i}"
                st.write(f"""
                {label}:  
                **{pred['class']}** ({pred['confidence']*100:.1f}%)
                """)

def load_sample_images():
    """Load sample images with improved error handling"""
    samples = []
    if os.path.exists(SAMPLE_IMAGES_DIR):
        for file in sorted(os.listdir(SAMPLE_IMAGES_DIR)):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                sample_path = os.path.join(SAMPLE_IMAGES_DIR, file)
                try:
                    with Image.open(sample_path) as img:
                        samples.append({
                            'path': sample_path,
                            'name': " ".join(os.path.splitext(file)[0].split("_")).title()
                        })
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
    return samples

# Streamlit UI Configuration
# st.set_page_config(
#     page_title="Skin Cancer ISIC Detection Assistant",
#     page_icon="🩺",
#     layout="wide"
# )

# Main App Interface
st.title("🔍 Skin Cancer ISIC Detection Assistant")
st.markdown("""
**Madhav Institute of Technology & Science, Gwalior**  
*Computer Science & Engineering Department*
""")

with st.expander("ℹ️ How to Use", expanded=True):
    st.markdown("""
    1. **Test Samples**: Try pre-loaded examples in the 🖼️ Sample Images tab  
    2. **Upload**: Use 📁 Upload Image for existing photos  
    3. **Camera**: Capture new images with 📸 Take Photo  
    4. **Review**: Get instant analysis with medical guidance  
    """)

# Image Input Section
upload_tab, sample_tab = st.tabs(["📁 Upload Image", "🖼️ Sample Images"])

# Session State Management
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Sample Images Tab
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
            if (idx + 1) % 3 == 0:
                st.markdown("---")
    else:
        st.warning("No sample images found in 'sample_images' directory")

# Upload Tab
with upload_tab:
    uploaded_file = st.file_uploader("Choose skin image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.session_state.current_image = Image.open(uploaded_file)

# # Camera Tab
# with camera_tab:
#     camera_image = st.camera_input("Capture lesion photo")
#     if camera_image:
#         st.session_state.current_image = Image.open(camera_image)

# Processing and Results Display
if st.session_state.current_image:
    if st.session_state.prediction is None:
        with st.spinner("🔬 Analyzing image..."):
            st.session_state.prediction = predict_skin_cancer(st.session_state.current_image)
    
    display_results(st.session_state.current_image, st.session_state.prediction)
    
    # Add clear button
    if st.button("🧹 Clear Current Analysis"):
        st.session_state.current_image = None
        st.session_state.prediction = None
        st.rerun()

# Sidebar Information
st.sidebar.header("Clinical Notes")
st.sidebar.markdown("""
- **Model Accuracy**: 81.3% (ISIC validation set)
- **Coverage**: 12 lesion types  
- **Sensitivity**: 92.8% (Malignant)  
- **Specificity**: 95.1% (Benign)
""")

st.sidebar.header("Development Team")
st.sidebar.markdown("""
**B.Tech CSE 2022-2026**  
*(Machine Learning Group)*  

👨💻 **Amul Agrawal**  
`0901CS233D03`  

👨💻 **Harshit Varshney**  
`0901CS233D07`  

👨💻 **Lokendra Sharma**  
`0901CS233D08`  

👨💻 **Tanmay Sawnkar**  
`0901CS221139`  
""")

st.sidebar.markdown("""
**Clinical Validation**  
Dr. Rahul Dubey  
Professor, CSE Department  
📧 22cs10ta64@mitsgwl.ac.in  
📞 +91 97139 99175  
""")

# Footer
st.markdown("---")
st.caption("""
🛠️ Skin Cancer Classification System v1.2 | MITS Gwalior  
🔗 [ISIC Dataset Source](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)  
⚠️ **Disclaimer**: This tool provides preliminary analysis only. Always consult a dermatologist for diagnosis.
""")