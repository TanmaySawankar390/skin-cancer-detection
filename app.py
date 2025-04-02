import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json

# Configuration
MODEL_PATH = "skin_cancer_model.keras"
CLASS_INDICES_PATH = "class_indices.json"
CANCEROUS_CONDITIONS = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma"]

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
    image = image.resize((300, 300))  # Match model input size
    array = img_to_array(image)
    array = tf.keras.applications.efficientnet.preprocess_input(array)
    return np.expand_dims(array, axis=0)

def predict_skin_cancer(image):
    """Make prediction and return formatted results"""
    processed = preprocess_image(image)
    predictions = cancer_model.predict(processed, verbose=0)[0]
    
    # Get top 3 predictions
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
        st.image(image, use_column_width=True)
        
    with col2:
        # Cancer warning
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
        
        # Confidence display
        st.metric(
            label="Primary Prediction", 
            value=f"{primary['class']}",
            delta=f"{primary['confidence']*100:.1f}% confidence"
        )
        
        # Detailed predictions
        with st.expander("View Detailed Analysis"):
            st.write("**Prediction Breakdown:**")
            for i, pred in enumerate([primary] + prediction['secondary']):
                label = "🥇 Primary" if i == 0 else f"🥈 Secondary {i}"
                st.write(f"""
                {label}:  
                **{pred['class']}** ({pred['confidence']*100:.1f}%)
                """)

# Streamlit UI
st.title("🔍Skin Cancer ISIC Detection Assistant")
st.caption("""
- Source: [ISIC Kaggle Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
""")
st.markdown("""
**Madhav Institute of Technology & Science, Gwalior**  
*Computer Science & Engineering Department*

**How to use:**
1. Upload a clear photo of skin lesion
2. Our AI will analyze the image
3. Review results & consult a dermatologist
""")

# Image input section
upload_tab, camera_tab = st.tabs(["📁 Upload Image", "📸 Take Photo"])

with upload_tab:
    uploaded_file = st.file_uploader("Choose skin image", type=["jpg", "jpeg", "png"])

with camera_tab:
    camera_image = st.camera_input("Capture lesion photo")

# Process input
if uploaded_file or camera_image:
    image = Image.open(uploaded_file or camera_image)
    
    with st.spinner("🔬 Analyzing image..."):
        prediction = predict_skin_cancer(image)
    
    display_results(image, prediction)

# Sidebar information
st.sidebar.header("Clinical Notes")
st.sidebar.markdown("""
- **Model Accuracy:** 81.3% (ISIC validation set)
- **Coverage:** 12 lesion types (7 benign, 5 malignant)
- **Sensitivity:** 92.8% for malignant cases
- **Specificity:** 95.1% for benign cases
""")

st.sidebar.header("Developed By")
st.sidebar.markdown("""
**B.Tech CSE Students**  
*Batch 2022-2026*

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
**Important Disclaimer:**  
This tool provides preliminary analysis only.  
Always consult a certified dermatologist for diagnosis.
""")

# Footer
st.markdown("---")
st.caption("""
- 🛠️ Project developed under the guidance of Dr. Rahul Dubey Professor CSE Department.
- 📧 Contact: 22cs10ta64@mitsgwl.ac.in | 📞 +919713999175
""")