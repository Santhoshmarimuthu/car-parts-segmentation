import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import math
from PIL import Image
import io
import tempfile
import os
import gdown

# Page configuration
st.set_page_config(
    page_title="Car Parts Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .title-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .title-text {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle-text {
        color: #f0f0f0;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .stCard {
        border-radius: 10px;
        padding: 1.5rem;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(102,126,234,0.4);
    }
    
    /* Upload box styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="title-container">
    <h1 class="title-text">Car Parts Detection System</h1>
    <p class="subtitle-text">Advanced AI-Powered Vehicle Component Segmentation</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("###  Configuration")
    st.markdown("---")
    
    # Model path input
    model_path = st.text_input(
        "Model Path",
        value="https://drive.google.com/file/d/1xFWWvq3tfyXGOVUnP5owWbz1ND0g3tyM/view?usp=sharing",
        help="Path to your trained YOLOv11 model"
    )
    
    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for detection"
    )
    
    # Image size
    img_size = st.select_slider(
        "Image Size",
        options=[320, 416, 512, 640, 768, 896, 1024],
        value=640,
        help="Input image size for model"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application uses YOLOv11 for real-time detection and segmentation "
        "of car parts. Upload an image to identify components like headlights, "
        "bumpers, doors, and more."
    )

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'results' not in st.session_state:
    st.session_state.results = None

# Load model function
@st.cache_resource
def load_model(model_path):
    try:
        # Check if the path is a Google Drive link
        if "drive.google.com" in model_path:
            # Extract file ID from URL
            import re
            match = re.search(r"/d/([a-zA-Z0-9_-]+)", model_path)
            if match:
                file_id = match.group(1)
                local_path = "yolov11m-ft.pt"
                
                if not os.path.exists(local_path):
                    st.info("Downloading model weights from Google Drive...")
                    gdown.download(f"https://drive.google.com/uc?id={file_id}", local_path, quiet=False)
                model_path = local_path
            else:
                st.error("Invalid Google Drive link")
                return None
        
        # Load YOLO model
        model = YOLO(model_path)
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main content area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload Car Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a car for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Analyze button
        if st.button(" Analyze Image", type="primary"):
            with st.spinner("Loading model and analyzing image..."):
                # Load model
                model = load_model(model_path)
                
                if model is not None:
                    # Save temp file for YOLO
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_path = tmp_file.name
                        cv2.imwrite(tmp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    
                    # Run inference
                    results = model.predict(tmp_path, imgsz=img_size, conf=conf_threshold)
                    st.session_state.results = results[0]
                    st.session_state.original_image = img_array
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    st.success(" Analysis complete!")
                    st.rerun()

with col2:
    st.markdown("###  Detection Results")
    
    if st.session_state.results is not None:
        result = st.session_state.results
        orig_img = st.session_state.original_image
        
        # Display metrics
        num_detections = len(result.boxes)
        unique_classes = len(set(result.boxes.cls.cpu().numpy().astype(int)))
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_detections}</div>
                <div class="metric-label">Parts Detected</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{unique_classes}</div>
                <div class="metric-label">Unique Classes</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Draw bounding boxes
        img_with_boxes = orig_img.copy()
        class_names = result.names
        
        # Fixed colors for consistency
        np.random.seed(42)
        fixed_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) 
                       for i in range(len(class_names))}
        
        if len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                color = fixed_colors[class_ids[i]]
                label = f"{class_names[class_ids[i]]} {confidences[i]:.2f}"
                
                # Draw box
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                text_w, text_h = text_size
                cv2.rectangle(img_with_boxes, 
                            (x1, max(y1 - text_h - 10, 0)), 
                            (x1 + text_w + 10, y1), 
                            color, -1)
                
                # Draw label text
                cv2.putText(img_with_boxes, label, 
                          (x1 + 5, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        st.image(img_with_boxes, caption="Detected Parts", use_container_width=True)
    else:
        st.info("Upload an image and click 'Analyze' to see results")

# Cropped parts section
if st.session_state.results is not None and len(st.session_state.results.boxes) > 0:
    st.markdown("---")
    st.markdown("### Individual Parts")
    
    result = st.session_state.results
    orig_img = st.session_state.original_image
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names
    confidences = result.boxes.conf.cpu().numpy()
    
    # Create grid of cropped images
    num_parts = len(boxes)
    cols = min(4, num_parts)
    
    for row_start in range(0, num_parts, cols):
        cols_list = st.columns(cols)
        for idx, col in enumerate(cols_list):
            part_idx = row_start + idx
            if part_idx < num_parts:
                x1, y1, x2, y2 = boxes[part_idx]
                cropped = orig_img[y1:y2, x1:x2]
                
                with col:
                    st.image(cropped, use_container_width=True)
                    st.markdown(
                        f"<div style='text-align: center; font-weight: 600; color: #667eea;'>"
                        f"{class_names[class_ids[part_idx]]}</div>"
                        f"<div style='text-align: center; font-size: 0.85rem; color: #888;'>"
                        f"Confidence: {confidences[part_idx]:.2%}</div>",
                        unsafe_allow_html=True
                    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; padding: 2rem 0;'>"
    "Powered by YOLOv11 • Built with Streamlit • © 2025"
    "</div>",
    unsafe_allow_html=True
)