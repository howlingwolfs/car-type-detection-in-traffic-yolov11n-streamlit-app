import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import requests

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Codeminds255: Vehicle Detection & Tracking",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
    }
    .stDownloadButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #238636;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 Codeminds255: Vehicle Detection & Tracking")
st.markdown(
    "An optimized computer vision dashboard powered by **YOLOv8** for real-time vehicle detection, tracking, and counting.")
st.markdown("---")


# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path="best.pt"):
    try:
        if not os.path.exists(model_path):
            st.warning(f"⚠️ '{model_path}' not found. Falling back to 'yolov8n.pt' for demonstration.")
            model_path = "yolov8n.pt"
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


with st.spinner("Loading YOLO model..."):
    model = load_model("best.pt")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Configuration")
media_type = st.sidebar.radio("Select Media Type", ["Image", "Video"])

# Vehicle classes from COCO (2=car, 3=motorbike, 5=bus, 7=truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# File size limits (in MB)
MAX_IMAGE_SIZE_MB = 5
MAX_VIDEO_SIZE_MB = 50

# --- IMAGE PROCESSING ---
if media_type == "Image":
    st.header("🖼️ Image Object Detection")

    use_sample_image = st.sidebar.checkbox("🧪 Use Sample Traffic Image", value=False,
                                           help="Test image detection instantly without uploading a file.")

    image = None
    if use_sample_image:
        st.info("📥 Loading sample traffic image...")
        sample_img_url = "https://images.unsplash.com/photo-1506521781263-d8422e82f27a?auto=format&fit=crop&w=1200&q=80"  # Traffic/Cars sample
        try:
            resp = requests.get(sample_img_url)
            image_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, 1)
            st.success("Sample image loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load sample image: {e}")
    else:
        uploaded_image = st.file_uploader(f"Upload an Image (Max {MAX_IMAGE_SIZE_MB}MB)", type=['png', 'jpg', 'jpeg'])
        if uploaded_image is not None:
            file_size_mb = uploaded_image.size / (1024 * 1024)
            if file_size_mb > MAX_IMAGE_SIZE_MB:
                st.error(
                    f"❌ File size exceeds limit! Uploaded: {file_size_mb:.2f}MB (Max allowed: {MAX_IMAGE_SIZE_MB}MB)")
            else:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)

    if image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Source Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("Detected Objects")
            if st.button("🚀 Run Detection"):
                with st.spinner("Analyzing image..."):
                    results = model(image)
                    annotated_img = results[0].plot()
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.info("Click 'Run Detection' above to process.")

# --- VIDEO PROCESSING ---
elif media_type == "Video":
    st.header("📹 Video Tracking & Counting")

    use_sample_video = st.sidebar.checkbox("🧪 Use Sample Highway Video", value=False,
                                           help="Test tracking instantly without uploading a file.")

    video_path = None
    if use_sample_video:
        st.info("📥 Loading sample highway video for analysis...")
        sample_url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/highway-detection.mp4"
        try:
            response = requests.get(sample_url, stream=True)
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    tfile.write(chunk)
            video_path = tfile.name
            st.success("Sample video loaded successfully!")
        except Exception as e:
            st.error(f"Failed to load sample video: {e}")
    else:
        uploaded_video = st.file_uploader(f"Upload a Video (Max {MAX_VIDEO_SIZE_MB}MB)", type=['mp4', 'mov', 'avi'])
        if uploaded_video is not None:
            file_size_mb = uploaded_video.size / (1024 * 1024)
            if file_size_mb > MAX_VIDEO_SIZE_MB:
                st.error(
                    f"❌ Video size exceeds limit! Uploaded: {file_size_mb:.2f}MB (Max allowed: {MAX_VIDEO_SIZE_MB}MB). Please use a smaller video or test with the sample video option.")
            else:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                video_path = tfile.name

    if video_path is not None:
        if st.button("🚀 Process Video"):
            cap = cv2.VideoCapture(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_file.name, fourcc, fps, (frame_width, frame_height))

            counted_ids = set()

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                metric_placeholder = st.empty()
            with col_m2:
                status_placeholder = st.empty()

            progress_bar = st.progress(0)
            frame_window = st.image([], channels="RGB", use_container_width=True)

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                resized = cv2.resize(frame, (640, 640))
                results = model.track(resized, persist=True, verbose=False)
                boxes = results[0].boxes

                if boxes is not None and boxes.id is not None:
                    for box, cls, track_id in zip(boxes.xyxy, boxes.cls, boxes.id):
                        cls = int(cls)
                        track_id = int(track_id)

                        if cls in VEHICLE_CLASSES:
                            counted_ids.add(track_id)
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(resized, f"ID {track_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(resized, f"Count: {len(counted_ids)}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                annotated_resized = cv2.resize(resized, (frame_width, frame_height))
                out.write(annotated_resized)

                frame_count += 1
                progress = min(frame_count / total_frames, 1.0) if total_frames > 0 else 0
                progress_bar.progress(progress)

                metric_placeholder.metric(label="Unique Vehicles Counted", value=len(counted_ids))
                status_placeholder.text(f"Processing Frame: {frame_count} / {total_frames}")

                if frame_count % 5 == 0:
                    frame_window.image(cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            out.release()
            progress_bar.empty()

            st.success(f"✅ Processing complete! Total unique vehicles detected: **{len(counted_ids)}**")

            with open(out_file.name, 'rb') as f:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=f,
                    file_name="vehicle_detection_output.mp4",
                    mime="video/mp4"
                )