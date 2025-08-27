import streamlit as st
from pathlib import Path
import numpy as np
from src.pose_extraction import extract_pose_sequence
from src.analysis import compare_shots

st.set_page_config(page_title="🏏 Cricket Shot Analyzer", layout="wide")

# --- Custom CSS for Futuristic Look ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .video-box {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .accuracy-circle {
        width: 160px;
        height: 160px;
        border-radius: 50%;
        background: conic-gradient(#4ade80 calc(var(--val) * 1%), #1e293b 0);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        margin: auto;
        box-shadow: 0px 4px 25px rgba(0,0,0,0.5);
    }
    .feedback-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 15px 20px;
        border-radius: 12px;
        margin: 10px 0;
        font-size: 16px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {
        color: #38bdf8;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏏 Cricket Shot Analyzer")
st.markdown("### Compare your shot with the pros and get **personalized feedback** 🚀")

# --- Shot selection ---
shots = ["coverdrive", "pull", "cut", "hook"]
selected_shot = st.selectbox("🎯 Select intended shot", shots)

# --- Reference video ---
ref_video_path = Path(f"data/perfect_shots/{selected_shot}.mp4")

# --- User upload ---
uploaded_file = st.file_uploader("📤 Upload your shot", type=["mp4", "mov", "avi"])

if uploaded_file:
    user_video_path = Path("data/user_uploaded.mp4")
    with open(user_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- Display videos side by side ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎥 Reference Shot")
        st.markdown('<div class="video-box">', unsafe_allow_html=True)
        st.video(str(ref_video_path))
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📹 Your Shot")
        st.markdown('<div class="video-box">', unsafe_allow_html=True)
        st.video(str(user_video_path))
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Extract keypoints ---
    ref_kps = extract_pose_sequence(str(ref_video_path))
    user_kps = extract_pose_sequence(str(user_video_path))
    np.savez("data/processed/ref_sample.npz", keypoints=ref_kps)
    np.savez("data/processed/user_sample.npz", keypoints=user_kps)

    # --- Compare shots ---
    similarity, feedback = compare_shots(ref_kps, user_kps)

    # --- Accuracy Score ---
    st.markdown("### ⚡ Accuracy Analysis")
    st.markdown(f"""
        <div class="accuracy-circle" style="--val:{similarity/100}">
            {similarity}%
        </div>
    """, unsafe_allow_html=True)

    # --- Feedback Section ---
    st.markdown("### 📝 Personalized Feedback")
    for f in feedback:
        st.markdown(f'<div class="feedback-card">✔ {f}</div>', unsafe_allow_html=True)
