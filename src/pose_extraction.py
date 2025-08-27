# src/pose_extraction.py
import cv2
import mediapipe as mp
import numpy as np
import os

mp_pose = mp.solutions.pose

def extract_pose_sequence(video_path):
    """
    Extract pose keypoints from a video and return as np.array [T, 33, 3]
    """
    cap = cv2.VideoCapture(video_path)
    keypoints = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                kps = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            else:
                kps = np.zeros((33, 3))  # if pose not detected
            keypoints.append(kps)

    cap.release()

    if len(keypoints) == 0:
        raise ValueError(f"No poses detected in video: {video_path}")

    keypoints = np.stack(keypoints)  # shape [T, 33, 3]
    return keypoints

def save_keypoints(video_path, output_dir="data/processed"):
    """
    Extract pose keypoints from video and save as .npz
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    kps = extract_pose_sequence(video_path)
    video_name = os.path.basename(video_path).split('.')[0]
    shot_name = os.path.basename(os.path.dirname(video_path))
    npz_path = os.path.join(output_dir, f"{shot_name}_{video_name}.npz")
    np.savez(npz_path, keypoints=kps)
    print(f"Saved keypoints to {npz_path}")
    return npz_path

if __name__ == "__main__":
    # Example: process all videos in data/raw/*
    raw_dir = "data/raw"
    for shot in os.listdir(raw_dir):
        shot_path = os.path.join(raw_dir, shot)
        if os.path.isdir(shot_path):
            for video_file in os.listdir(shot_path):
                if video_file.endswith(".mp4"):
                    video_path = os.path.join(shot_path, video_file)
                    save_keypoints(video_path)
