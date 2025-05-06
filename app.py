import cv2
import mediapipe as mp
import numpy as np
import torch
from my_model import LSTMModel
import torch.nn as nn

# === Load model ===
classes = ['name', 'you', 'my', 'please', 'sit', 'meet', 'nice', 'yes', 'hello']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(258, 160, len(classes)).to(device)
model.load_state_dict(torch.load("F:/TTCS/best_model.pt", map_location=device))
model.eval()

# === Chuẩn hóa giống lúc train ===
def normalize_keypoint_block(block, dim=3):
    if np.all(block == 0):
        return block.flatten()
    block = block.reshape(-1, dim)
    mean = np.mean(block, axis=0)
    std = np.std(block, axis=0) + 1e-6
    normed = (block - mean) / std
    return normed.flatten()

def normalize_frames(frame):
    pose = frame[:132].reshape(33, 4)
    left = frame[132:195].reshape(21, 3)
    right = frame[195:258].reshape(21, 3)

    pose_norm = normalize_keypoint_block(pose, dim=4)
    left_norm = normalize_keypoint_block(left, dim=3)
    right_norm = normalize_keypoint_block(right, dim=3)

    return np.concatenate([pose_norm, left_norm, right_norm])

def normalize_keypoints(x):
    return np.array([normalize_frames(f) for f in x])

# === Hàm extract keypoint từ 1 frame ===
def extract_keypoints(results):
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints.extend([0]*33*4)

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([0]*21*3)

    return np.array(keypoints)

# === Predict từ video ===
def predict_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False)

    keypoints_buffer = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        keypoints = extract_keypoints(results)

        if not np.all(keypoints == 0):
            keypoints_buffer.append(keypoints)

        if len(keypoints_buffer) >= 80:
            break

    cap.release()

    if len(keypoints_buffer) < 80:
        last_frame = keypoints_buffer[-1]
        while len(keypoints_buffer) < 80:
            keypoints_buffer.append(last_frame)

    x = np.array(keypoints_buffer[:80])
    x = normalize_keypoints(x)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred = classes[np.argmax(probs)]
        conf = np.max(probs)

    print(f"Predicted: {pred} ({conf:.2f})")
    print(f"Probabilities:", dict(zip(classes, map(lambda x: round(x, 3), probs))))



predict_from_video("F:\TTCS\dataset\sit\Recording from 2025-05-02 18-02-26.023432.webm")
