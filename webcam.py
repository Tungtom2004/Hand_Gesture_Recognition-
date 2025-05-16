import cv2
import mediapipe as mp
import torch
import numpy as np
from collections import deque, Counter
from my_model import LSTMModel
import torch.nn as nn

# === Load model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(258, 160, 9).to(device)
model.load_state_dict(torch.load('F:/TTCS/best_model.pt', map_location=device))
model.eval()


classes = ['hello', 'meet', 'my', 'name', 'nice', 'please', 'sit', 'yes', 'you']

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

# === Webcam setup ===

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,   
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
buffer = deque(maxlen=80)
pred_history = deque(maxlen=5)
label = ""
last_pred_time = 0
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    keypoints = extract_keypoints(results)

    if keypoints is not None and np.std(keypoints) > 1e-3:
        buffer.append(keypoints)

    if len(buffer) == 80 and frame_count % 5 == 0:
        x = np.array(buffer)
        x = normalize_keypoints(x)
        x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x_tensor)
            pred = torch.argmax(output, dim=1).item()
            conf = torch.softmax(output, dim=1)[0][pred].item()

        if conf > 0.7:
            pred_history.append(pred)
            most_common, count = Counter(pred_history).most_common(1)[0]
            if count >= 3:
                label = f"{classes[most_common]} ({conf:.2f})"
        else:
            label = ""

    if label:
        cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

