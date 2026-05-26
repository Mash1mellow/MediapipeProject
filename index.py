import math
import random
import time
import os
from dataclasses import dataclass
from urllib.request import urlopen

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


def download_pose_model(model_path: str = "pose_landmarker_lite.task") -> str:
    """Download MediaPipe pose model if not present."""
    if os.path.exists(model_path):
        return model_path
    
    print(f"Downloading MediaPipe pose model to {model_path}...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/pose_landmarker_lite.task"
    
    try:
        with urlopen(url) as response:
            with open(model_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise RuntimeError("Could not download MediaPipe pose model. Please check your internet connection.")

FEATURE_TEMPLATES = [
    {
        "name": "T-Pose",
        "targets": {
            "left_upper_arm_vertical": 90,
            "right_upper_arm_vertical": 90,
            "left_elbow": 180,
            "right_elbow": 180,
            "left_knee": 180,
            "right_knee": 180,
        },
    },
    {
        "name": "Hands Up",
        "targets": {
            "left_upper_arm_vertical": 0,
            "right_upper_arm_vertical": 0,
            "left_elbow": 180,
            "right_elbow": 180,
        },
    },
    {
        "name": "Squat",
        "targets": {
            "left_knee": 90,
            "right_knee": 90,
            "left_hip": 90,
            "right_hip": 90,
        },
    },
    {
        "name": "Left Stretch",
        "targets": {
            "left_upper_arm_vertical": 0,
            "right_upper_arm_vertical": 90,
            "left_elbow": 180,
            "right_elbow": 180,
        },
    },
    {
        "name": "Right Stretch",
        "targets": {
            "left_upper_arm_vertical": 90,
            "right_upper_arm_vertical": 0,
            "left_elbow": 180,
            "right_elbow": 180,
        },
    },
]

FEATURE_WEIGHTS = {
    "left_upper_arm_vertical": 1.0,
    "right_upper_arm_vertical": 1.0,
    "left_elbow": 0.8,
    "right_elbow": 0.8,
    "left_knee": 0.7,
    "right_knee": 0.7,
    "left_hip": 0.7,
    "right_hip": 0.7,
}


@dataclass
class PoseGameState:
    current_pose_index: int = 0
    best_score: float = 0.0
    round_start: float = 0.0
    round_duration: float = 10.0
    running: bool = False
    current_score: float = 0.0


def vector_angle(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    cosine = np.clip(np.dot(a_norm, b_norm), -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def get_landmark(landmarks, index: int) -> np.ndarray:
    """Get landmark coordinates by index."""
    if index >= len(landmarks):
        return np.array([0.0, 0.0], dtype=np.float32)
    landmark = landmarks[index]
    return np.array([landmark.x, landmark.y], dtype=np.float32)


def compute_pose_features(landmarks) -> dict[str, float]:
    """Compute pose features from MediaPipe landmarks (33-point model)."""
    left_shoulder = get_landmark(landmarks, 11)
    right_shoulder = get_landmark(landmarks, 12)
    left_elbow = get_landmark(landmarks, 13)
    right_elbow = get_landmark(landmarks, 14)
    left_wrist = get_landmark(landmarks, 15)
    right_wrist = get_landmark(landmarks, 16)
    left_hip = get_landmark(landmarks, 23)
    right_hip = get_landmark(landmarks, 24)
    left_knee = get_landmark(landmarks, 25)
    right_knee = get_landmark(landmarks, 26)
    left_ankle = get_landmark(landmarks, 27)
    right_ankle = get_landmark(landmarks, 28)

    left_upper_arm = left_elbow - left_shoulder
    right_upper_arm = right_elbow - right_shoulder
    left_forearm = left_wrist - left_elbow
    right_forearm = right_wrist - right_elbow
    left_thigh = left_knee - left_hip
    right_thigh = right_knee - right_hip
    left_shin = left_ankle - left_knee
    right_shin = right_ankle - right_knee

    vertical = np.array([0.0, -1.0], dtype=np.float32)

    features = {
        "left_upper_arm_vertical": vector_angle(left_upper_arm, vertical),
        "right_upper_arm_vertical": vector_angle(right_upper_arm, vertical),
        "left_elbow": vector_angle(left_shoulder - left_elbow, left_wrist - left_elbow),
        "right_elbow": vector_angle(right_shoulder - right_elbow, right_wrist - right_elbow),
        "left_hip": vector_angle(left_shoulder - left_hip, left_knee - left_hip),
        "right_hip": vector_angle(right_shoulder - right_hip, right_knee - right_hip),
        "left_knee": vector_angle(left_hip - left_knee, left_ankle - left_knee),
        "right_knee": vector_angle(right_hip - right_knee, right_ankle - right_knee),
        "torso_tilt": vector_angle(right_shoulder - left_shoulder, np.array([1.0, 0.0], dtype=np.float32)),
    }

    return features


def compare_pose_to_template(features: dict[str, float], template: dict[str, float]) -> float:
    total_weight = 0.0
    total_error = 0.0

    for feature_name, target_value in template.items():
        if feature_name not in features:
            continue

        weight = FEATURE_WEIGHTS.get(feature_name, 1.0)
        diff = abs(features[feature_name] - target_value) / 180.0
        total_error += diff * weight
        total_weight += weight

    if total_weight <= 0:
        return 0.0

    normalized = total_error / total_weight
    score = max(0.0, 100.0 * (1.0 - normalized))
    return score


def run_sample_similarity() -> dict[str, object]:
    sample_features = {
        "left_upper_arm_vertical": 90.0,
        "right_upper_arm_vertical": 95.0,
        "left_elbow": 170.0,
        "right_elbow": 175.0,
        "left_knee": 175.0,
        "right_knee": 178.0,
        "left_hip": 85.0,
        "right_hip": 88.0,
    }
    template = FEATURE_TEMPLATES[0]["targets"]
    score = compare_pose_to_template(sample_features, template)
    return {
        "template": FEATURE_TEMPLATES[0]["name"],
        "score": score,
        "features": sample_features,
    }


def draw_game_overlay(frame: np.ndarray, state: PoseGameState, target_name: str, score: float) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()

    instructions = [
        "Press 's' to start / restart",
        "Press 'n' for next pose",
        "Press 'q' or Esc to quit",
    ]

    for index, text in enumerate(instructions):
        cv2.putText(
            overlay,
            text,
            (16, 30 + index * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        f"Target: {target_name}",
        (16, height - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 180, 30),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay,
        f"Score: {score:.1f}%",
        (16, height - 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (60, 220, 60),
        2,
        cv2.LINE_AA,
    )

    time_left = max(0.0, state.round_duration - (time.time() - state.round_start))
    cv2.putText(
        overlay,
        f"Time: {time_left:.1f}s",
        (16, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    if state.running and time_left <= 0.0:
        cv2.putText(
            overlay,
            "Round ended! Press 'n' or 's' to continue.",
            (16, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )

    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(
        frame,
        f"Best: {state.best_score:.1f}%",
        (width - 260, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (180, 90, 250),
        2,
        cv2.LINE_AA,
    )


def draw_landmarks(frame, landmarks):
    """Draw pose landmarks on the frame."""
    if not landmarks:
        return
    
    h, w = frame.shape[:2]
    
    connections = [
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
        (11, 23), (12, 24),
        (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28),
        (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10),
        (0, 1), (1, 2), (2, 3), (3, 4),
    ]
    
    for conn in connections:
        p1 = landmarks[conn[0]]
        p2 = landmarks[conn[1]]
        if p1 and p2:
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
    
    for landmark in landmarks:
        if landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)


def main() -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Please connect a camera and try again.")

    state = PoseGameState(round_start=time.time())
    current_template = FEATURE_TEMPLATES[state.current_pose_index]

    model_path = download_pose_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            results = detector.detect(mp_image)

            score = state.current_score
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                draw_landmarks(frame, results.pose_landmarks[0])

                features = compute_pose_features(results.pose_landmarks[0])
                score = compare_pose_to_template(features, current_template["targets"])
                state.current_score = score

                if score > state.best_score:
                    state.best_score = score

            draw_game_overlay(frame, state, current_template["name"], score)
            cv2.imshow("MediaPipe Pose Similarity Game", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("n"):
                state.current_pose_index = (state.current_pose_index + 1) % len(FEATURE_TEMPLATES)
                current_template = FEATURE_TEMPLATES[state.current_pose_index]
                state.round_start = time.time()
                state.running = True
                state.current_score = 0.0
            if key == ord("s"):
                state.round_start = time.time()
                state.running = True
                state.current_score = 0.0

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
