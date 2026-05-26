import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
from urllib.request import urlopen


def download_face_model(model_path: str = "face_detector.task") -> str:
    """Download MediaPipe face detector model if not present."""
    if os.path.exists(model_path):
        return model_path
    
    print(f"Downloading MediaPipe face detector model to {model_path}...")
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/face_detector.task"
    
    try:
        with urlopen(url) as response:
            with open(model_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"Model downloaded successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise RuntimeError("Could not download face detector model. Check internet connection.")


def detect_faces(detector, frame):
    """Detect faces in the frame using MediaPipe."""
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    results = detector.detect(mp_image)
    return results.detections if results.detections else []


def simple_emotion_classifier(face_roi):
    """Improved emotion classifier based on face region features."""
    if face_roi is None or face_roi.size == 0:
        return "neutral", 0.0
    
    h, w = face_roi.shape[:2]
    
    # Convert to grayscale and HSV
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    
    # Divide face into regions
    top_quarter = gray[:h//4, :]
    middle_half = gray[h//4:3*h//4, :]
    bottom_quarter = gray[3*h//4:, :]
    
    # Calculate statistics for each region
    top_mean = np.mean(top_quarter)
    middle_mean = np.mean(middle_half)
    bottom_mean = np.mean(bottom_quarter)
    
    top_std = np.std(top_quarter)
    middle_std = np.std(middle_half)
    bottom_std = np.std(bottom_quarter)
    
    # Detect edges more carefully
    edges = cv2.Canny(gray, 50, 150)
    
    # Count edges in different regions
    top_edges = np.sum(edges[:h//3, :])
    middle_edges = np.sum(edges[h//3:2*h//3, :])
    bottom_edges = np.sum(edges[2*h//3:, :])
    
    # Total edge count
    total_edges = np.sum(edges)
    
    # Detect horizontal lines (mouth indicators)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    h_lines_bottom = np.sum(horizontal_lines[2*h//3:, :])
    
    # Detect vertical/diagonal lines (eyebrow/frown indicators)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
    v_lines_top = np.sum(vertical_lines[:h//3, :])
    
    scores = {
        "happy": 0.0,
        "sad": 0.0,
        "angry": 0.0,
        "neutral": 0.0
    }
    
    # HAPPY: Bright bottom (smile), less variation in face
    if bottom_edges > 300 and h_lines_bottom > 100:
        scores["happy"] += 50
    if middle_std < 40:  # Smooth face = happy
        scores["happy"] += 20
    if bottom_mean > middle_mean + 10:  # Brighter mouth area
        scores["happy"] += 15
    
    # SAD: Lower eye region darker, less bottom edge activity
    if bottom_edges < 200 and total_edges < 2000:
        scores["sad"] += 40
    if top_mean < 100:  # Dark eyes
        scores["sad"] += 30
    if bottom_mean < middle_mean:  # Darker lower face
        scores["sad"] += 15
    
    # ANGRY: Heavy edge activity in middle/top (frown, furrowed brows)
    if middle_edges > 800 or v_lines_top > 150:
        scores["angry"] += 50
    if top_std > 50:  # Varied/harsh features
        scores["angry"] += 20
    if top_mean < 90:  # Very dark eyes
        scores["angry"] += 15
    
    # Get max emotion
    max_emotion = max(scores, key=scores.get)
    max_score = scores[max_emotion]
    
    # Normalize confidence
    if max_score == 0:
        return "neutral", 0.5
    
    confidence = min(0.99, max_score / 100.0)
    return max_emotion, confidence


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    
    # Initialize face detector
    model_path = download_face_model()
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    
    print("Face Emotion Detector started. Press 'q' to exit.")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # Detect faces
        detections = detect_faces(detector, frame)
        
        if detections:
            for detection in detections:
                # Get bounding box
                bbox = detection.bounding_box
                start_x = int(bbox.origin_x * w)
                start_y = int(bbox.origin_y * h)
                end_x = int((bbox.origin_x + bbox.width) * w)
                end_y = int((bbox.origin_y + bbox.height) * h)
                
                # Ensure bbox is within frame
                start_x = max(0, start_x)
                start_y = max(0, start_y)
                end_x = min(w, end_x)
                end_y = min(h, end_y)
                
                # Extract face region
                face_roi = frame[start_y:end_y, start_x:end_x]
                
                # Classify emotion
                emotion, confidence = simple_emotion_classifier(face_roi)
                
                # Draw bounding box
                color = (0, 255, 0) if emotion == "happy" else (0, 165, 255) if emotion == "angry" else (255, 0, 0)
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
                
                # Put text
                label = f"{emotion.upper()} ({confidence:.2f})"
                cv2.putText(frame, label, (start_x, start_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(frame, "No face detected", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Face Emotion Detector", frame)
        
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
