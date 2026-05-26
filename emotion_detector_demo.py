import cv2
import numpy as np
from emotion_detector import simple_emotion_classifier


def create_demo_face():
    """Create a synthetic face image for demo purposes."""
    # Create a blank image (480x640x3)
    face_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw a simple face with eyes and mouth
    # Head
    cv2.circle(face_img, (320, 240), 100, (150, 100, 70), -1)
    
    # Eyes
    cv2.circle(face_img, (280, 200), 15, (50, 50, 50), -1)
    cv2.circle(face_img, (360, 200), 15, (50, 50, 50), -1)
    
    # Eyebrows
    cv2.line(face_img, (260, 180), (300, 170), (100, 70, 40), 3)
    cv2.line(face_img, (340, 170), (380, 180), (100, 70, 40), 3)
    
    return face_img


def demo_happy():
    """Demo: Happy face."""
    face = create_demo_face()
    
    # Big smile (wide curve)
    pts = np.array([[240, 280], [270, 310], [320, 320], [370, 310], [400, 280]], np.int32)
    cv2.polylines(face, [pts], False, (0, 0, 0), 4)
    
    # Brighter mouth area
    cv2.ellipse(face, (320, 300), (80, 30), 0, 0, 180, (200, 150, 100), -1)
    
    # Happy eyebrows (raised)
    cv2.line(face, (250, 160), (310, 150), (100, 70, 40), 4)
    cv2.line(face, (330, 150), (390, 160), (100, 70, 40), 4)
    
    return face


def demo_sad():
    """Demo: Sad face."""
    face = create_demo_face()
    
    # Downturned mouth (sad frown)
    pts = np.array([[240, 270], [270, 260], [320, 250], [370, 260], [400, 270]], np.int32)
    cv2.polylines(face, [pts], False, (0, 0, 0), 4)
    
    # Sad eyebrows (angled inward/down)
    cv2.line(face, (250, 175), (290, 190), (100, 70, 40), 4)
    cv2.line(face, (350, 190), (390, 175), (100, 70, 40), 4)
    
    # Darken eye region
    cv2.rectangle(face, (260, 180), (380, 220), (80, 60, 40), -1)
    
    # Redraw eyes
    cv2.circle(face, (280, 200), 12, (40, 40, 40), -1)
    cv2.circle(face, (360, 200), 12, (40, 40, 40), -1)
    
    return face


def demo_angry():
    """Demo: Angry face."""
    face = create_demo_face()
    
    # Angry straight line mouth (grimace)
    cv2.line(face, (250, 280), (390, 280), (0, 0, 0), 4)
    
    # Very angry eyebrows (harsh V-shape)
    cv2.line(face, (260, 160), (300, 180), (50, 30, 20), 5)
    cv2.line(face, (340, 180), (380, 160), (50, 30, 20), 5)
    
    # Darken upper face (eyes/brows region)
    cv2.rectangle(face, (250, 150), (390, 210), (60, 40, 30), -1)
    
    # Redraw eyes darker
    cv2.circle(face, (280, 200), 12, (30, 30, 30), -1)
    cv2.circle(face, (360, 200), 12, (30, 30, 30), -1)
    
    return face


def run_demo():
    """Run emotion detector in demo mode without a webcam."""
    print("Running Emotion Detector in DEMO MODE (no webcam)")
    print("Press keys to test different emotions:")
    print("  1 - Happy")
    print("  2 - Sad")
    print("  3 - Angry")
    print("  q - Quit")
    print("-" * 50)
    
    while True:
        # Create a demo face based on user input
        key = cv2.waitKey(500) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('1'):
            demo_frame = demo_happy()
        elif key == ord('2'):
            demo_frame = demo_sad()
        elif key == ord('3'):
            demo_frame = demo_angry()
        else:
            # Default: show instruction screen
            demo_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
            cv2.putText(demo_frame, "Emotion Detector - DEMO MODE", (80, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(demo_frame, "Press: 1=Happy, 2=Sad, 3=Angry, q=Quit", (60, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(demo_frame, "No webcam detected", (150, 350),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Emotion Detector - DEMO", demo_frame)
            continue
        
        # Analyze the demo face
        emotion, confidence = simple_emotion_classifier(demo_frame)
        
        # Draw result on frame
        color = (0, 255, 0) if emotion == "happy" else (0, 165, 255) if emotion == "angry" else (255, 0, 0)
        label = f"Detected: {emotion.upper()} ({confidence:.2f})"
        cv2.putText(demo_frame, label, (50, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Face Emotion Detector - DEMO", demo_frame)
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()
