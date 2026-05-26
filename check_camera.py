import cv2

print("Checking available cameras...")
print("-" * 50)

# Try different camera indices
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Camera found at index {i}")
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Resolution: {int(w)}x{int(h)}, FPS: {fps}")
        cap.release()
    else:
        if i == 0:
            print(f"✗ No camera at index {i} (default)")

print("-" * 50)
print("\nTrying different backends...")

backends = [
    ('CAP_DSHOW', cv2.CAP_DSHOW),
    ('CAP_MSMF', cv2.CAP_MSMF),
    ('CAP_V4L2', cv2.CAP_V4L2),
    ('Default', -1),
]

for name, backend_id in backends:
    if backend_id == -1:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0, backend_id)
    
    if cap.isOpened():
        print(f"✓ {name} backend works!")
        cap.release()
    else:
        print(f"✗ {name} backend failed")

print("\nIf no camera was found:")
print("1. Check if a physical webcam is connected to your USB port")
print("2. Make sure no other application is using the camera")
print("3. Check Device Manager > Imaging devices for camera")
