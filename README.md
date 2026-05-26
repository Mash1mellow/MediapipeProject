# Mobile Face Emotion Reader

A lightweight Flask demo that uses the browser camera to detect happy / sad face expressions in real time.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Test on Phone

1. Make sure your phone and computer are on the same Wi-Fi network.
2. Open your phone browser.
3. Visit `http://<computer-ip>:5000`.
4. Allow camera access.
5. Point the camera at your face and watch the live emotion result.

## Notes

- The app uses browser-side face detection with `face-api.js`.
- No extra Python computer vision libraries are required.
- For mobile testing, use a modern browser and allow camera permission.
