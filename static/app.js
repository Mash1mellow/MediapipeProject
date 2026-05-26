const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const startButton = document.getElementById('start-button');
const statusText = document.getElementById('emotion-label');

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights';

function setStatus(text) {
    statusText.textContent = text;
}

async function loadFaceApiModels() {
    setStatus('Loading face detection models...');
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
    await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
    setStatus('Models loaded. Press Start Camera.');
}

function chooseEmotion(expressions) {
    const entries = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
    const [name, score] = entries[0] || ['neutral', 0];
    const confidence = Math.round(score * 100);
    let label = 'Neutral';
    let emoji = '😐';

    if (name === 'happy' && score >= 0.25) {
        label = 'Happy';
        emoji = '😊';
    } else if (name === 'sad' && score >= 0.15) {
        label = 'Sad';
        emoji = '😢';
    } else if (name === 'angry' && score >= 0.20) {
        label = 'Angry';
        emoji = '😠';
    } else if (name === 'surprised' && score >= 0.20) {
        label = 'Surprised';
        emoji = '😲';
    }

    return `${label} ${emoji} (${confidence}%)`;
}

video.addEventListener('loadedmetadata', () => {
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
});

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        video.srcObject = stream;
        await video.play();
        setStatus('Camera started. Looking for face...');
        requestAnimationFrame(detectExpressions);
    } catch (error) {
        console.error(error);
        setStatus('Unable to open camera. Allow camera access and try again.');
    }
}

async function detectExpressions() {
    if (video.paused || video.ended) {
        requestAnimationFrame(detectExpressions);
        return;
    }

    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(overlay, displaySize);

    const result = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
        .withFaceExpressions();

    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    if (result) {
        const resized = faceapi.resizeResults(result, displaySize);
        faceapi.draw.drawDetections(overlay, resized);
        setStatus(chooseEmotion(result.expressions));
    } else {
        setStatus('No face found yet. Please move your face into the camera frame.');
    }

    requestAnimationFrame(detectExpressions);
}

startButton.addEventListener('click', async () => {
    startButton.disabled = true;
    setStatus('Starting camera...');
    await startCamera();
});

window.addEventListener('load', async () => {
    await loadFaceApiModels();
});
