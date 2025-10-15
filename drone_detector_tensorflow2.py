#!/usr/bin/env python3
import sys, os, subprocess, time
from collections import deque

# -------------------------------------------------
# Virtual environment auto-setup
# -------------------------------------------------
VENV_PATH = os.path.expanduser("~/ai-env")

def ensure_venv():
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"✅ Inside virtual environment: {sys.prefix}")
        return
    if not os.path.exists(VENV_PATH):
        print("📦 Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_PATH])
    python_bin = os.path.join(VENV_PATH, "bin", "python3")
    pip_bin = os.path.join(VENV_PATH, "bin", "pip")
    print("📦 Installing dependencies...")
    subprocess.check_call([pip_bin, "install", "--upgrade", "pip"])
    subprocess.check_call([pip_bin, "install", "opencv-python", "tensorflow", "pillow", "numpy"])
    print("🔄 Relaunching script inside virtual environment...")
    os.execv(python_bin, [python_bin] + sys.argv)

ensure_venv()

# -------------------------------------------------
# Imports after venv is ready
# -------------------------------------------------
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# -------------------------------------------------
# Config
# -------------------------------------------------
CAM1_INDEX = 0
CAM2_INDEX = 2
TARGET_WIDTH, TARGET_HEIGHT = 640, 640
DISPLAY_SIZE = 320
FPS = 20
FRAME_DELAY = 1.0 / FPS
BUFFER_SIZE = 2
THRESHOLD = 600
PATCH_SIZE = 100
COOLDOWN_FRAMES = 3  # Skip AI for N frames after non-drone

# -------------------------------------------------
# Load AI model
# -------------------------------------------------
print("🤖 Loading TensorFlow model...")
try:
    model = tf.keras.models.load_model("drone_classifier.h5")
    print("✅ TensorFlow model loaded successfully")
    
    try:
        import json
        with open('class_labels.json', 'r') as f:
            class_labels = json.load(f)
        label_names = {v: k for k, v in class_labels.items()}
        print(f"✅ Class labels loaded: {class_labels}")
    except FileNotFoundError:
        print("⚠️ No class_labels.json found, using default labels")
        label_names = {0: 'drone', 1: 'not_drone'}
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def extract_patch(frame, center_x, center_y, size=PATCH_SIZE):
    half = size // 2
    x1, x2 = center_x - half, center_x + half
    y1, y2 = center_y - half, center_y + half
    if x1 < 0:
        x2 -= x1; x1 = 0
    if y1 < 0:
        y2 -= y1; y1 = 0
    if x2 > frame.shape[1]:
        x1 -= (x2 - frame.shape[1]); x2 = frame.shape[1]
    if y2 > frame.shape[0]:
        y1 -= (y2 - frame.shape[0]); y2 = frame.shape[0]
    patch = frame[y1:y2, x1:x2]
    return cv2.resize(patch, (size, size))

def preprocess_patch_for_tf(patch):
    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(patch_rgb).resize((PATCH_SIZE, PATCH_SIZE))
    img_array = np.array(pil_image) / 255.0
    return np.expand_dims(img_array, axis=0)

def run_ai_on_patch(patch, camera_name=""):
    try:
        processed_patch = preprocess_patch_for_tf(patch)
        prediction = model.predict(processed_patch, verbose=0)[0][0]
        if prediction > 0.5:
            predicted_class = label_names.get(1, 'not_drone')
            confidence = prediction
        else:
            predicted_class = label_names.get(0, 'drone')
            confidence = 1 - prediction
        print(f"🤖 {camera_name} Prediction: {predicted_class} (confidence: {confidence:.3f})")
        return predicted_class == 'drone'
    except Exception as e:
        print(f"❌ AI prediction error on {camera_name}: {e}")
        return False

# -------------------------------------------------
# Main loop
# -------------------------------------------------
def main():
    cap1 = cv2.VideoCapture(CAM1_INDEX)
    cap2 = cv2.VideoCapture(CAM2_INDEX)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Could not open one or both cameras.")
        return

    print("✅ Cameras opened. Press 'q' to quit.")

    buffer1, buffer2 = deque(maxlen=BUFFER_SIZE), deque(maxlen=BUFFER_SIZE)
    cv2.namedWindow("Original + Frame Difference (2x2)", cv2.WINDOW_NORMAL)

    ai_cooldown_1, ai_cooldown_2 = 0, 0
    last_time = time.time()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("⚠️ Failed to grab frames.")
            break

        frame1 = cv2.resize(frame1, (TARGET_WIDTH, TARGET_HEIGHT))
        frame2 = cv2.resize(frame2, (TARGET_WIDTH, TARGET_HEIGHT))
        buffer1.append(frame1.copy())
        buffer2.append(frame2.copy())

        diff1 = cv2.absdiff(buffer1[-1], buffer1[-2]) if len(buffer1) == 2 else np.zeros_like(frame1)
        diff2 = cv2.absdiff(buffer2[-1], buffer2[-2]) if len(buffer2) == 2 else np.zeros_like(frame2)

        obj1 = np.sum(diff1, axis=2, dtype=np.uint16)
        obj2 = np.sum(diff2, axis=2, dtype=np.uint16)

        max_val1, max_val2 = int(np.max(obj1)), int(np.max(obj2))
        y1, x1 = np.unravel_index(np.argmax(obj1), obj1.shape)
        y2, x2 = np.unravel_index(np.argmax(obj2), obj2.shape)

        # Green boxes if threshold exceeded
        if (max_val1 + max_val2) > THRESHOLD:
            cv2.rectangle(frame1, (x1-50, y1-50), (x1+50, y1+50), (0,255,0), 2)
            cv2.rectangle(frame2, (x2-50, y2-50), (x2+50, y2+50), (0,255,0), 2)

            # Camera 1 AI with cooldown
            if ai_cooldown_1 == 0:
                patch1 = extract_patch(frame1, x1, y1)
                drone_detected_1 = run_ai_on_patch(patch1, "CAM1")
                if not drone_detected_1:
                    ai_cooldown_1 = COOLDOWN_FRAMES
            else:
                drone_detected_1 = False
                ai_cooldown_1 -= 1

            # Camera 2 AI with cooldown
            if ai_cooldown_2 == 0:
                patch2 = extract_patch(frame2, x2, y2)
                drone_detected_2 = run_ai_on_patch(patch2, "CAM2")
                if not drone_detected_2:
                    ai_cooldown_2 = COOLDOWN_FRAMES
            else:
                drone_detected_2 = False
                ai_cooldown_2 -= 1

            # Red boxes for confirmed drones
            if drone_detected_1:
                cv2.rectangle(frame1, (x1-50, y1-50), (x1+50, y1+50), (0,0,255), 4)
                cv2.putText(frame1, "DRONE!", (x1-40, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if drone_detected_2:
                cv2.rectangle(frame2, (x2-50, y2-50), (x2+50, y2+50), (0,0,255), 4)
                cv2.putText(frame2, "DRONE!", (x2-40, y2-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Display 2x2 grid
        cam1_combo = np.vstack((frame1, diff1))
        cam2_combo = np.vstack((frame2, diff2))
        combined = np.hstack((cam1_combo, cam2_combo))
        display_image = cv2.resize(combined, (DISPLAY_SIZE*2, DISPLAY_SIZE*2))
        cv2.imshow("Original + Frame Difference (2x2)", display_image)

        # Frame rate control
        elapsed = time.time() - last_time
        if elapsed < FRAME_DELAY:
            time.sleep(FRAME_DELAY - elapsed)
        last_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
