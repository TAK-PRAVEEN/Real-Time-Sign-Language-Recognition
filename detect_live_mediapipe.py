import cv2
import json
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp

# ------------------- Config -------------------
MODEL_PATH = "asl_mobilenetv2.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = (224, 224)           # must match training
CONF_THRESH = 0.60              # only accept predictions above this
SMOOTH_WINDOW = 15              # majority vote window
MARGIN = 40                     # bbox padding (pixels)

# ------------------- Load model/labels -------------------
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)
labels = {int(k): v for k, v in labels.items()}  # {idx: "A", ...}

# ------------------- MediaPipe Hands -------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------- Helpers -------------------
def square_crop_coords(xmin, ymin, xmax, ymax, w, h, margin=0):
    xmin = max(0, xmin - margin)
    ymin = max(0, ymin - margin)
    xmax = min(w, xmax + margin)
    ymax = min(h, ymax + margin)

    bw = xmax - xmin
    bh = ymax - ymin
    side = max(bw, bh)

    # center square on current bbox
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    half = side // 2

    sx = max(0, cx - half)
    sy = max(0, cy - half)
    ex = min(w, sx + side)
    ey = min(h, sy + side)

    # adjust if near border
    sx = max(0, ex - side)
    sy = max(0, ey - side)
    return int(sx), int(sy), int(ex), int(ey)

pred_queue = deque(maxlen=SMOOTH_WINDOW)

# ------------------- Camera -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open camera")
    raise SystemExit

print("✅ Live detection started. Press 'q' to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    disp_label = ""
    conf_txt = ""

    if res.multi_hand_landmarks:
        # draw landmarks and make bbox from landmarks
        for hand_landmarks in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
            ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)

            # square crop with margin
            sx, sy, ex, ey = square_crop_coords(xmin, ymin, xmax, ymax, w, h, MARGIN)
            cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)

            roi = frame[sy:ey, sx:ex]
            if roi.size > 0:
                roi_resized = cv2.resize(roi, IMG_SIZE)
                x = roi_resized.astype("float32") / 255.0
                x = np.expand_dims(x, axis=0)

                probs = model.predict(x, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                conf = float(np.max(probs))

                if conf >= CONF_THRESH:
                    pred_queue.append(pred_idx)

                    # majority vote smoothing
                    final_idx = max(set(pred_queue), key=pred_queue.count)
                    disp_label = labels[final_idx]
                    conf_txt = f"{conf*100:.1f}%"
                else:
                    disp_label = "Detecting..."
                    conf_txt = ""


            break  # only first (strongest) hand

    # HUD
    cv2.putText(frame, f"Pred: {disp_label} {conf_txt}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("ASL (MediaPipe + MobileNetV2)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
