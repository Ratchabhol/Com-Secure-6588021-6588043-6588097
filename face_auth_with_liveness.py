import cv2
import mediapipe as mp
import time
from scipy.spatial import distance as dist

# --- ค่าคงที่สำหรับ EAR ---
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 2
REQUIRED_BLINKS = 2
BLINK_TIMEOUT = 5

# --- ตัวแปรตรวจจับการกะพริบ ---
blink_counter = 0
total_blinks = 0
liveness_confirmed = False
start_time = time.time()

# --- ตำแหน่ง landmark ตาซ้าย/ขวา (FaceMesh) ---
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# --- EAR calculation function ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- โหลด Face Recognizer ---
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trainer.yml")

# --- โหลดชื่อ (หากมี user_id -> ชื่อ)
labels = {1: "User 1", 2: "User 2"}  # ตัวอย่าง

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# --- เปิดกล้อง ---
cap = cv2.VideoCapture(1)  # ถ้าใช้มือถือเป็นเว็บแคมแล้วขึ้นภาพ ให้ใช้ index อื่นเช่น 1
time.sleep(1.0)

print(f"[INFO] Please blink {REQUIRED_BLINKS} times within {BLINK_TIMEOUT} seconds.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]

        left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
        right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # --- ตรวจจับการกะพริบ ---
        if not liveness_confirmed:
            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    total_blinks += 1
                    print(f"Blink {total_blinks}")
                blink_counter = 0

            if total_blinks >= REQUIRED_BLINKS:
                liveness_confirmed = True
                print("[INFO] Liveness Confirmed ✅")
            elif time.time() - start_time > BLINK_TIMEOUT:
                cv2.putText(frame, "❌ Liveness Check Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- เมื่อยืนยันว่าเป็นคนจริงแล้ว ---
        if liveness_confirmed:
            x_coords = [p[0] for p in landmarks]
            y_coords = [p[1] for p in landmarks]
            x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

            face_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            try:
                resized_face = cv2.resize(face_gray, (100, 100))
                label, confidence = recognizer.predict(resized_face)

                if confidence < 100:  # กำหนด threshold
                    name = labels.get(label, f"User {label}")
                    text = f"{name} ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    text = "Unknown"
                    color = (0, 0, 255)

                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            except:
                cv2.putText(frame, "Face too small or error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Debug แสดง EAR & Blink
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Face Auth with Blink Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        blink_counter = 0
        total_blinks = 0
        liveness_confirmed = False
        start_time = time.time()
        print("[INFO] Resetting Liveness Check...")

cap.release()
cv2.destroyAllWindows()