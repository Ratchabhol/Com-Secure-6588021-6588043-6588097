import cv2
import os
import mediapipe as mp
import time  # เพิ่มเพื่อใช้หน่วงเวลา

# ตั้งชื่อ User ID สำหรับผู้ใช้งานใหม่
user_id = input("Enter User ID: ")  # เช่น "123"
output_dir = f"dataset/{user_id}"  # เก็บรูปในโฟลเดอร์ dataset/123

# สร้างโฟลเดอร์สำหรับเก็บข้อมูล (ถ้ายังไม่มี)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# เปิดกล้อง
cap = cv2.VideoCapture(1)  # ใช้กล้องที่ 1 (0 สำหรับกล้องหลัก)

# ใช้ MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

print("Preparing to capture face images...")
print("Please follow the instructions on the screen.")

count = 0
max_images = 120  # จำนวนรูปภาพที่ต้องการเก็บ (40 หน้าตรง, 40 หันซ้าย, 40 หันขวา)
phase = "front"  # เริ่มจากหน้าตรง
phase_start_time = time.time()  # บันทึกเวลาเริ่มต้นของ phase
delay_time = 5  # เพิ่มเวลารอ 5 วินาทีในแต่ละ phase

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    display_frame = frame.copy()  # สร้างสำเนาเพื่อวาดข้อความ

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]

        # คำนวณกรอบใบหน้าที่ครอบคลุมกรามและหู
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]
        x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

        # วาดกรอบรอบใบหน้า
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # บันทึกภาพใบหน้า
        if count < max_images:
            face = frame[y1:y2, x1:x2]
            file_path = os.path.join(output_dir, f"{count + 1}.jpg")
            cv2.imwrite(file_path, face)
            count += 1

        # แสดงหมายเลขภาพที่บันทึก
        cv2.putText(display_frame, f"Image: {count}/{max_images}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # แสดงข้อความแนะนำตาม phase
    if phase == "front":
        cv2.putText(display_frame, "Look FRONT (1/3)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if count >= max_images // 3:
            phase = "left"
            phase_start_time = time.time()  # รีเซ็ตเวลาเริ่มต้น
    elif phase == "left":
        if time.time() - phase_start_time < delay_time:  # เพิ่มเวลารอ 5 วินาที
            cv2.putText(display_frame, "Prepare to turn LEFT...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Turn LEFT (2/3)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if count >= (max_images // 3) * 2:
                phase = "right"
                phase_start_time = time.time()  # รีเซ็ตเวลาเริ่มต้น
    elif phase == "right":
        if time.time() - phase_start_time < delay_time:  # เพิ่มเวลารอ 5 วินาที
            cv2.putText(display_frame, "Prepare to turn RIGHT...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Turn RIGHT (3/3)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # แสดงภาพพร้อมกรอบและข้อความแนะนำ
    cv2.imshow("Capturing Faces", display_frame)

    # หยุดเมื่อกด 'q' หรือเก็บครบตามจำนวน
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images in {output_dir}")