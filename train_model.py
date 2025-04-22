import cv2
import numpy as np
import os
import random # เพิ่มเข้ามาเพื่อใช้สุ่มค่า augmentation

# --- ฟังก์ชันสำหรับ Data Augmentation ---
def augment_image(image):
    """
    ทำการสุ่ม Augmentation ให้กับภาพ
    """
    augmented_images = []
    rows, cols = image.shape

    # 1. เพิ่มภาพต้นฉบับ (หลัง resize) เข้าไปก่อน
    resized_original = cv2.resize(image, (100, 100))
    augmented_images.append(resized_original)

    # สร้างภาพ Augmented เพิ่มเติม (เช่น 4 ภาพ)
    for _ in range(4): # สร้าง 4 variations เพิ่มเติม
        img_copy = image.copy() # ทำงานกับสำเนา

        # --- สุ่มการปรับแต่ง ---

        # 2. Random Rotation (-10 ถึง +10 องศา)
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_copy = cv2.warpAffine(img_copy, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)

        # 3. Random Brightness Adjustment (-40 ถึง +40)
        brightness_val = random.randint(-40, 40)
        # ใช้ np.clip เพื่อจำกัดค่า intensity ให้อยู่ในช่วง 0-255
        img_copy = np.clip(img_copy.astype(int) + brightness_val, 0, 255).astype(np.uint8)

        # 4. Random Horizontal Flip (โอกาส 50%)
        if random.random() > 0.5:
            img_copy = cv2.flip(img_copy, 1) # 1 คือ flip แนวนอน

        # --- สิ้นสุดการปรับแต่ง ---

        # ปรับขนาดภาพ augmented ให้เป็น 100x100
        img_copy_resized = cv2.resize(img_copy, (100, 100))
        augmented_images.append(img_copy_resized)

    return augmented_images
# --- สิ้นสุดฟังก์ชัน Augmentation ---


# เตรียมข้อมูลจากโฟลเดอร์ dataset
data_dir = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()  # สร้างโมเดลจดจำใบหน้า

faces = []  # เก็บภาพใบหน้า (รวม original และ augmented)
ids = []    # เก็บ User ID

# อ่านข้อมูลจาก dataset
print("Loading images and applying augmentation...")
for user_id in os.listdir(data_dir):
    user_path = os.path.join(data_dir, user_id)
    if not os.path.isdir(user_path): # ข้ามไฟล์ที่ไม่ใช่โฟลเดอร์
        continue

    print(f"Processing User ID: {user_id}")
    try:
        numeric_user_id = int(user_id) # แปลง ID เป็นตัวเลข
    except ValueError:
        print(f"Skipping non-numeric user ID: {user_id}")
        continue # ข้ามถ้า user_id ไม่ใช่ตัวเลข

    for image_file in os.listdir(user_path):
        img_path = os.path.join(user_path, image_file)
        print(f"  Loading image: {image_file}")

        # โหลดภาพในรูปแบบขาวดำ (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"  Warning: Could not load image {img_path}. Skipping.")
            continue

        # --- ทำ Augmentation ---
        augmented = augment_image(img)

        # เพิ่มภาพทั้งหมด (original + augmented) และ User ID ลงในลิสต์
        for augmented_img in augmented:
            faces.append(augmented_img)
            ids.append(numeric_user_id) # ใช้ numeric_user_id ที่แปลงแล้ว

# ตรวจสอบว่ามีข้อมูลสำหรับเทรนหรือไม่
if not faces or not ids:
    print("Error: No faces found to train the model. Please capture faces first.")
    exit() # ออกจากโปรแกรมถ้าไม่มีข้อมูล

# แปลงลิสต์ให้เป็น NumPy Array
faces = np.array(faces)
ids = np.array(ids)

# เทรนโมเดลจดจำใบหน้า
print("\nTraining the model with augmented data, please wait...")
recognizer.train(faces, ids)

# บันทึกโมเดลในไฟล์ face_trainer.yml
recognizer.save("face_trainer.yml") #
print(f"Model trained and saved as 'face_trainer.yml'") #
print(f"Total training images (including augmented): {len(faces)}")