import os
import time
import numpy as np
import face_recognition
import pandas as pd
import pickle
import cv2
from datetime import datetime

# === Paths ===
encodings_dir = "/home/ztd/fras/encodings"
face_output_dir = "/home/ztd/fras/recognized_faces"
os.makedirs(face_output_dir, exist_ok=True)

# === Load known encodings
known_encodings = []
known_names = []
for file in os.listdir(encodings_dir):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(os.path.join(encodings_dir, file), "rb") as f:
            encodings = pickle.load(f)
            known_encodings.extend(encodings)
            known_names.extend([name] * len(encodings))

# === Attendance
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])
attendance_set = set()


# === Beep sound
def beep():
    os.system('play -nq -t alsa synth 0.15 sine 1000')


# === Save attendance log
def save_log():
    attendance_df.to_csv("attendance_log.csv", index=False)
    print("💾 Saved attendance_log.csv")


# === Initialize camera (GStreamer pipeline for Jetson)
gst_str = ("nvarguscamerasrc ! "
           "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
           "nvvidconv flip-method=0 ! "
           "video/x-raw, format=BGRx ! "
           "videoconvert ! "
           "video/x-raw, format=BGR ! appsink")

cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print(" Failed to open camera")
    exit()

# === Recognition interval
RECOGNITION_INTERVAL = 1.0  # 每1秒识别一次
last_recognition_time = 0

print("📸 Starting GPU-accelerated attendance system... Press 'q' to exit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame")
            continue

        current_time = time.time()

        # === 人脸识别（每隔一段时间识别一次）
        if current_time - last_recognition_time > RECOGNITION_INTERVAL:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB

            # 使用GPU版dlib加速人脸检测（CNN模型）
            face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                name = "Unknown"

                if len(face_distances) > 0:
                    best_match_idx = np.argmin(face_distances)
                    if matches[best_match_idx]:
                        name = known_names[best_match_idx]

                        if name not in attendance_set:
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            attendance_df.loc[len(attendance_df)] = [name, timestamp]
                            attendance_set.add(name)
                            beep()
                            print(f" {name} logged at {timestamp}")

                            # 保存签到成功的单独人脸图片
                            face_img = frame[top * 2:bottom * 2, left * 2:right * 2]
                            if face_img.size != 0:
                                save_path = os.path.join(face_output_dir, f"{name}_{timestamp}.jpg")
                                cv2.imwrite(save_path, face_img)

                # === 绘制人脸框（坐标要放大回原尺寸）
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            last_recognition_time = current_time

        # === 显示签到人数
        cv2.putText(frame, f"Signed in: {len(attendance_set)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # === 展示画面
        cv2.imshow('Face Attendance System (GPU Accelerated)', frame)

        # === 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n KeyboardInterrupt detected...")

finally:
    print("Saving attendance log...")
    save_log()
    cap.release()
    cv2.destroyAllWindows()
