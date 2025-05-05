import os
import cv2
import face_recognition
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import winsound
from tkinter import *
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# === Paths ===
encodings_dir = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/encodings"
face_output_dir = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Project Code/src/recognized_faces"
os.makedirs(face_output_dir, exist_ok=True)

# === Load known encodings ===
known_encodings = []
known_names = []
for file in os.listdir(encodings_dir):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(os.path.join(encodings_dir, file), "rb") as f:
            encodings = pickle.load(f)
            known_encodings.extend(encodings)
            known_names.extend([name] * len(encodings))

# === Data structures
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])
attendance_set = set()  # to avoid duplicates

# === Initialize GUI
window = Tk()
window.title("FRAS - Facial Recognition Attendance System")
window.geometry("950x700")

video_label = Label(window)
video_label.pack()

cap = None

# === Treeview Attendance Table ===
tree_frame = Frame(window)
tree_frame.pack(pady=10)

attendance_table = ttk.Treeview(tree_frame, columns=("Name", "Timestamp"), show="headings", height=10)
attendance_table.heading("Name", text="Name")
attendance_table.heading("Timestamp", text="Timestamp")
attendance_table.pack()


def update_attendance_table(name, timestamp):
    attendance_table.insert('', 'end', values=(name, timestamp))


def start_camera():
    global cap
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    update_frame()


def update_frame():
    global cap
    if not cap or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        if len(face_distances) > 0:
            best_match_idx = np.argmin(face_distances)
            if matches[best_match_idx]:
                name = known_names[best_match_idx]

                if name not in attendance_set:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_df.loc[len(attendance_df)] = [name, timestamp]
                    attendance_set.add(name)

                    # 📸 Save face image
                    cropped = frame[top:bottom, left:right]
                    filename = f"{name}_{timestamp.replace(':', '-')}.jpg"
                    cv2.imwrite(os.path.join(face_output_dir, filename), cropped)

                    # 🔊 Sound
                    winsound.Beep(1000, 150)

                    # 🧾 Update GUI table
                    update_attendance_table(name, timestamp)
                    print(f"✅ {name} logged at {timestamp}")

        # Draw label on screen
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Convert to Tkinter format
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    window.after(10, update_frame)


def stop_camera():
    global cap
    if cap:
        cap.release()
    video_label.config(image='')


def save_log():
    attendance_df.to_csv("attendance_log.csv", index=False)
    messagebox.showinfo("Saved", "Attendance log saved to attendance_log.csv")


# === Buttons
btn_frame = Frame(window)
btn_frame.pack(pady=5)

Button(btn_frame, text="▶ Start Camera", command=start_camera, width=15).grid(row=0, column=0, padx=10)
Button(btn_frame, text="⏹ Stop Camera", command=stop_camera, width=15).grid(row=0, column=1, padx=10)
Button(btn_frame, text="💾 Save Log", command=save_log, width=15).grid(row=0, column=2, padx=10)

window.mainloop()
