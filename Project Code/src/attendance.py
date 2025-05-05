import os
import cv2
import face_recognition
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

encodings_dir = "C:/Users/26506/Desktop/Senior 2/Project/project file/obu-project/obu-project/Dataset/encodings"

# Initialize lists for names and encodings
known_names = []
known_encodings = []

# Load all encoding files from the encodings folder
for file in os.listdir(encodings_dir):
    if file.endswith(".pkl"):
        person_name = file.split(".pkl")[0]
        with open(os.path.join(encodings_dir, file), "rb") as f:
            person_encodings = pickle.load(f)
            for encoding in person_encodings:
                known_names.append(person_name)
                known_encodings.append(encoding)

# Initialize attendance DataFrame
attendance_df = pd.DataFrame(columns=["Name", "Timestamp"])

# Initialize webcam
video_capture = cv2.VideoCapture(0)
print("Starting real-time face recognition. Press 'q' to exit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and compute encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known encodings
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        # Find best match based on face distance
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            # Log attendance if not already logged
            if name not in attendance_df["Name"].values:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance_df = attendance_df._append({"Name": name, "Timestamp": timestamp}, ignore_index=True)
                print(f"Attendance logged for {name} at {timestamp}")

        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save attendance log to CSV
attendance_df.to_csv("attendance_log.csv", index=False)
print("Attendance log saved to 'attendance_log.csv'.")

video_capture.release()
cv2.destroyAllWindows()
