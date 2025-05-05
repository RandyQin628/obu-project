import os
import pickle
import numpy as np
import face_recognition
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === Configuration ===
encodings_dir = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/encodings"   # .pkl files for known people
test_dir = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/splitted/test"  # structured as test/PersonName/*.jpg
threshold = 0.6  # distance threshold for match

# === Load known embeddings ===
known_encodings = []
known_labels = []

# Load all .pkl files containing embeddings for the 5 classes
# Limiting the loading to the top 5 known classes, you can choose specific ones.
class_filter = ['Ben Stiller', 'Daniel Radcliffe', 'Kate Winslet', 'Tom Cruise', 'George W Bush']  # Example 5 classes
for file in os.listdir(encodings_dir):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        if name in class_filter:  # Only load embeddings for the 5 chosen classes
            with open(os.path.join(encodings_dir, file), "rb") as f:
                encodings = pickle.load(f)
                known_encodings.extend(encodings)
                known_labels.extend([name] * len(encodings))

print("✅ Loaded known identities:", set(known_labels))

# === Test evaluation ===
y_true = []
y_pred = []

# Loop through test images and only consider the 5-class test
for person_name in os.listdir(test_dir):
    if person_name not in class_filter:  # Skip non-relevant classes
        continue
    person_folder = os.path.join(test_dir, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_file in os.listdir(person_folder):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(person_folder, image_file)
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            print(f"⚠️ No face found in: {image_file}")
            continue

        test_encoding = face_encodings[0]
        distances = face_recognition.face_distance(known_encodings, test_encoding)
        best_index = np.argmin(distances)
        best_distance = distances[best_index]

        if best_distance < threshold:
            predicted_name = known_labels[best_index]
        else:
            predicted_name = "Unknown"

        y_true.append(person_name)
        y_pred.append(predicted_name)

# === Confusion Matrix ===
labels = class_filter  # Using only the 5 specified classes for confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(10, 8))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Figure 4.3.1: 5-Class Confusion Matrix for Face Recognition")
plt.tight_layout()
plt.show()
