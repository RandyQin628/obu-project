import os
import shutil
import random
import math

# === CONFIGURATION ===
original_dataset = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/images"
output_root = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/splitted"

# === OUTPUT FOLDERS ===
train_dir = os.path.join(output_root, "train")
test_dir = os.path.join(output_root, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

split_summary = []

# === PROCESS ALL PEOPLE ===
for person in os.listdir(original_dataset):
    person_path = os.path.join(original_dataset, person)
    if not os.path.isdir(person_path):
        continue
    images = [img for img in os.listdir(person_path) if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    if len(images) < 2:
        continue

    random.shuffle(images)
    split_idx = math.floor(len(images) * 0.8)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    # Create subdirectories
    train_person_dir = os.path.join(train_dir, person)
    test_person_dir = os.path.join(test_dir, person)
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(test_person_dir, exist_ok=True)

    # Copy images
    for img in train_images:
        shutil.copy(os.path.join(person_path, img), os.path.join(train_person_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(person_path, img), os.path.join(test_person_dir, img))

    split_summary.append((person, len(train_images), len(test_images)))

# === PRINT SUMMARY
print("\n✅ Dataset split complete:")
for entry in split_summary:
    print(f"{entry[0]} — Train: {entry[1]}, Test: {entry[2]}")
