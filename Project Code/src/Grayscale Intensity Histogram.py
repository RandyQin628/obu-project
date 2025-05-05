import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# === Top-level dataset folder with class subfolders ===
image_folder = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/splitted/train"  # should point to the folder with subfolders for each class

all_pixels = []

# === Walk through all subdirectories ===
for root, dirs, files in os.walk(image_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"❌ Cannot read: {img_path}")
                continue

            all_pixels.extend(img.flatten())

# === Check and plot ===
if len(all_pixels) == 0:
    print("⚠️ No pixel data found. Please check your image path and formats.")
    exit()

all_pixels = np.array(all_pixels)

plt.figure(figsize=(10, 6))
plt.hist(all_pixels, bins=256, range=(0, 255), color='gray', alpha=0.75)
plt.title("Grayscale Intensity Distribution")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
