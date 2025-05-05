import os
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

# Set the path to the dataset directory
dataset_path = "C:/Users/26506/Desktop/Senior 2/Project/project file/obu-project/obu-project/Dataset/images"

# Initialize variables
class_names = []
images_count = []
image_samples = []
pca_images = []
pca_labels = []
all_pixel_values = []

# 1. Class Distribution
for person in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person)
    if os.path.isdir(person_dir):
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_names.append(person)
        images_count.append(len(image_files))

        # Collect random samples for visualization
        if image_files:
            sample_image_path = os.path.join(person_dir, random.choice(image_files))
            image_samples.append(np.array(Image.open(sample_image_path)))

        # Aggregate pixel intensities
        for img_file in image_files[:10]:  # Limit to 10 images per class for speed
            img_path = os.path.join(person_dir, img_file)
            img = np.array(Image.open(img_path).convert('L'))  # Grayscale
            all_pixel_values.extend(img.flatten())

# Plot class distribution
plt.figure(figsize=(12, 6))
plt.bar(range(len(class_names)), images_count, color='skyblue')
plt.xlabel('Class Index (Individual)')
plt.ylabel('Number of Images')
plt.title('Figure 4.1: Class Distribution (Images per Public Figure)')
plt.tight_layout()
plt.show()

# 2. Sample Images Visualization
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Figure 4.2: Representative Sample Images from PubFig Dataset')
for i, ax in enumerate(axes.flat):
    if i < len(image_samples):
        ax.imshow(image_samples[i])
        ax.set_title(class_names[i], fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()

# --------- Step 3: Load images and extract pixel data ---------
for person in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person)
    if not os.path.isdir(person_dir):
        continue

    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Store name
    class_names.append(person)

    # Only use first 5 images per person to reduce compute for PCA
    for img_file in image_files[:5]:
        img_path = os.path.join(person_dir, img_file)
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
            resized_img = img.resize((64, 64))  # Resize for PCA
            img_array = np.array(resized_img).flatten()

            all_pixel_values.extend(img_array)
            pca_images.append(img_array)
            pca_labels.append(person)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# 4. PCA Visualization
# Gather images data for PCA (limited subset due to performance)
pca_images = []
pca_labels = []
for person in class_names[:20]:  # Limit to 20 classes for PCA visualization clarity
    person_dir = os.path.join(dataset_path, person)
    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in image_files[:5]:  # Up to 5 images per class
        img_path = os.path.join(person_dir, img_file)
        img = np.array(Image.open(img_path).resize((64,64)).convert('L')).flatten()  # Resized for PCA
        pca_images.append(img)
        pca_labels.append(person)

pca_images = np.array(pca_images)

# Apply PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(pca_images)

# Plot PCA results
plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_result[:,0], pca_result[:,1], c=[class_names.index(lbl) for lbl in pca_labels], cmap='tab20', alpha=0.6)
plt.legend(handles=scatter.legend_elements()[0], labels=class_names[:20], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Figure 4.4: PCA Visualization (First Two Components)')
plt.tight_layout()
plt.show()
