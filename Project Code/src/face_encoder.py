import os
import cv2
import face_recognition
import pickle

# Paths
images_dir = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/splitted/train"
encodings_dir = "C:/Users/26506/Desktop/Senior 2/Project/project file/Facial_Recognition_Attendance/Dataset/encodings"


# Iterate over each person's folder in the images directory
for person in os.listdir(images_dir):
    person_dir = os.path.join(images_dir, person)
    if not os.path.isdir(person_dir):
        continue  # Skip if not a folder
    print(f"Processing images for: {person}")

    person_encodings = []  # List to store encodings for this person

    # Process each image in each person's folder
    for image_file in os.listdir(person_dir):
        if image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_path}. Skipping.")
                continue

            # Convert from BGR (OpenCV) to RGB (face_recognition)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face locations and compute encodings
            face_locations = face_recognition.face_locations(rgb_image)
            encodings = face_recognition.face_encodings(rgb_image, face_locations)

            if encodings:
                # Save the first detected face encoding
                person_encodings.append(encodings[0])
            else:
                print(f"No face found in {image_path}.")

    # Save the list of encodings for the person
    if person_encodings:
        encoding_file = os.path.join(encodings_dir, f"{person}.pkl")
        with open(encoding_file, "wb") as f:
            pickle.dump(person_encodings, f)
        print(f"Saved {len(person_encodings)} encoding(s) for {person} in {encoding_file}")
    else:
        print(f"No valid face encodings for {person}.")
