import os
import pickle
import face_recognition
import cv2
import numpy as np

# === Paths ===
KNOWN_ENCODING_FILE = r"C:\Users\james\FaceChain\data\encodings.pkl"
NEW_IMAGES_DIR = r"C:\Users\james\FaceChain\data\images\group_photos"
OUTPUT_DIR = r"C:\Users\james\FaceChain\data\matched_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load known face encodings ===
with open(KNOWN_ENCODING_FILE, "rb") as f:
    known_faces = pickle.load(f)

known_encodings = [person["encoding"] for person in known_faces]
known_names = [person["name"] for person in known_faces]

# === Track unknown identities ===
unknown_counter = 1

# === Loop through new group photos ===
for filename in os.listdir(NEW_IMAGES_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(NEW_IMAGES_DIR, filename)
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    print(f"📷 {filename}: {len(face_encodings)} face(s) found")

    for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
        # Compare to known encodings
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
        name = "unknown"

        if True in matches:
            matched_index = matches.index(True)
            name = known_names[matched_index]
        else:
            name = f"unknown_{unknown_counter}"
            unknown_counter += 1

        # Save the labeled face image
        top, right, bottom, left = location
        face_image = image[top:bottom, left:right]
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        save_name = f"{os.path.splitext(filename)[0]}_face{i}_{name}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, face_bgr)

        print(f"🔍 Face {i+1}: labeled as {name}, saved to {save_name}")

