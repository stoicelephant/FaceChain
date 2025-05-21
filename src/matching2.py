import os
import pickle
import face_recognition
import cv2
import numpy as np

# === Your Paths ===
KNOWN_ENCODING_FILE = r"C:\Users\james\FaceChain\data\encodings.pkl"
NEW_IMAGES_DIR = r"C:\Users\james\FaceChain\data\images\group_photos"
OUTPUT_DIR = r"C:\Users\james\FaceChain\data\matched_faces"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")

# === Create output folders if needed ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# === Load known face encodings ===
with open(KNOWN_ENCODING_FILE, "rb") as f:
    known_faces = pickle.load(f)

known_encodings = [person["encoding"] for person in known_faces]
known_names = [person["name"] for person in known_faces]

# === Track unknown identities ===
unknown_counter = 1
MATCH_THRESHOLD = 0.5  # updated tolerance

# === Loop through new group photos ===
for filename in os.listdir(NEW_IMAGES_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(NEW_IMAGES_DIR, filename)
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    print(f"📷 {filename}: {len(face_encodings)} face(s) found")

    # Convert to BGR for annotation
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
        # Compute distances to known faces
        distances = face_recognition.face_distance(known_encodings, encoding)
        min_distance = np.min(distances)
        best_index = np.argmin(distances)
        best_name = known_names[best_index]

        print(f"→ Face {i+1} best match: {best_name} (distance: {min_distance:.3f})")

        if min_distance < MATCH_THRESHOLD:
            name = best_name
        else:
            name = f"unknown_{unknown_counter}"
            unknown_counter += 1

        # Save cropped face image
        top, right, bottom, left = location
        face_image = image[top:bottom, left:right]
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        save_name = f"{os.path.splitext(filename)[0]}_face{i}_{name}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, face_bgr)

        print(f"🔍 Saved: {save_name}")

        # Annotate the original image
        label_text = f"{name} ({min_distance:.2f})" if 'unknown' not in name else name
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_bgr, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save the annotated full image
    annotated_path = os.path.join(ANNOTATED_DIR, f"annotated_{filename}")
    cv2.imwrite(annotated_path, image_bgr)
    print(f"🖼️ Annotated image saved: {annotated_path}\n")
