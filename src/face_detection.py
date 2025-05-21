# so before face detectio can be used the photos needd to be processed to jpg or png from heic 
import os
import cv2
import pickle
import face_recognition

IMAGE_DIR = "C:\\Users\\james\\FaceChain\\data\\images\\james"
ENCODING_FILE = "C:\\Users\\james\\FaceChain\\data\\encodings.pkl"
FACE_DIR = "C:\\Users\\james\\FaceChain\\data\\data\\faces"
# Ensure the directories exist
os.makedirs(FACE_DIR, exist_ok=True)

encodings = []

for filename in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, filename)
    image = face_recognition.load_image_file(path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        print(f"No face found in {filename}")
        continue

    face_encs = face_recognition.face_encodings(image, face_locations)
    for i, (encoding, loc) in enumerate(zip(face_encs, face_locations)):
        top, right, bottom, left = loc
        face_img = image[top:bottom, left:right]
        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

        # Save cropped face for visual inspection
        save_path = os.path.join(FACE_DIR, f"{filename}_face{i}.jpg")
        cv2.imwrite(save_path, face_bgr)

        encodings.append({
            "name": "james",
            "encoding": encoding
        })

# Save all encodings to file
with open(ENCODING_FILE, "wb") as f:
    pickle.dump(encodings, f)

print(f"✅ Saved {len(encodings)} face encodings to {ENCODING_FILE}")
