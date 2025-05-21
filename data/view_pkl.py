import pickle

with open("C:\\Users\\james\\FaceChain\\data\\encodings.pkl", "rb") as f:
    encodings = pickle.load(f)

print(f"Loaded {len(encodings)} encodings.")
print("Example encoding vector (first 5 values):")
print(encodings[0])


