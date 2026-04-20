import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load data
A = np.load("data/A.npy")
B = np.load("data/B.npy")
C = np.load("data/C.npy")

# Create labels
labels_A = ["A"] * len(A)
labels_B = ["B"] * len(B)
labels_C = ["C"] * len(C)

# Combine data
X = np.vstack((A, B, C))
y = np.array(labels_A + labels_B + labels_C)

print("Data shape:", X.shape)
print("Labels:", set(y))

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")