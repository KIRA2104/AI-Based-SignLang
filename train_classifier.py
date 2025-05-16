import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# === Paths ===
data_path = '/Users/gauravtalele/Desktop/sign-language-detector-python-master/data.pickle'
model_path = '/Users/gauravtalele/Desktop/sign-language-detector-python-master/model.p'

# === Load the landmark dataset ===
with open(data_path, 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# === Encode labels as numbers (if they're not already) ===
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# === Split into training and test sets ===
x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded, random_state=42
)

# === Train the classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# === Predict and evaluate ===
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_pred, y_test)

print(f" Model Accuracy: {accuracy * 100:.2f}%")

# === Save model and label encoder ===
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'label_encoder': label_encoder
    }, f)

print(f"Model saved to: {model_path}")
