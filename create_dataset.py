import os
import cv2
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the hand landmark model
model_path = 'hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

detector = HandLandmarker.create_from_options(options)

# Dataset directory
DATA_DIR = '/Users/gauravtalele/Desktop/sign-language-detector-python-master/data'

data = []
labels = []

# Process each image in each class
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    if not os.path.isdir(dir_path):
        continue

    print(f" Processing class '{dir_}'...")

    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f" Could not read image {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                x_, y_, data_aux = [], [], []

                for landmark in hand_landmarks:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

# Save the processed data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f" Saved {len(data)} samples to 'data.pickle'")
