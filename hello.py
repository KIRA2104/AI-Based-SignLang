import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# === Load model and label encoder ===
model_path = '/Users/gauravtalele/Desktop/sign-language-detector-python-master/model.p'
model_dict = pickle.load(open(model_path, 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']

# === Webcam setup ===
cap = cv2.VideoCapture(0)

# === Mediapipe hands setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# === Prediction and word logic ===
predictions_buffer = deque(maxlen=5)
buffer_threshold = 3
confirmed_letters = []
last_saved_letter = None
current_prediction = ""

# === Load reference image ===
reference_image_path = '/Users/gauravtalele/Desktop/sign-language-detector-python-master/Sign_alphabet_chart_abc.jpg'
reference_image = cv2.imread(reference_image_path)
if reference_image is not None:
    reference_image = cv2.resize(reference_image, (300, 200))
else:
    print(f"❌ Could not load reference image at: {reference_image_path}")

exit_after_print = False

while True:
    if exit_after_print:
        break

    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            x1 = int(min(x_) * W) - 20
            y1 = int(min(y_) * H) - 20
            x2 = int(max(x_) * W) + 20
            y2 = int(max(y_) * H) + 20

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = label_encoder.inverse_transform(prediction)[0]

            predictions_buffer.append(predicted_character)
            if len(predictions_buffer) == predictions_buffer.maxlen:
                most_common = max(set(predictions_buffer), key=predictions_buffer.count)
                if predictions_buffer.count(most_common) >= buffer_threshold:
                    current_prediction = most_common
    else:
        current_prediction = ""  # Only reset if no hand is detected

    # === Display reference image top-right ===
    if reference_image is not None:
        ref_h, ref_w = reference_image.shape[:2]
        frame[0:ref_h, W - ref_w:W] = reference_image

    # === Display the current word and predicted letter ===
    word_display = ''.join(confirmed_letters)
    cv2.putText(frame, f"Word: {word_display}", (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    if current_prediction:
        cv2.putText(frame, f"Prediction: {current_prediction}", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

    # Show frame
    cv2.imshow('Sign Language Recognition', frame)

    # === Key press handling ===
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r') and confirmed_letters:
        removed = confirmed_letters.pop()
        last_saved_letter = None
    elif key == ord('a') and current_prediction:
        if current_prediction != last_saved_letter:
            confirmed_letters.append(current_prediction)
            last_saved_letter = current_prediction
    elif key == ord('e'):
        final_word = ''.join(confirmed_letters)
        frame[:] = (0, 0, 0)  # Black out screen
        cv2.putText(frame, f"Final Word: {final_word}", (50, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Sign Language Recognition', frame)
        cv2.waitKey(3000)  # Show final word for 3 seconds
        exit_after_print = True

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
