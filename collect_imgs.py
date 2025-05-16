import os
import cv2
import string


DATA_DIR = '/Users/gauravtalele/Desktop/sign-language-detector-python-master/data'
os.makedirs(DATA_DIR, exist_ok=True)


gesture_classes = list(string.ascii_uppercase)  # ['A', 'B', ..., 'Z']
dataset_size = 100


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera. Exiting.")
    exit()

# Collect data for each letter
for gesture in gesture_classes:
    class_dir = os.path.join(DATA_DIR, gesture)
    os.makedirs(class_dir, exist_ok=True)

    print(f'📸 Collecting data for gesture "{gesture}"...')

    # Prompt user to prepare
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame. Skipping...")
            continue

        cv2.putText(frame, f'Show: "{gesture}" | Press "Q" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Start capturing frames
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed.")
            continue

        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("⏹️ Early exit requested.")
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Dataset collection complete.")
