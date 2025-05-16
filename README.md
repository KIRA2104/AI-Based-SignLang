# 🤟 Sign Language Detection System

This is a real-time Sign Language Detection System using OpenCV, MediaPipe, and a Machine Learning model trained on custom hand gesture data. The system recognizes hand gestures representing the English alphabet (A–Z) and can build words based on live webcam input.

---

## 📁 Project Structure

sign-language-detector-python/
│
├── data/ # Collected hand gesture images for each letter
├── hand_landmarker.task # MediaPipe hand landmark detection model
├── model.p # Trained Random Forest model
├── data.pickle # Processed dataset with hand landmark features
├── Sign_alphabet_chart_abc.jpg # Reference image for gesture guide
├── collect_data.py # Script to collect images for A–Z signs
├── process_data.py # Extracts landmarks and stores them
├── train_model.py # Trains the classifier
└── run_prediction.py # Real-time webcam-based sign detection


---

##Tech Stack

- **Python 3.8+**
- **OpenCV**
- **MediaPipe**
- **scikit-learn**
- **NumPy / Pickle**
- **Real-time webcam processing**

---

##Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/sign-language-detector-python.git
cd sign-language-detector-python
