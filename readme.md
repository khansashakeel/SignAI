# Signify - Urdu Sign Language Translator

Signify is a web-based application that translates Urdu sign language gestures into text and speech in real-time. It uses deep learning and computer vision to recognize hand gestures from a webcam feed and provides a user-friendly interface for communication.

## Features

- **Real-time Sign Language Recognition**: Translates Urdu sign language gestures from a live webcam feed.
- **Text and Speech Output**: Displays the translated words and sentences, and can also speak them out loud.
- **Autocorrection**: Suggests corrections for the translated words to improve accuracy.
- **User-Friendly Interface**: A simple and intuitive web interface for a seamless user experience.
- **Session Management**: Each user session is tracked for personalized predictions.

## How it Works

The application uses a client-server architecture:

1. **Frontend**: The user interacts with a web page that captures video from their webcam. The JavaScript code on the frontend sends video frames to the backend for processing.
2. **Backend**: A Flask server receives the video frames. It then uses the following pipeline:
    - **Hand Tracking**: `OpenCV` and `MediaPipe` are used to detect and track hand landmarks in the video frames.
    - **Prediction**: The extracted hand landmark data is fed into a pre-trained 1D Convolutional Neural Network (CNN) model built with `TensorFlow` and `Keras`.
    - **Word Formation**: The model's predictions are used to form words.
    - **Autocorrection**: The formed words are autocorrected using a vocabulary of Urdu words and the Levenshtein distance algorithm.
    - **Text-to-Speech**: The final translated sentence is converted into speech using `gTTS` (Google Text-to-Speech).
3. **Frontend Update**: The translated text, autocorrected suggestions, and audio are sent back to the frontend to be displayed to the user.

## Technologies Used

- **Backend**: Python, Flask, TensorFlow, Keras, OpenCV, MediaPipe, gTTS, python-Levenshtein, scikit-learn
- **Frontend**: HTML, CSS, JavaScript

## Setup and Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/signify.git
    cd signify
    ```
2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the application:**
    ```bash
    python backend/app.py
    ```
4. Open your web browser and go to `http://127.0.0.1:5000`.

## Usage

1. Click on the "TRANSLATE" button on the homepage.
2. Allow the browser to access your webcam.
3. Start making Urdu sign language gestures in front of the camera.
4. The recognized words will appear in the "Words Created" box.
5. Use the "Space" button (or press the 'S' key) to add a space between words.
6. Click the "Finalize" button to form a sentence.
7. The autocorrected sentence will be displayed, and you can play the audio of the translated sentence.
8. Use the "Clear" button to start over.
9. Use the "Delete Last" button to remove the last predicted sign.

## Project Structure

```
requirements.txt
backend/
    app.py
    readme.md
    urdu_words.csv
frontend/
    index.html
    predict.html
    static/
        audio/
        css/
            style.css
        images/
            background.jpg
            logo.png
        js/
            predict.js
model/
    CNN1D_label_encoder.pkl
    cnn1d_model.h5
    CNN1D_scaler.pkl
```

## License

This project is licensed under the MIT License.