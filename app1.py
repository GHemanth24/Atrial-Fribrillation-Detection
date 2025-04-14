import pandas as pd
import os
import pickle
from flask import Flask, render_template, request
import numpy as np
import cv2
from scipy.signal import resample

app = Flask(__name__)

# Configurations for image upload
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Set up the app's upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Load the pickle model
model = None
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to preprocess ECG image
# def preprocess_ecg_image(image_path):
#     """Load and preprocess the ECG image."""
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.resize(img, (1000, 800))  # Resize for consistency
#     img = cv2.GaussianBlur(img, (5, 5), 0)
#     _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
#     return img
def preprocess_ecg_image(image_path):
    """Load and preprocess the ECG image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Could not load image at {image_path}")

    img = cv2.resize(img, (1000, 800))  # Resize for consistency
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    return img



# Function to extract 12 lead waveforms
def extract_leads(img):
    """Extract the 12 lead waveforms from the image."""
    lead_height = img.shape[0] // 12  # Assuming 12 evenly spaced leads
    leads = [img[i*lead_height:(i+1)*lead_height, :] for i in range(12)]
    return leads


# Function to convert lead image to 1D signal
def convert_to_signal(lead_img):
    """Convert a single lead image into a 1D waveform."""
    height, width = lead_img.shape
    signal = np.zeros(width)
    for x in range(width):
        y_values = np.where(lead_img[:, x] == 255)[0]
        if len(y_values) > 0:
            signal[x] = np.mean(y_values)  # Average pixel height

    # Normalize and invert (if needed)
    signal = (height - signal) / height
    return signal


# Function to process ECG image to numpy array
def process_ecg_image(image_path):
    """Complete pipeline: image to numpy array (12, 5000)."""
    img = preprocess_ecg_image(image_path)
    leads = extract_leads(img)

    signals = [convert_to_signal(lead) for lead in leads]
    # Resample each lead to 5000 points (10 sec at 500Hz)
    signals_resampled = [resample(sig, 5000) for sig in signals]
    return np.array(signals_resampled)  # Shape: (12, 5000)


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Handling form data (if age, height, weight, gender are provided)
    if 'age' in request.form and 'height' in request.form and 'weight' in request.form and 'gender' in request.form:
        try:
            age = (request.form['age'])
            height = (request.form['height'])
            weight = (request.form['weight'])
            sex = 0 if request.form['gender'] == 'male' else 1
        except Exception:
            return render_template('index.html', prediction_text="Error in prediction. Please check your inputs.")

    # Handling image upload for ECG image
    if 'image' not in request.files:
        return render_template('index.html', prediction_text="No image file found.")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction_text="No selected file.")

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded ECG image and make a prediction
        ecg_data = process_ecg_image(filepath)
        np.save("ecg_output.npy", ecg_data)

        arr = np.load('ecg_output.npy')
        input_data = arr[5]
        order = [1, 3, 5, 4, 2, 0, 6, 7, 8, 9, 10, 11]
        input_data = input_data[order]
        input_data = np.append(input_data, [age, sex, height, weight])
        feature_names = [
            'I', 'II', 'III', 'aVF', 'aVR', 'aVL',
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
            'age', 'sex', 'height', 'weight'
        ]

        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_df = input_df.astype('float64')
        prediction = model.predict(input_df)
        print('model working')
        return render_template('index.html', prediction_text=f"Prediction from ECG image: {prediction[0]}")
    else:
        return render_template('index.html', prediction_text="Invalid file format. Only PNG, JPG, JPEG, GIF are allowed.")


if __name__ == "__main__":
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
