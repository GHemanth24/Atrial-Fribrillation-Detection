# Atrial Fibrillation Detection

## 📌 Overview

Atrial Fibrillation Detection is a Machine Learning and web application project developed during my training. With proper guidance and self-learning, I developed and customized the project to make it more practical and user-friendly.

The application processes medical images and patient-related data using a machine learning model and presents the results through a web-based interface.

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Backend:** Python, Flask
- **Machine Learning:** Model training and prediction using the libraries implemented in the project
- **Data Processing:** Pandas, NumPy (where applicable)
- **Dataset:** CSV dataset and sample medical images

## ✨ Features

### 1. Image Upload
- Allows users to upload medical images for processing.
- Provides an interface for entering patient-related data.
- Supports sample inputs for testing the application.

### 2. Image Processing and Prediction
- Uses `imageCheck.py` to verify and process uploaded images.
- Uses `model.py` to perform model-related processing and generate predictions.
- Displays the prediction results through the web interface.

### 3. Results Display
- Presents prediction results in a structured format.
- Displays relevant patient information alongside the results.
- Includes sample output screenshots in the `Output/` directory.

### 4. Model Training
- Includes `training.ipynb` for model training.
- Uses a CSV dataset for training and evaluation.
- Allows experimentation with the dataset and model configuration.

## 📂 Dataset

The dataset is large, so it is hosted externally on Google Drive instead of being stored directly in this GitHub repository.

**Dataset download link:**

[Download the Atrial Fibrillation Detection Dataset from Google Drive](https://drive.google.com/file/d/1aVs1JFG1C0AP6svyWmhqFFHLIjG_jmMz/view?usp=drive_link)

**Instructions:**
1. Open the Google Drive link.
2. Download the dataset file.
3. Extract it if it is provided as a ZIP archive.
4. Place the dataset in the appropriate project directory.
5. Update the dataset path in the training notebook or application if necessary.

Make sure the Google Drive sharing permissions allow users to access the file.

## 📁 Project Structure

```text
Major_Project/
│
├── Execution Logic/
│   ├── app1_final.py
│   ├── imageCheck.py
│   ├── model.py
│   └── index.html
│
├── Training/
│   └── training.ipynb
│
├── Inputs/
│   ├── Sample Images for Input/
│   └── Dataset/
│       └── dataset1.csv  (download separately)
│
├── Output/
│   ├── Image Upload.png
│   ├── Patient Data.png
│   ├── Sample Output1.png
│   └── Sample Output2.png
│
└── Major Project Documentation.docx
```

## ⚙️ Installation

### Prerequisites
- Python 3.x
- Git
- Required Python libraries used by the project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

Replace the example placeholders with your actual GitHub repository URL and project folder name.

### 2. Install Dependencies

If the repository contains a `requirements.txt` file, run:

```bash
pip install -r requirements.txt
```

Otherwise, install the libraries required by the application and training notebook.

### 3. Download the Dataset

Download the dataset from the Google Drive link in the Dataset section and place it in the expected directory.

### 4. Run the Application

Navigate to the directory containing `app1_final.py` and run:

```bash
python app1_final.py
```

If the application starts a local Flask server, open the local URL displayed in the terminal.

### 5. Test the Application

- Upload a sample medical image.
- Provide the required input data.
- View the generated prediction results.

Follow the dataset paths and model-loading requirements defined in the project code.

## 📊 Input and Output

**Inputs**
- Medical images
- Patient-related data, where required
- Dataset used for model training and evaluation

**Outputs**
- Model-generated prediction results
- Processed input information
- Results displayed through the web interface

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. Its predictions should not be used as a substitute for professional medical advice, diagnosis, or treatment.
