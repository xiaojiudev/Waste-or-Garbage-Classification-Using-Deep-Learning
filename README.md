# 🌱 AI-Powered Waste Classification Application 🚮

This project develops an AI-based system to classify waste from images into categories such as recyclable, organic, hazardous, and reusable waste. The application leverages deep learning models and provides a user-friendly interface for waste identification and disposal guidance.

**Author**: [Ly Dai Phat](https://github.com/xiaojiudev) 👨‍💻

## 📋 Table of Contents

- *Project Overview*
- *Objectives*
- *Dataset*
- *Methodology*
- *Results*
- *Installation*
- *Usage*
- *Project Structure*
- *License*

## 🏷️ Project Overview

The system uses Convolutional Neural Networks (CNNs) to classify waste images into nine categories: battery, cardboard, clothes, glass, metal, organic, paper, plastic, and shoes. It provides disposal instructions to promote sustainable waste management. The backend is built with FastAPI, and the frontend is developed using Flutter for a seamless mobile experience.

## 🎯 Objectives

- Develop an AI model to accurately classify waste into:
  - ♻️ **Recyclable**: Plastic, paper, metal, cardboard, glass
  - 🍃 **Organic**: Organic
  - ☢️ **Hazardous**: Battery
  - 👕 **Reusable**: Clothes, shoes
- Create a user-friendly application allowing users to upload images and receive classification results with disposal guidance.
- Achieve high classification accuracy using state-of-the-art deep learning models.

## 📊 Dataset

- **Sources**:
  - Open datasets from Kaggle and Roboflow.
  - Custom dataset created by capturing waste images in various angles and environments.
- **Preprocessing**:
  - Images resized to 224x224 pixels.
  - Color normalization applied.
  - Data augmentation (rotation, flipping, zooming) to enhance model robustness.
- **Split**: Total 18,416 images
  - Training: 80% of the entire dataset (11,792 images)
  - Validation: 20% of the training set (2,944 images)
  - Testing: 20% of the entire dataset (3,680 images)

## 🧠 Methodology

1. **Data Collection and Preprocessing**:
   - Manual labeling of images into nine categories.
   - Data augmentation to increase dataset diversity.
2. **Model Development**:
   - Implemented three CNN architectures: ResNet-50, EfficientNet-B0, and MobileNetV2.
   - Fine-tuned models using transfer learning with ImageNet weights.
   - Applied techniques like dropout, L2 regularization, and class weighting to mitigate overfitting.
   - Evaluated models using metrics: Accuracy, Precision, Recall, F1-Score.
3. **Application Development**:
   - **Backend**: FastAPI serves the AI model, handling image uploads and URL-based predictions.
   - **Frontend**: Flutter-based mobile app for image uploads and result visualization.
   - Features:
     - Upload waste images via camera or gallery.
     - Display classification results with confidence scores and disposal instructions.

## 📈 Results

The models were evaluated on a test set of 3,680 images. Below are the performance metrics:

### 🏆 ResNet-50

- **Test Accuracy**: 90.65%

- **Classification Report**:

  ```
              precision    recall  f1-score   support
     battery       0.91      0.88      0.89       260
   cardboard       0.95      0.90      0.93       278
     clothes       0.97      0.95      0.96      1006
       glass       0.94      0.81      0.87       401
       metal       0.80      0.90      0.85       272
     organic       0.94      0.94      0.94       363
       paper       0.83      0.94      0.88       309
     plastic       0.83      0.87      0.85       396
       shoes       0.88      0.89      0.89       395
    accuracy                           0.91      3680
  ```
  
### 🥇 EfficientNet-B0

- **Test Accuracy**: 96.98%

- **Classification Report**:

  ```
              precision    recall  f1-score   support
     battery       0.97      0.97      0.97       260
   cardboard       0.98      0.93      0.96       278
     clothes       0.99      1.00      0.99      1006
       glass       0.96      0.93      0.94       401
       metal       0.93      0.96      0.94       272
     organic       0.99      0.99      0.99       363
       paper       0.93      0.97      0.95       309
     plastic       0.94      0.95      0.94       396
       shoes       0.98      0.99      0.99       395
    accuracy                           0.97      3680
  ```

### 🥈 MobileNetV2

- **Test Accuracy**: 95.79%

- **Classification Report**:

  ```
              precision    recall  f1-score   support
     battery       0.96      0.98      0.97       260
   cardboard       0.98      0.91      0.94       278
     clothes       0.99      0.99      0.99      1006
       glass       0.95      0.91      0.93       401
       metal       0.90      0.92      0.91       272
     organic       0.98      0.95      0.97       363
       paper       0.92      0.96      0.94       309
     plastic       0.91      0.94      0.93       396
       shoes       0.96      0.98      0.97       395
    accuracy                           0.96      3680
  ```
  
**🔍 Key Observations**:

- EfficientNet-B0 outperformed others with a test accuracy of 96.98%, followed by MobileNetV2 (95.79%) and ResNet-50 (90.65%).
- Classes like clothes and organic achieved near-perfect classification across models.
- Glass and plastic showed lower precision/recall in ResNet-50, indicating challenges in distinguishing these materials.

## ⚙️ Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/xiaojiudev/Waste-or-Garbage-Classification-Using-Deep-Learning.git
   cd Waste-or-Garbage-Classification-Using-Deep-Learning
   ```

2. **Install Dependencies**:

   - Python 3.10+

   - Install required packages:

     ```bash
     pip install -r requirements.txt
     ```

   - Requirements include `tensorflow`, `fastapi`, `uvicorn`, `httpx`, `pillow`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

3. **Setup Dataset**:

   - Place dataset in `my_dataset/` with `train/` and `test/` subdirectories, each containing class folders (e.g., `battery/`, `cardboard/`).

4. **Setup Model**:

   - Place trained model files (e.g., `efficientnet.keras`) in the project root — Either by downloading a pre-trained version or training your own.

## 🚀 Usage

1. **Run the FastAPI Server**:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **API Endpoints**:

   - **POST** `/predict/file`:
     - Upload an image file to classify.
     - Returns JSON with class, confidence, category, disposal instructions, and top predictions.
   - **POST** `/predict/url`:
     - Provide an image URL to classify.
     - Returns similar JSON response.

3. **Mobile App**:

   - Use the Flutter app to capture/upload images and view results.
   - Ensure the app is configured to communicate with the FastAPI server URL.

## 🗂️ Project Structure

```
flutter_app/                    # Flutter app
waste-classification/
├── main.py                     # FastAPI backend
├── MODEL_*.py                  # Our model (ResNet-50, EfficientNet-B0, MobileNet-V2)
├── my_dataset/                 # Dataset directory
│   ├── train/                  # Training images
│   └── test/                   # Testing images
├── logs/                       # Training logs
├── screen_shot/                # Visualizations (e.g., confusion matrix)
├── *_indices.json              # Class index mapping
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.