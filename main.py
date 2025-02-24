import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
from io import BytesIO

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
    )

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("resnet50_sample_for_test.keras")

# Load class names từ file json
with open('class_indices.json', 'r') as f:  # Đảm bảo file tồn tại cùng thư mục
    class_indices = json.load(f)
    class_names = list(class_indices.keys())

# Hàm dự đoán ảnh mới
def predict_image(img_path_or_url):
    if img_path_or_url.startswith('http://') or img_path_or_url.startswith('https://'):
        response = requests.get(img_path_or_url)
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))
    else:
        img = image.load_img(img_path_or_url, target_size=(224, 224))  # Resize ảnh
    img_array = image.img_to_array(img) / 255.0  # Chuẩn hóa
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất
    class_name = class_names[class_index] # Lấy tên nhãn từ danh sách labels
    confidence = float(np.max(prediction))  # Lấy độ tin cậy của dự đoán

    return class_index, class_name, confidence

# img_path = "plastic145.jpg"
img_path = "https://ychef.files.bbci.co.uk/1280x720/p06z3hgy.jpg"
predicted_class, predicted_name, confidence = predict_image(img_path)
print(f"Predicted class: {predicted_class}, Predicted name: {predicted_name} Confidence: {confidence}")