from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import requests
from io import BytesIO
from typing import Optional

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model và class names
model = tf.keras.models.load_model("resnet50_sample_for_test.keras")
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    class_names = list(class_indices.keys())

# Mapping phân loại và xử lý rác
WASTE_CATEGORIES = {
    'recyclable': {
        'classes': ['cardboard', 'metal', 'paper', 'plastic',
                   'brown-glass', 'green-glass', 'white-glass'],
        'disposal': 'Có thể tái chế. Vui lòng phân loại vào thùng rác tái chế.'
    },
    'organic': {
        'classes': ['biological'],
        'disposal': 'Rác hữu cơ. Có thể ủ phân hoặc xử lý bằng phương pháp sinh học.'
    },
    'hazardous': {
        'classes': ['battery'],
        'disposal': 'Chất thải nguy hại. Cần xử lý đặc biệt tại các điểm thu gom chuyên dụng.'
    },
    'other': {
        'classes': ['clothes', 'shoes'],
        'disposal': 'Đồ dùng có thể tái sử dụng hoặc quyên góp. Nếu hỏng, vui lòng bỏ vào thùng rác chung.'
    }
}

def preprocess_image(image):
    """Tiền xử lý ảnh đầu vào"""
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def get_waste_info(predicted_class):
    """Lấy thông tin phân loại và cách xử lý"""
    for category, info in WASTE_CATEGORIES.items():
        if predicted_class in info['classes']:
            return {
                'category': category,
                'disposal': info['disposal'],
            }
    return {
        'category': 'unknown',
        'disposal': 'Vui lòng tham khảo hướng dẫn xử lý rác tại địa phương.'
    }

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.post('/predict')
async def predict(
        file: Optional[UploadFile] = File(None),
        url: Optional[str] = Form(None),
):
    print("Received:", file, url)

    try:
        # Kiểm tra đầu vào
        if not file and not url:
            raise HTTPException(status_code=400, detail='Vui lòng cung cấp file ảnh hoặc URL')

        if file and url:
            raise HTTPException(status_code=400, detail='Chỉ chấp nhận một loại đầu vào (file hoặc URL)')

        # Đọc ảnh
        if file:
            image = Image.open(file.file)
        else:
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail='Không thể tải ảnh từ URL')
            image = Image.open(BytesIO(response.content))

        # Tiền xử lý và dự đoán
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)

        # Xử lý kết quả
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        class_name = class_names[class_index]
        waste_info = get_waste_info(class_name)

        return {
            'class': class_name,
            'confidence': confidence,
            'category': waste_info['category'],
            'disposal_instruction': waste_info['disposal'],
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Log lỗi chi tiết
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)