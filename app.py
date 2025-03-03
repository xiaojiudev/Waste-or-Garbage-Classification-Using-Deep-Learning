import asyncio

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np
import json
import requests
import httpx
from io import BytesIO
from typing import Optional, Annotated
from pydantic import BaseModel

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình mức sử dụng bộ nhớ GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
    )

# Load model và class names
model = tf.keras.models.load_model("resnet50_sample_for_test.keras")
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
    class_names = list(class_indices.keys())

# Mapping phân loại và xử lý rác
WASTE_CATEGORIES = {
    "recyclable": {
        "classes": ["cardboard", "metal", "paper", "plastic",
                   "brown-glass", "green-glass", "white-glass"],
        "disposal": "Có thể tái chế. Vui lòng phân loại vào thùng rác tái chế."
    },
    "organic": {
        "classes": ["biological"],
        "disposal": "Rác hữu cơ. Có thể ủ phân hoặc xử lý bằng phương pháp sinh học."
    },
    "hazardous": {
        "classes": ["battery"],
        "disposal": "Chất thải nguy hại. Cần xử lý đặc biệt tại các điểm thu gom chuyên dụng."
    },
    "other": {
        "classes": ["clothes", "shoes"],
        "disposal": "Đồ dùng có thể tái sử dụng hoặc quyên góp. Nếu hỏng, vui lòng bỏ vào thùng rác chung."
    }
}

def preprocess_image(image : Image.Image) -> np.ndarray:
    """Tiền xử lý ảnh đầu vào"""
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def get_waste_info(predicted_class : str) -> dict:
    """Lấy thông tin phân loại và cách xử lý"""
    for category, info in WASTE_CATEGORIES.items():
        if predicted_class in info["classes"]:
            return {
                "category": category,
                "disposal": info["disposal"],
            }
    return {
        "category": "unknown",
        "disposal": "Vui lòng tham khảo hướng dẫn xử lý rác tại địa phương."
    }

async def predict_image(image: Image.Image) -> dict:
    """Logic xử lý dự đoán chung cho 2 endpoint"""
    try:
        # Tiền xử lý và dự đoán
        processed_image = preprocess_image(image)

        # Chạy dự đoán mô hình trong nhóm luồng để tránh bị chặn
        predictions = await asyncio.to_thread(model.predict, processed_image, verbose=0)

        # Xử lý kết quả
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        class_name = class_names[class_index]
        waste_info = get_waste_info(class_name)

        # Get top N classes sorted by confidence
        top_n = 5  # Có thể điều chỉnh số này dựa trên số lượng lớp bạn muốn theo dõi
        sorted_indices = np.argsort(predictions[0])[::-1]  # Sắp xếp dự đoán theo thứ tự giảm dần
        top_classes = [(class_names[i], float(predictions[0][i])) for i in sorted_indices[:top_n]]

        # Print or log the classes with their confidence
        top_predictions = [{"class": name, "confidence": conf} for name, conf in top_classes]

        return {
            "class": class_name,
            "confidence": confidence,
            "category": waste_info["category"],
            "disposal_instruction": waste_info["disposal"],
            "top_predictions": top_predictions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/file")
async def predict(item: Annotated[UploadFile, File(..., description="Image file to classify")]):
    print("Received:", item)

    try:
        # Kiểm tra định dạng hình ảnh
        if not item.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail= "Định dạng không hợp lệ. Vui lòng chọn tệp tin có định dạng là hình ảnh.")

        # Đọc ảnh
        with Image.open(item.file) as img:
            image = img.copy()
            return await predict_image(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Tệp hình ảnh không hợp lệ hoặc bị hỏng.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ImageURLItem(BaseModel):
    url: str

@app.post("/predict/url")
async def predict(item: ImageURLItem):
    print("Received:", item)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(item.url)
            if response.status_code != 200:
                raise HTTPException(status_code=404, detail="Không thể tải ảnh từ URL")

            # Validate image content
            try:
                image = Image.open(BytesIO(response.content))
            except UnidentifiedImageError:
                raise HTTPException(status_code=404, detail="Hình ảnh từ URL không hợp lệ.")
            return await predict_image(image)
    except httpx.RequestError:
        raise HTTPException(status_code=500, detail="Không thể kết nối tới URL hình ảnh.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, )