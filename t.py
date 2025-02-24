import requests

url = "http://localhost:8000/predict"

# Gửi dưới dạng form-data
response = requests.post(
    url,
    files={},  # Bắt buộc có trường files ngay cả khi gửi URL
    data={"url": "https://www.sas.org.uk/wp-content/uploads/2021/08/sas-donate-page-pollution-1920x1080.jpg"}
)

print(response.status_code)
print(response.json())