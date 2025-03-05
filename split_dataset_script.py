# import os
# import shutil
# import random
#
# # Đường dẫn thư mục gốc
# base_dir = "my_dataset"
# train_dir = os.path.join(base_dir, "train")
# test_dir = os.path.join(base_dir, "test")
#
# # Tạo thư mục test nếu chưa có
# os.makedirs(test_dir, exist_ok=True)
#
# # Duyệt qua từng thư mục con trong train
# for category in os.listdir(train_dir):
#     category_path = os.path.join(train_dir, category)
#     test_category_path = os.path.join(test_dir, category)
#
#     if os.path.isdir(category_path):  # Kiểm tra xem có phải thư mục không
#         os.makedirs(test_category_path, exist_ok=True)
#
#         # Lấy danh sách tất cả ảnh trong thư mục
#         images = os.listdir(category_path)
#         random.shuffle(images)  # Xáo trộn danh sách để chia ngẫu nhiên
#
#         # Xác định số lượng ảnh cần di chuyển
#         num_test = int(len(images) * 0.3)
#         test_images = images[:num_test]
#
#         # Di chuyển ảnh từ train sang test
#         for img in test_images:
#             src_path = os.path.join(category_path, img)
#             dest_path = os.path.join(test_category_path, img)
#             shutil.move(src_path, dest_path)
#
# print("Dữ liệu đã được chia thành công theo tỷ lệ 7:3.")