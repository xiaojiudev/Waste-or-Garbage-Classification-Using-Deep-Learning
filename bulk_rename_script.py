# import os
#
# # Đường dẫn đến thư mục train
# train_dir = 'my_dataset/train/'
#
# # Duyệt qua từng thư mục con trong thư mục train
# for folder_name in os.listdir(train_dir):
#     folder_path = os.path.join(train_dir, folder_name)
#
#     # Kiểm tra xem có phải là thư mục không
#     if os.path.isdir(folder_path):
#         count = 1  # Bắt đầu đếm từ 1 cho mỗi thư mục
#
#         # Lấy danh sách file và sắp xếp
#         files = sorted(os.listdir(folder_path))
#
#         for filename in files:
#             file_path = os.path.join(folder_path, filename)
#
#             # Chỉ xử lý file thông thường, không phải thư mục
#             if os.path.isfile(file_path):
#                 # Tách phần tên và đuôi file
#                 _, ext = os.path.splitext(filename)
#                 ext = ext.lower()  # Chuẩn hóa đuôi file về chữ thường
#
#                 # Kiểm tra định dạng ảnh hợp lệ
#                 if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
#                     # Tạo tên mới và đường dẫn mới
#                     new_name = f"{folder_name}_{count}{ext}"
#                     new_path = os.path.join(folder_path, new_name)
#
#                     # Đổi tên file
#                     os.rename(file_path, new_path)
#                     count += 1  # Tăng biến đếm