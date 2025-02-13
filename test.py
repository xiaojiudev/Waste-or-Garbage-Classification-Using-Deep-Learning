import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models
ResNet50 = tf.keras.applications.ResNet50
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping

from tensorflow.keras import mixed_precision
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
    )

logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
# mixed_precision.set_global_policy('mixed_float16')
# tf.keras.backend.clear_session()

# Thiết lập đường dẫn
train_path = 'D:/Study/CNTT/A.MHUD/CNN_Practice/train'
test_path = 'D:/Study/CNTT/A.MHUD/CNN_Practice/test'

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    # brightness_range=[0.5, 1.5], # NOTE: không xài cái này, làm ảnh gốc và ảnh augmentation khác nhau rất lớn
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Tạo dataset từ thư mục
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Lấy một batch từ train_generator
batch_images, batch_labels = next(train_generator)

# Lấy một ảnh từ batch
original_image = batch_images[0]

# Hiển thị ảnh gốc
plt.figure(figsize=(3, 3))
plt.imshow(original_image)
plt.title("Ảnh Gốc (Đã Rescale)")
plt.axis("off")
plt.show()

# Hiển thị các ảnh đã Augmentation
plt.figure(figsize=(12, 6))
for i in range(6):  # Hiển thị 6 ảnh đã biến đổi
    augmented_image = train_datagen.random_transform(original_image)  # Áp dụng Augmentation lên ảnh gốc
    augmented_image = np.clip(augmented_image, 0, 1)  # Đảm bảo giá trị nằm trong [0, 1]
    plt.subplot(2, 3, i + 1)
    plt.imshow(augmented_image)  # Lấy ảnh đầu tiên trong batch mới
    plt.title(f"Ảnh Augmentation {i+1}")
    plt.axis("off")

plt.show()

# NOTE: Load mô hình ResNet50
# NOTE: Kích thước ảnh đầu vào (224x224 pixels, 3 kênh màu RGB).
# NOTE: weights='imagenet': Sử dụng trọng số đã được huấn luyện sẵn trên tập ImageNet.
# NOTE: include_top=False: Không sử dụng phần Fully Connected Layer gốc của ResNet50 (vì ta sẽ thay bằng lớp FC riêng).
base_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
# for layer in base_model.layers:
#     layer.trainable = False

# NOTE: Bật fine-tuning trên ResNet50
# NOTE: Là quá trình tiếp tục huấn luyện một mô hình đã được huấn luyện trước bằng cách điều chỉnh một số lớp cụ thể thay vì huấn luyện từ đầu.
# NOTE: Cách làm này giúp mô hình học thêm các đặc trưng cụ thể của dữ liệu mới mà không làm mất đi kiến thức tổng quát từ ImageNet.
# NOTE: Nếu ta không fine-tune, mô hình chỉ sử dụng các đặc trưng có sẵn của ResNet50 mà không điều chỉnh cho bài toán cụ thể.
for layer in base_model.layers[-50:]:  # NOTE: Fine-tune 50 lớp cuối - 50 lớp cuối cùng của ResNet50 sẽ tiếp tục được huấn luyện trên dữ liệu mới thay vì giữ nguyên trọng số cũ.
    layer.trainable = True

# NOTE: Thêm các lớp Fully Connected
x = layers.Flatten()(base_model.output) # NOTE: Chuyển đầu ra từ ResNet50 thành một vector 1 chiều.
x = layers.Dense(512, activation='relu')(x) #  NOTE: Một lớp FC với 512 neuron và hàm kích hoạt ReLU.
x = layers.Dropout(0.5)(x) # NOTE: Bỏ ngẫu nhiên 50% neuron trong quá trình huấn luyện để tránh overfitting.
x = layers.Dense(7, activation='softmax')(x) # NOTE: Lớp đầu ra có 7 neuron (tương ứng với 7 lớp phân loại) và sử dụng hàm kích hoạt softmax.
model = models.Model(inputs=base_model.input, outputs=x)

# NOTE: Compile mô hình - là bước cấu hình thuật toán tối ưu, hàm mất mát và các metrics để mô hình có thể học.
# NOTE: nên sử dụng learning rate nhỏ hơn để tránh mô hình bị "quên" những gì ResNet50 đã học:
# NOTE: Optimizer (Bộ tối ưu hóa): Dùng Adam với learning_rate=1e-4 để cập nhật trọng số của mô hình.
# NOTE: Loss function (Hàm mất mát): Dùng categorical_crossentropy vì đây là bài toán phân loại nhiều lớp (multi-class classification).
# NOTE: Metrics: Theo dõi accuracy để đánh giá độ chính xác trong quá trình huấn luyện.
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# NOTE: Mục đích của compile: Giúp mô hình sẵn sàng để huấn luyện bằng cách xác định cách nó sẽ học và tối ưu trọng số.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# NOTE: Callbacks - là các hàm được gọi tự động trong quá trình huấn luyện để kiểm soát quá trình training.
# NOTE: Mục đích giúp tối ưu hiệu suất mô hình,tránh overfitting, cải thiện tốc độ hội tụ bằng cách điều chỉnh learning rate.
# NOTE: ModelCheckpoint - Lưu mô hình tốt nhất dựa trên val_loss
checkpoint = ModelCheckpoint('resnet50_best.h5', save_best_only=True, monitor='val_loss', mode='min')

# NOTE: EarlyStopping - Dừng huấn luyện sớm nếu val_loss không cải thiện sau 10 epoch, tránh overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# NOTE: ReduceLROnPlateau - Nếu val_loss không giảm sau 5 epoch, giảm learning_rate xuống 20% để model học tốt hơn
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Đánh giá mô hình
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Đánh giá trên tập test
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.4f}')

# Dự đoán trên tập test
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# In ra classification report
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# In ra confusion matrix
print(confusion_matrix(y_true, y_pred))

# NOTE: Support - Số lượng mẫu thực tế thuộc về mỗi lớp trong tập dữ liệu kiểm tra.
# NOTE: Macro Avg: Trung bình cộng của precision, recall và F1-score trên tất cả các lớp mà không xét đến số lượng mẫu của từng lớp.
#  Điều này giúp đánh giá mô hình một cách công bằng trên tất cả các lớp, kể cả những lớp có ít mẫu.
# NOTE: Weighted Avg - Trung bình có trọng số của precision, recall và F1-score, trong đó trọng số là số lượng mẫu của từng lớp.
#  Nó giúp phản ánh độ chính xác của mô hình theo tỷ lệ kích thước của từng lớp.