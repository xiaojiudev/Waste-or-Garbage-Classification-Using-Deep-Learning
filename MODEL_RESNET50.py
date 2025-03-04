import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau

keras = tf.keras
layers = tf.keras.layers
Model = tf.keras.models.Model
ResNet50 = tf.keras.applications.ResNet50
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping

print("GPU available:", tf.config.list_physical_devices("GPU"))
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
    )

logical_gpus = tf.config.list_logical_devices("GPU")
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

# Thiết lập đường dẫn
train_path = "D:/Study/CNTT/A.MHUD/CNN_Practice/kaggle_dataset/train"
test_path = "D:/Study/CNTT/A.MHUD/CNN_Practice/kaggle_dataset/test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255, # Chuẩn hoá về [0, 1]
    rotation_range=40, # Xoay một góc 40 độ
    width_shift_range=0.2, # Di chuyển hình ảnh theo chiều ngang (trái hoặc phải) với một phạm vi tối đa là 20% so với chiều rộng của hình ảnh.
    height_shift_range=0.2, # Di chuyển hình ảnh theo chiều dọc (lên hoặc xuống) với một phạm vi tối đa là 20% so với chiều cao của hình ảnh.
    shear_range=0.2, # Xén hình ảnh, hình ảnh sẽ bị nghiêng một góc
    zoom_range=0.2, # Phóng to hình ảnh 20%
    horizontal_flip=True, # Lật ngang hình ảnh, giống như dối xứng qua gương
    # brightness_range=[0.5, 1.5], # NOTE: không xài cái này, làm ảnh gốc và ảnh augmentation khác nhau rất lớn
    fill_mode="nearest", # Điền vào các pixel bị thiếu
    validation_split=0.2 # Tách 20% làm validation
)

# Chỉ áp dụng augmentation cho tập train, tập test CHỈ rescale. Nếu không sẽ gây nhiễu dữ liệu và làm sai lệch kết quả đánh giá.
test_datagen = ImageDataGenerator(
    rescale=1.0/255,
)

# Tạo dataset từ thư mục
# batch_size xác định số lượng hình ảnh được xử lý trong một lần (batch) khi mô hình huấn luyện.
# class_mode xác định cách nhãn (labels) được xử lý trong dữ liệu đầu vào: categorical, binary, sparse
training_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    subset="training",
)

# Tạo validation data (20%)
validation_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"  # Phần dành cho validation
)

testing_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False # NOTE: Đảm bảo thứ tự dữ liệu test không bị xáo trộn, giúp đánh giá metrics chính xác
)

# Lưu class indices vào file json
with open("class_indices.json", "w") as f:
    json.dump(training_data.class_indices, f)

# Kiểm tra tỷ lệ số lượng mẫu của từng lớp:
from collections import Counter
print("Training class distribution:", Counter(training_data.classes))
print("Testing class distribution:", Counter(testing_data.classes))

# Lấy một batch từ training_data
batch_images, batch_labels = next(training_data)

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
#  Kích thước ảnh đầu vào (224x224 pixels, 3 kênh màu RGB).
#  weights="imagenet": Sử dụng trọng số đã được huấn luyện sẵn trên tập ImageNet.
#  include_top=False: Không sử dụng phần Fully Connected Layer gốc của ResNet50 (vì ta sẽ thay bằng lớp FC riêng).
BASE_MODEL = ResNet50(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

# NOTE: Fine-tuning - Cho phép huấn luyện lại các lớp cuối của ResNet50
#  Là quá trình tiếp tục huấn luyện một mô hình đã được huấn luyện trước bằng cách điều chỉnh một số lớp cụ thể thay vì huấn luyện từ đầu.
#  Cách làm này giúp mô hình học thêm các đặc trưng cụ thể của dữ liệu mới mà không làm mất đi kiến thức tổng quát từ ImageNet.
#  Nếu ta không fine-tune, mô hình chỉ sử dụng các đặc trưng có sẵn của ResNet50 mà không điều chỉnh cho bài toán cụ thể.
#  layer.trainable = False - trọng số của lớp này sẽ không thay đổi trong quá trình huấn luyện
#  layer.trainable = True - trọng số của lớp này sẽ được huấn luyện và cập nhật trọng số bình thường
for layer in BASE_MODEL.layers[-50:]:  # NOTE: Fine-tune 50 lớp cuối
    layer.trainable = True

# NOTE: Thêm các lớp mới vào mô hình
x = BASE_MODEL.output

# NOTE: Lớp Conv2D, MaxPooling2D, BatchNormalization thêm vào sau ResNet50
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)

# NOTE: Thêm các lớp Fully Connected
#  Dropout - Bỏ ngẫu nhiên 50% neuron trong quá trình huấn luyện để tránh overfitting.
#  Dense(512, activation="relu") - Một lớp FC với 512 neuron và hàm kích hoạt ReLU.
x = layers.Flatten()(x) # NOTE: Chuyển đầu ra từ ResNet50 thành một vector 1 chiều.
x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

# NOTE: Lớp đầu ra phân loại (11 lớp)
prediction = layers.Dense(11, activation="softmax")(x)

MODEL_RESNET50 = Model(inputs=BASE_MODEL.input, outputs=prediction)
MODEL_RESNET50.summary()

# NOTE: Callbacks - là các hàm được gọi tự động trong quá trình huấn luyện để kiểm soát quá trình training.
#  Mục đích giúp tối ưu hiệu suất mô hình,tránh overfitting, cải thiện tốc độ hội tụ bằng cách điều chỉnh learning rate.
MODEL_RESNET50.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

# NOTE: ModelCheckpoint - Lưu mô hình tốt nhất dựa trên val_loss
checkpoint = ModelCheckpoint(
    filepath="resnet50_TEST.keras",
    verbose=False,
    save_best_only=True,
    monitor="val_loss",
    mode="min")

# NOTE: EarlyStopping - Dừng huấn luyện sớm nếu val_loss không cải thiện sau 10 epoch, tránh overfitting
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    mode="min")

# NOTE: ReduceLROnPlateau - Nếu val_loss không giảm sau 5 epoch, giảm learning_rate xuống 20% để model học tốt hơn
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=3,
    min_lr=1e-7)

# NOTE: Cân bằng lại tập dữ liệu - dùng class_weight trong model.fit()
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(training_data.classes),
    y=training_data.classes)
class_weights_dict = dict(enumerate(class_weights))

# Huấn luyện mô hình
history = MODEL_RESNET50.fit(
    training_data,
    steps_per_epoch=len(training_data),
    epochs=50,
    validation_data=validation_data,
    validation_steps=len(validation_data),
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# Đánh giá mô hình
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# Đánh giá trên tập test
test_loss, test_acc = MODEL_RESNET50.evaluate(testing_data, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Dự đoán trên tập test
y_pred = MODEL_RESNET50.predict(testing_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = testing_data.classes

# NOTE: In ra classification report
#  zero_division=1 tránh lỗi chia cho 0 khi một lớp không có mẫu nào được dự đoán
print(classification_report(y_true, y_pred, target_names=testing_data.class_indices.keys(), zero_division=1))

# In ra confusion matrix
print(confusion_matrix(y_true, y_pred))

# NOTE: Support - Số lượng mẫu thực tế thuộc về mỗi lớp trong tập dữ liệu kiểm tra.
#  Macro Avg: Trung bình cộng của precision, recall và F1-score trên tất cả các lớp mà không xét đến số lượng mẫu của từng lớp.
#       Điều này giúp đánh giá mô hình một cách công bằng trên tất cả các lớp, kể cả những lớp có ít mẫu.
#  Weighted Avg - Trung bình có trọng số của precision, recall và F1-score, trong đó trọng số là số lượng mẫu của từng lớp.
#       Nó giúp phản ánh độ chính xác của mô hình theo tỷ lệ kích thước của từng lớp.
#  Learning rate – Tốc độ học là một siêu tham số sử dụng trong việc huấn luyện các mạng nơ ron.
#       Giá trị của nó là một số dương, thường nằm trong khoảng giữa 0 và 1.
#       Tốc độ học kiểm soát tốc độ mô hình thay đổi các trọng số để phù hợp với bài toán.
#       Tốc độ học lớn giúp mạng nơ ron được huấn luyện nhanh hơn nhưng cũng có thể làm giảm độ chính xác.
#  Trong Deep Learning, việc phân chia dữ liệu thành 3 tập riêng biệt là cực kỳ quan trọng:
#       Train set: Huấn luyện mô hình.
#       Validation set: Điều chỉnh siêu tham số (learning rate, số lớp, v.v.) và theo dõi overfitting trong quá trình train.
#       Test set: Đánh giá cuối cùng một lần duy nhất sau khi mô hình đã hoàn thiện.