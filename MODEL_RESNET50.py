import os
import json
import numpy as np
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

keras = tf.keras
layers = tf.keras.layers
Model = tf.keras.models.Model
ResNet50 = tf.keras.applications.ResNet50
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping
TensorBoard = tf.keras.callbacks.TensorBoard
CSVLogger = tf.keras.callbacks.CSVLogger
TerminateOnNaN = tf.keras.callbacks.TerminateOnNaN

print("GPU available:", tf.config.list_physical_devices("GPU"))
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())

# Tối ưu bộ nhớ GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
        )
    except RuntimeError as e:
        print(e)

logical_gpus = tf.config.list_logical_devices("GPU")
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")

# Thiết lập đường dẫn
train_path = "D:/Study/CNTT/A.MHUD/CNN_Practice/my_dataset/train"
test_path = "D:/Study/CNTT/A.MHUD/CNN_Practice/my_dataset/test"

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
    validation_split=0.2, # Tách 20% làm validation
)

# Chỉ áp dụng augmentation cho tập train, tập test CHỈ rescale. Nếu không sẽ gây nhiễu dữ liệu và làm sai lệch kết quả đánh giá.
test_datagen = ImageDataGenerator(
    rescale=1.0/255,
)

# Tạo training dataset từ thư mục
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
    subset="validation",  # Phần dành cho validation
)

# Tạo testing dataset từ thư mục
testing_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False, # NOTE: Đảm bảo thứ tự dữ liệu test không bị xáo trộn, giúp đánh giá metrics chính xác
)

# Lưu class indices vào file json
with open("class_indices.json", "w") as f:
    json.dump(training_data.class_indices, f)

# Kiểm tra tỷ lệ số lượng mẫu của từng lớp:
print("Training class distribution:", Counter(training_data.classes))
print("Validation class distribution:", Counter(validation_data.classes))
print("Testing class distribution:", Counter(testing_data.classes))

# NOTE: Vẽ biểu đồ Phân bố Lớp (Class Distribution)
#  Cho thấy sự cân bằng của dữ liệu. Mất cân bằng lớp có thể ảnh hưởng đến độ chính xác
def plot_class_distribution(counter, title, path):
    plt.figure(figsize=(10, 6))

    # Ánh xạ index thành tên lớp
    labels = [list(training_data.class_indices.keys())[idx] for idx in counter.keys()]
    counts = list(counter.values())

    # Vẽ biểu đồ
    bars = plt.bar(labels, counts)

    # Thêm giá trị lên trên mỗi thanh cột
    for bar in bars:
        height = bar.get_height()
        plt.text(
            x=bar.get_x() + bar.get_width() / 2,  # X position: center of bar
            y=height + 0.02 * max(counts),  # Y position: slightly above bar
            s=f'{int(height)}',  # Display integer value
            ha='center',  # Horizontal alignment
            va='bottom',  # Vertical alignment
            fontsize=10
        )

    # Thiết lập tiêu đề và nhãn
    plt.title(title, fontsize=14)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Điều chỉnh giới hạn trục y để chứa nhãn văn bản
    plt.ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)
    plt.close()

current_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f"./screen_shot/{current_timestamp}"
os.makedirs(save_dir, exist_ok=True) # Tạo thư mục nếu chưa tồn tại

# Vẽ cho cả 3 tập train, validate, test
plot_class_distribution(Counter(training_data.classes),
                       "Training Set Class Distribution",
                       f"./screen_shot/{current_timestamp}/train_dist.png")

plot_class_distribution(Counter(validation_data.classes),
                       "Validation Set Class Distribution",
                       f"./screen_shot/{current_timestamp}/val_dist.png")

plot_class_distribution(Counter(testing_data.classes),
                       "Testing Set Class Distribution",
                       f"./screen_shot/{current_timestamp}/test_dist.png")

# Lấy một batch từ training_data
batch_images, batch_labels = next(training_data)

# Lấy một ảnh từ batch
original_image = batch_images[0]

# NOTE: Hiển thị ảnh gốc trước khi Augmentation
plt.figure(figsize=(3, 3))
plt.imshow(original_image)
plt.title("Ảnh Gốc (Đã Rescale)")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"./screen_shot/{current_timestamp}/original_image.png", format="png", dpi=300)
plt.show()
plt.close()

# NOTE: Hiển thị các ảnh sau khi Augmentation
plt.figure(figsize=(12, 6))
for i in range(6):  # Hiển thị 6 ảnh đã biến đổi
    augmented_image = train_datagen.random_transform(original_image)  # Áp dụng Augmentation lên ảnh gốc
    augmented_image = np.clip(augmented_image, 0, 1)  # Đảm bảo giá trị nằm trong [0, 1]
    plt.subplot(2, 3, i + 1)
    plt.imshow(augmented_image)  # Lấy ảnh đầu tiên trong batch mới
    plt.title(f"Ảnh Augmentation {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(f"./screen_shot/{current_timestamp}/augmented_image.png", format="png", dpi=300)
plt.show()
plt.close()

# NOTE: Load mô hình ResNet50
#  1. Kích thước ảnh đầu vào (224x224 pixels, 3 kênh màu RGB).
#  2. weights="imagenet": Sử dụng trọng số đã được huấn luyện sẵn trên tập ImageNet.
#  3. include_top=False: Không sử dụng phần Fully Connected Layer gốc của ResNet50 (vì ta sẽ thay bằng lớp FC riêng).
BASE_MODEL = ResNet50(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

# NOTE: Fine-tuning - Cho phép huấn luyện lại các lớp cuối của ResNet50
#  1. Là quá trình tiếp tục huấn luyện một mô hình đã được huấn luyện trước bằng cách điều chỉnh một số lớp cụ thể thay vì huấn luyện từ đầu.
#  2. Cách làm này giúp mô hình học thêm các đặc trưng cụ thể của dữ liệu mới mà không làm mất đi kiến thức tổng quát từ ImageNet.
#  3. Nếu ta không fine-tune, mô hình chỉ sử dụng các đặc trưng có sẵn của ResNet50 mà không điều chỉnh cho bài toán cụ thể.
#       layer.trainable = False - trọng số của lớp này sẽ không thay đổi trong quá trình huấn luyện
#       layer.trainable = True - trọng số của lớp này sẽ được huấn luyện và cập nhật trọng số bình thường
#  4. Các lớp sâu trong CNN học các đặc trưng tổng quát (ví dụ: cạnh, hình dạng),
#       trong khi các lớp cuối học đặc trưng cụ thể cho bài toán (ví dụ: hình dạng đối tượng).
#       Fine-tuning giúp tối ưu hóa các đặc trưng này.
for layer in BASE_MODEL.layers[-50:]:  # NOTE: Fine-tune 50 lớp cuối
    layer.trainable = True

# NOTE: Thêm các lớp mới vào mô hình
x = BASE_MODEL.output

# NOTE: Lớp Conv2D, MaxPooling2D, BatchNormalization thêm vào sau ResNet50
#  1. Conv2D (Convolutional Layer)
#       Mục đích: Trích xuất đặc trưng từ đầu ra của ResNet50
#       Tham số:
#           128: Số filters (bộ lọc) để phát hiện 128 đặc trưng khác nhau
#           (3, 3): Kích thước kernel (3x3 pixel)
#           activation="relu": Hàm kích hoạt ReLU để tạo tính phi tuyến
#           padding="same": Giữ nguyên kích thước ảnh sau tích chập
#  2. MaxPooling2D
#       Mục đích: Giảm kích thước không gian của đặc trưng, tập trung vào thông tin quan trọng
#       Cơ chế: Chọn giá trị lớn nhất trong vùng 2x2, giúp giảm kích thước ảnh một nửa (ví dụ: từ 224x224 xuống 112x112)
#  3. BatchNormalization
#       Mục đích: Chuẩn hóa đầu vào về phân phối chuẩn (mean=0, std=1) để tăng tốc độ huấn luyện và ổn định mô hình.
#       Lợi ích: Giảm hiện tượng "internal covariate shift", giúp mô hình hội tụ nhanh hơn.
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)
# x = layers.GlobalAveragePooling2D()(x)

# NOTE: Thêm các layers Fully Connected
#  1. Flatten
#       Mục đích: Chuyển tensor đa chiều (ví dụ: 7x7x128) thành vector 1D (ví dụ: 6272 chiều) để đưa vào lớp FC
#  2. Dense (Fully Connected Layer)
#       Mục đích: Kết hợp các đặc trưng để học mối quan hệ phi tuyến phức tạp
#           512: Số neuron trong lớp
#           activation="relu": Hàm kích hoạt ReLU
#  3. Dropout
#       Mục đích: Giảm overfitting bằng cách tắt ngẫu nhiên 50% neuron trong quá trình huấn luyện,
#           buộc mạng học các đặc trưng tổng quát, tránh overfitting.
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

# NOTE: Lớp đầu ra phân loại (9 lớp)
#       Mục đích: Phân loại ảnh vào 9 lớp
#           activation="softmax": Chuẩn hóa đầu ra thành xác suất (tổng bằng 1)
prediction = layers.Dense(9, activation="softmax")(x)

MODEL_RESNET50 = Model(inputs=BASE_MODEL.input, outputs=prediction)
MODEL_RESNET50.summary()

# NOTE: Callbacks - là các hàm được gọi tự động trong quá trình huấn luyện để kiểm soát quá trình training.
#  Mục đích giúp tối ưu hiệu suất mô hình,tránh overfitting, cải thiện tốc độ hội tụ bằng cách điều chỉnh learning rate.
MODEL_RESNET50.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

# NOTE: ModelCheckpoint - Lưu mô hình tốt nhất dựa trên val_loss
checkpoint = ModelCheckpoint(
    filepath=f"resnet50_saved_{current_timestamp}.keras", # Tên file-định dạng muốn lưu lại
    monitor="val_loss", # Metric theo dõi (val_loss, val_accuracy,...)
    mode="min", # Xác định metric mong muốn là tăng hay giảm ("min" cho loss, "max" cho accuracy)
    save_best_only=True,
    save_weights_only=False, # False = Lưu cả kiến trúc model + weights, True = Chỉ lưu weight
    save_freq="epoch", # lưu model sau mỗi "epoch" hoặc số batch (ví dụ 1000: lưu mỗi 1000 batch)
    verbose=True, # Hiển thị thông báo khi model được lưu
)

# NOTE: EarlyStopping - Dừng huấn luyện sớm nếu val_loss không cải thiện sau 10 epoch, tránh overfitting
early_stop = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10, # Số epoch chờ đợi trước khi dừng
    min_delta=0.001,  # Độ thay đổi tối thiểu để coi là cải thiện
    baseline=None,  # Ngưỡng metric phải đạt (ví dụ baseline=0.9 cho accuracy)
    restore_best_weights=True, # Khôi phục weights tốt nhất khi dừng
)

# NOTE: ReduceLROnPlateau - Nếu val_loss không giảm sau 5 epoch, giảm learning_rate xuống 10% để model học tốt hơn
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.1, # Hệ số giảm learning rate (new_lr = lr * factor)
    patience=3, # Số epoch chờ trước khi giảm lr
    min_lr=1e-7, # Giới hạn dưới của learning rate
    min_delta=0.001, # Ngưỡng thay đổi tối thiểu để coi là "plateau"
    verbose=1, # Hiển thị thông báo khi thay đổi lr
)

# NOTE: TensorBoard - Visualize quá trình training
tensorboard = TensorBoard(
    log_dir="./logs", # Thư mục lưu log
    histogram_freq=1, # Log histograms mỗi epoch
    update_freq="epoch", # Cập nhật log theo: epoch hoặc sau mỗi batch
    write_graph=True, # Hiển thị graph model trong TensorBoard (ảnh hưởng performance)
    write_images=True, # Lưu weights dưới dạng ảnh (tăng kích thước log)
)

# NOTE: CSVLogger - Lưu log training ra file CSV
log_filename = f"./logs/training_log_{current_timestamp}.csv"

csv_logger = CSVLogger(
    filename= log_filename, # Tên file CSV output
    separator=",", # Ký tự phân cách trong file csv (có thể dùng ; hoặc \t)
    append=False, #  Ghi tiếp vào file cũ (True) hay tạo mới (False)
)

# NOTE: TerminateOnNaN - Dừng training nếu loss thành NaN
termination_on = TerminateOnNaN()

# NOTE: Cân bằng lại tập dữ liệu - dùng class_weight trong model.fit()
class_weights = compute_class_weight(
    class_weight="balanced", # "balane" - Tính weights tỷ lệ nghịch với tần suất lớp - None: Không cân bằng
    classes=np.unique(training_data.classes), # Danh sách các class duy nhất
    y=training_data.classes, # Mảng nhãn thực tế
)
class_weights_dict = dict(enumerate(class_weights))

# Huấn luyện mô hình
epochs = 100
history = MODEL_RESNET50.fit(
    training_data, # Data generator cung cấp dữ liệu huấn luyện
    steps_per_epoch=len(training_data), # Số batch mỗi epoch (None = tự động)
    epochs=epochs,
    validation_data=validation_data, # Data generator cung cấp dữ liệu dùng để validate model sau mỗi epoch
    validation_steps=len(validation_data), # Số batch validation (quan trọng khi dùng generator)
    class_weight=class_weights_dict, # Dictionary chứa trọng số của từng lớp, giúp model cân bằng học khi dữ liệu có sự mất cân bằng giữa các lớp
    callbacks=[checkpoint, early_stop, reduce_lr, tensorboard, csv_logger, termination_on],
    verbose= "auto", # Hiển thị log trong quá trình training (0: im lặng, 1: thanh progress bar, 2: hiển thị mỗi epoch)
)

# NOTE: Biểu đồ đánh giá mô hình
#  Hiển thị mối tương quan giữa accuracy và val_accuracy (hình bên trái)
#  Hiển thị mối tương quan giữa loss và val_loss (hình bên phải)
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

# Biểu đồ phụ đầu tiên (bên trái): Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.xlabel("Actual Epochs")
plt.ylabel("Accuracy")

# Biểu đồ phụ đầu tiên (bên phải): Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("Actual Epochs")
plt.ylabel("Loss")

# Điều chỉnh bố cục để tránh chồng chéo
plt.tight_layout()

# Save the combined figure with a reasonable name
plt.savefig(f"./screen_shot/{current_timestamp}/training_metrics.png", format="png", dpi=300)

# Hiển thị hình ảnh
plt.show()
plt.close()

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

# NOTE: Biểu đồ Confusion Matrix Heatmap
#  Đánh giá chi tiết hiệu năng từng lớp, nhận diện lớp nào model hay nhầm lẫn
def plot_confusion_matrix(true, predict, class_names, path):
    cm = confusion_matrix(true, predict)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)
    plt.close()

class_names = list(testing_data.class_indices.keys())
plot_confusion_matrix(y_true, y_pred, class_names,f"./screen_shot/{current_timestamp}/confusion_matrix.png")

# NOTE: Biểu đồ ROC Curve cho Multi-class
#  Đánh giá khả năng phân loại ở các ngưỡng khác nhau, AUC càng gần 1 càng tốt
# Chuẩn bị dữ liệu
y_true_bin = label_binarize(y_true, classes=np.arange(9))
y_pred_bin = label_binarize(y_pred, classes=np.arange(9))

# Tính ROC cho từng lớp
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(9):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Vẽ tất cả ROC curves
plt.figure(figsize=(10, 8))
colors = ["blue", "red", "green", "orange", "purple",
         "brown", "pink", "gray", "olive"]
for i, color in zip(range(9), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("ROC Curves for All Classes", fontsize=16)
plt.legend(loc="lower right", prop={"size": 8})
plt.tight_layout()
plt.savefig(f"./screen_shot/{current_timestamp}/roc_curve.png", format="png", dpi=300)
plt.close()

# NOTE: Biểu đồ Ví dụ Dự đoán Đúng/Sai
#  Trực quan hóa lỗi model, giúp phân tích nguyên nhân (ảnh mờ, góc chụp lạ...)
# Lấy một số mẫu dự đoán sai
incorrect_indices = np.where(y_pred != y_true)[0]
num_samples = min(9, len(incorrect_indices))

plt.figure(figsize=(12, 12))
for i, idx in enumerate(incorrect_indices[:num_samples]):
    img_path = testing_data.filepaths[idx]
    img = plt.imread(img_path)
    plt.subplot(3, 3, i+1)
    plt.imshow(img)
    plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}")
    plt.axis("off")

plt.suptitle("Example of Incorrect Predictions", fontsize=16)
plt.tight_layout()
plt.savefig(f"./screen_shot/{current_timestamp}/wrong_predictions.png", format="png", dpi=300)
plt.close()

# NOTE: Biểu đồ Precision-Recall cho từng lớp
#  Đặc biệt hữu ích khi dữ liệu mất cân bằng, cho thấy trade-off giữa precision và recall
plt.figure(figsize=(10, 8))
for i, color in zip(range(9), colors):
    precision, recall, _ = precision_recall_curve(y_true_bin[:,i], y_pred_bin[:,i])
    plt.plot(recall, precision, color=color, lw=2,
             label=f"Class {class_names[i]}")

plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision-Recall Curves", fontsize=16)
plt.legend(loc="best", prop={"size": 8})
plt.tight_layout()
plt.savefig(f"./screen_shot/{current_timestamp}/precision_recall.png", format="png", dpi=300)
plt.close()


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