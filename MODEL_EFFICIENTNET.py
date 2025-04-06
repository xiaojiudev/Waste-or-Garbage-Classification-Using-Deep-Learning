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
EfficientNetB0 = tf.keras.applications.EfficientNetB0
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
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input, # Chuyển dải 0-255 -> [-1, 1] qua công thức (x/127.5) - 1.0
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

# Chỉ áp dụng augmentation cho tập train, tập test CHỈ rescale. Nếu không sẽ gây nhiễu dữ liệu và làm sai lệch kết quả đánh giá.
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet_v2.preprocess_input,
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
    subset="validation",
)

# Tạo testing dataset từ thư mục
testing_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

# Lưu class indices vào file json
with open("efficientnet_class_indices.json", "w") as f:
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
            s=f"{int(height)}",  # Display integer value
            ha="center",  # Horizontal alignment
            va="bottom",  # Vertical alignment
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

current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"./screen_shot/efficientnet/{current_timestamp}"
os.makedirs(save_dir, exist_ok=True) # Tạo thư mục nếu chưa tồn tại

# Vẽ cho cả 3 tập train, validate, test
plot_class_distribution(Counter(training_data.classes),
                       "Training Set Class Distribution",
                       f"{save_dir}/train_dist.png")

plot_class_distribution(Counter(validation_data.classes),
                       "Validation Set Class Distribution",
                       f"{save_dir}/val_dist.png")

plot_class_distribution(Counter(testing_data.classes),
                       "Testing Set Class Distribution",
                       f"{save_dir}/test_dist.png")

# Lấy một batch từ training_data
batch_images, batch_labels = next(training_data)

# Lấy một ảnh từ batch
original_image = batch_images[0]

# NOTE: Hiển thị ảnh gốc trước khi Augmentation
plt.figure(figsize=(3, 3))
plt.imshow(original_image)
plt.title("Original Image (Rescaled)")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{save_dir}/original_image.png", format="png", dpi=300)
plt.show()
plt.close()

# NOTE: Hiển thị các ảnh sau khi Augmentation
plt.figure(figsize=(12, 6))
for i in range(6):  # Hiển thị 6 ảnh đã biến đổi
    augmented_image = train_datagen.random_transform(original_image)  # Áp dụng Augmentation lên ảnh gốc
    augmented_image = np.clip(augmented_image, 0, 1)  # Đảm bảo giá trị nằm trong [0, 1]
    plt.subplot(2, 3, i + 1)
    plt.imshow(augmented_image)  # Lấy ảnh đầu tiên trong batch mới
    plt.title(f"Image Augmentation {i+1}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(f"{save_dir}/augmented_image.png", format="png", dpi=300)
plt.show()
plt.close()

# NOTE: Load mô hình EfficientNet
BASE_MODEL = EfficientNetB0(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

# NOTE: Fine-tuning - Cho phép huấn luyện lại một phần cuối của EfficientNet
#  Tính số lớp cần fine-tune (20% lớp cuối)
total_layers = len(BASE_MODEL.layers)
trainable_layers = int(total_layers * 0.2)
print(f"Total layers: {total_layers}, Fine-tuning last {trainable_layers} layers")

for layer in BASE_MODEL.layers[-trainable_layers:]:  # NOTE: Fine-tune 20% lớp cuối = 47
    layer.trainable = True

# NOTE: Thêm các lớp mới vào mô hình
x = BASE_MODEL.output

x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.BatchNormalization()(x)

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
prediction = layers.Dense(9, activation="softmax")(x)

MODEL_EFFICIENTNET = Model(inputs=BASE_MODEL.input, outputs=prediction)
MODEL_EFFICIENTNET.summary()

MODEL_EFFICIENTNET.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

# NOTE: ModelCheckpoint - Lưu mô hình tốt nhất dựa trên val_loss
checkpoint = ModelCheckpoint(
    filepath=f"efficientnet_saved_{current_timestamp}.keras", # Tên file-định dạng muốn lưu lại
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
    log_dir="./logs/efficientnet", # Thư mục lưu log
    histogram_freq=1, # Log histograms mỗi epoch
    update_freq="epoch", # Cập nhật log theo: epoch hoặc sau mỗi batch
    write_graph=True, # Hiển thị graph model trong TensorBoard (ảnh hưởng performance)
    write_images=True, # Lưu weights dưới dạng ảnh (tăng kích thước log)
)

# NOTE: CSVLogger - Lưu log training ra file CSV
log_filename = f"./logs/efficientnet/training_log_{current_timestamp}.csv"

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
class_weights_dict = {i: float(weight) for i, weight in enumerate(class_weights)}

# Huấn luyện mô hình
epochs = 100
history = MODEL_EFFICIENTNET.fit(
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

# Tìm index của epoch có val_loss nhỏ nhất
best_epoch = np.argmin(val_loss)

# Lấy giá trị cuối cùng từ history
best_train_acc = acc[best_epoch] * 100
best_val_acc = val_acc[best_epoch] * 100
best_train_loss = loss[best_epoch]
best_val_loss = val_loss[best_epoch]

# Tạo text hiển thị các giá trị
text_acc = f"Train: {best_train_acc:.2f}%\nVal: {best_val_acc:.2f}%"
text_loss = f"Train: {best_train_loss:.4f}\nVal: {best_val_loss:.4f}"

epochs_range = range(len(acc))

plt.figure(figsize=(12, 8))

# Biểu đồ phụ đầu tiên (bên trái): Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.scatter(best_epoch, acc[best_epoch], marker="o", s=100, color="blue", zorder=5,
           edgecolors="black", linewidths=1, label=f"Best Epoch ({best_epoch+1})")
plt.scatter(best_epoch, val_acc[best_epoch], marker="o", s=100, color="orange", zorder=5,
           edgecolors="black", linewidths=1)
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.text(0.98, 0.5, text_acc, transform=plt.gca().transAxes,
         ha="right", va="center", fontsize=10,
         bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

# Biểu đồ phụ đầu tiên (bên phải): Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.scatter(best_epoch, loss[best_epoch], marker="o", s=100, color="blue", zorder=5,
           edgecolors="black", linewidths=1, label=f"Best Epoch ({best_epoch+1})")
plt.scatter(best_epoch, val_loss[best_epoch], marker="o", s=100, color="orange", zorder=5,
           edgecolors="black", linewidths=1)
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.text(0.98, 0.5, text_loss, transform=plt.gca().transAxes,
         ha="right", va="center", fontsize=10,
         bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"))

# Điều chỉnh bố cục để tránh chồng chéo
plt.tight_layout(pad=3)
# plt.subplots_adjust(bottom=0.25)  # Tăng khoảng trống phía dưới lên 25%

# Thêm subtitle note
note_text = f"NOTE: Early stopping mechanism was applied, causing the training to end at {len(acc)} epochs rather than continuing to {epochs}."
plt.figtext(
    0.5, 0.01,
    note_text,
    ha="center",
    va="bottom",
    fontsize=10,
    style="italic",
    bbox=dict(facecolor="white", alpha=0.8),
    wrap=True
)

# Save the combined figure with a reasonable name
plt.savefig(f"{save_dir}/training_metrics.png", format="png", dpi=300)

# Hiển thị hình ảnh
plt.show()
plt.close()

# Đánh giá trên tập test
test_loss, test_acc = MODEL_EFFICIENTNET.evaluate(testing_data, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Dự đoán trên tập test
y_pred = MODEL_EFFICIENTNET.predict(testing_data)
y_pred = np.argmax(y_pred, axis=1)
y_true = testing_data.classes

# NOTE: In ra classification report
#  zero_division=1 tránh lỗi chia cho 0 khi một lớp không có mẫu nào được dự đoán
report = classification_report(y_true, y_pred, target_names=testing_data.class_indices.keys(), zero_division=1)
print(f"Classification Report:\n{report}")

# In ra confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix: {cm}")

def save_log_as_image(log_text, path):
    plt.figure(figsize=(12, 8))
    plt.axis("off")  # Tắt trục
    plt.text(0.05, 0.95, log_text,
             fontfamily="monospace",  # Font monospace để căn đều cột
             fontsize=10,
             verticalalignment="top") # Căn lề trên
    plt.tight_layout()
    plt.savefig(path, format="png", dpi=300, bbox_inches="tight")
    plt.close()

# Tạo chuỗi log và lưu thành ảnh
log_content = (
    f"Model: EfficientNet\n\n"
    f"Test Accuracy: {test_acc:.4f}\n\n"
    f"Classification Report:\n{report}\n\n"
    f"Confusion Matrix:\n{cm}"
)
save_log_as_image(
    log_content,
    f"{save_dir}/evaluation_log.png"
)

# NOTE: Biểu đồ Confusion Matrix Heatmap
#  Đánh giá chi tiết hiệu năng từng lớp, nhận diện lớp nào model hay nhầm lẫn
def plot_confusion_matrix(true, predict, class_names, path):
    cm = confusion_matrix(true, predict)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.title("EfficientNet: Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)
    plt.close()

class_names = list(testing_data.class_indices.keys())
plot_confusion_matrix(y_true, y_pred, class_names,f"{save_dir}/confusion_matrix.png")

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
plt.title("EfficientNet: ROC Curves for All Classes", fontsize=16)
plt.legend(loc="lower right", prop={"size": 8})
plt.tight_layout()
plt.savefig(f"{save_dir}/roc_curve.png", format="png", dpi=300)
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

plt.suptitle("EfficientNet: Example of Incorrect Predictions", fontsize=16)
plt.tight_layout()
plt.savefig(f"{save_dir}/wrong_predictions.png", format="png", dpi=300)
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
plt.title("EfficientNet: Precision-Recall Curves", fontsize=16)
plt.legend(loc="best", prop={"size": 8})
plt.tight_layout()
plt.savefig(f"{save_dir}/precision_recall.png", format="png", dpi=300)
plt.close()