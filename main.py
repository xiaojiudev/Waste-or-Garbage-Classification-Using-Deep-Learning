import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)


keras = tf.keras
layers = tf.keras.layers
models = tf.keras.models
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
EarlyStopping = tf.keras.callbacks.EarlyStopping

# NOTE - Datasets Understanding

os.chdir('D:/Study/CNTT/A.MHUD/CNN_Practice/train')

train_path = os.getcwd()
print(train_path)

# Show the data classes
classes = os.listdir(train_path)
print(classes)

#Loading train datasets
train_data = []
train_labels = []
classes = 7

# Duyệt qua từng thư mục con (mỗi thư mục là một lớp rác thải)
for i in os.listdir(train_path):
    # Tạo đường dẫn đầy đủ đến thư mục của từng lớp
    dir = train_path + '/' + i

    # Duyệt qua từng ảnh trong thư mục của lớp đó
    for j in os.listdir(dir):
        # Tạo đường dẫn đầy đủ đến từng ảnh
        img_path = dir + '/' + j

        # Đọc ảnh bằng OpenCV
        # -1 nghĩa là đọc ảnh nguyên gốc (bao gồm cả kênh alpha nếu có)
        # img = cv2.imread(img_path,-1)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # Chỉ lấy 3 kênh BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Chuyển từ BGR sang RGB

        # Thay đổi kích thước ảnh về 224x224 pixel
        # Phương pháp nội suy INTER_NEAREST (nearest-neighbor interpolation)
        img = cv2.resize(img,(224,224), interpolation = cv2.INTER_NEAREST)

        # Thêm ảnh vào danh sách train_data
        train_data.append(img)

        # Thêm nhãn của ảnh (tên thư mục) vào train_labels
        train_labels.append(i)

plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(img)

# Chuyển danh sách thành mảng numpy
train_data = np.array(train_data)
train_labels = np.array(train_labels)

# In ra số lượng và kích thước dữ liệu
print(train_data.shape, train_labels.shape)

os.chdir('D:/Study/CNTT/A.MHUD/CNN_Practice/test')

test_path = os.getcwd()
print(test_path)

classes = os.listdir(test_path)
print(classes)

#Loading test datasets
test_data = []
test_labels = []
classes = 7 #data belongs to 7 class
for i in os.listdir(test_path):
    dir = test_path + '/' + i
    for j in os.listdir(dir):
        img_path = dir + '/' + j
        img = cv2.imread(img_path,-1)
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_NEAREST)
        test_data.append(img)
        test_labels.append(i)


test_data = np.array(test_data)
test_labels = np.array(test_labels)
print(test_data.shape, test_labels.shape)

# NOTE - Data Augmentation
def create_augmentation_layer():
    return keras.Sequential([
        layers.RandomRotation(0.4),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomZoom((-0.2, 0.2)),
        layers.RandomFlip(mode="horizontal"),
        layers.RandomContrast(0.2),
    ])

def create_dataset(path, augmentation=None, batch_size=32):
    dataset = keras.utils.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical'
    )

    # Áp dụng rescale
    rescale_layer = layers.Rescaling(1. / 255)
    dataset = dataset.map(lambda x, y: (rescale_layer(x), y))

    # Áp dụng augmentation nếu có
    if augmentation:
        dataset = dataset.map(lambda x, y: (augmentation(x, training=True), y))

    return dataset

training_data = create_dataset(
    train_path,
    augmentation=create_augmentation_layer(),
    batch_size=32
)

# validation_data = create_dataset(
#     test_path,
#     augmentation=None,
#     batch_size=32
# )

# NOTE - Data Augmentation Visualization
for images, labels in training_data.take(1):
    print(images.numpy().min(), images.numpy().max())  # Kiểm tra min, max giá trị pixel

# Visualize the augmented images

# Vẽ hình ảnh trước và sau khi augmentation để kiểm tra:
# for images, labels in training_data.take(5):
#     plt.figure(figsize=(10, 5))
#
#     # Ảnh gốc
#     plt.subplot(1, 2, 1)
#     plt.imshow(images[0].numpy())
#     plt.title("Original Image")
#
#     # Ảnh sau augmentation
#     augmented = create_augmentation_layer()(tf.expand_dims(images[0], axis=0))
#     plt.subplot(1, 2, 2)
#     plt.imshow(augmented[0].numpy())
#     plt.title("Augmented Image")
#
#     plt.show()

def plotImages(images_arr):
    """Hàm vẽ một danh sách các ảnh."""
    fig, axes = plt.subplots(1, len(images_arr), figsize=(15, 5))
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

# Lấy một ảnh từ tập dữ liệu
for images, labels in training_data.take(1):
    original_image = images[9]  # Lấy ảnh đầu tiên trong batch

# Tạo 5 ảnh với các biến đổi augmentation khác nhau
augmented_images = [create_augmentation_layer()(tf.expand_dims(original_image, axis=0))[0].numpy() for _ in range(5)]

# Vẽ ảnh sau augmentation
plotImages(augmented_images)


