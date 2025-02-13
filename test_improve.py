import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers

# Kiểm tra GPU
print("GPU available:", tf.config.list_physical_devices('GPU'))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
    )

# Đường dẫn dữ liệu
train_path = 'D:/Study/CNTT/A.MHUD/CNN_Practice/train'
test_path = 'D:/Study/CNTT/A.MHUD/CNN_Practice/test'

# 🔥 Cải thiện Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,       # Tăng độ xoay lên 30 độ
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],  # Điều chỉnh độ sáng
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load dataset
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# 🔥 Load ResNet50 (Giai đoạn 1: Freeze toàn bộ)
base_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
base_model.trainable = False  # Ban đầu đóng băng toàn bộ mạng ResNet50

# Xây dựng mô hình
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)  # 🔥 Thêm BatchNorm
x = layers.Dropout(0.6)(x)  # 🔥 Tăng Dropout lên 0.6
x = layers.Dense(7, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=x)

# 🔥 Sử dụng Cosine Decay Learning Rate
lr_schedule = optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=10000, alpha=1e-6)
optimizer = optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint('resnet50_best.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 🔥 Huấn luyện giai đoạn 1 (Chỉ train FC Layer)
history = model.fit(
    train_generator, epochs=15, validation_data=test_generator, callbacks=[checkpoint, early_stop]
)

# 🔥 Giai đoạn 2: Fine-tune 50 lớp cuối
base_model.trainable = True
for layer in base_model.layers[:-50]:  # 🔥 Chỉ mở khóa 50 lớp cuối
    layer.trainable = False

# Compile lại với learning rate thấp hơn
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# 🔥 Huấn luyện lại với Fine-Tuning
history_finetune = model.fit(
    train_generator, epochs=30, validation_data=test_generator, callbacks=[checkpoint, early_stop]
)

# 🔥 Đánh giá mô hình
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_finetune.history['accuracy'], label='Fine-Tuned Train Accuracy')
plt.plot(history_finetune.history['val_accuracy'], label='Fine-Tuned Val Accuracy')
plt.legend()
plt.show()

# Kiểm tra trên tập test
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc:.4f}')

# Dự đoán và đánh giá
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))
print(confusion_matrix(y_true, y_pred))
