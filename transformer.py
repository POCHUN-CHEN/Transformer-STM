import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# # 创建自定义回调函数
# class PredictionCallback(keras.callbacks.Callback):
#     def __init__(self, test_data):
#         self.test_data = test_data  # 传入验证集数据

#     def on_epoch_end(self, epoch, logs=None):
#         # 在每个训练周期结束时进行预测
#         x_test, y_test = self.test_data
#         predictions = self.model.predict(x_test)

#         # 在這裡可以印出預測結果
#         print("Sample predictions: ", predictions[:5])
#         # 打印预测结果
#         print(f"Predictions at end of epoch {epoch}: {predictions}")

# 读取Excel文件中的标签数据
excel_data = pd.read_excel('Circle_test.xlsx')

# 提取物理量标签数据并转换为NumPy数组，存储为列表
label_groups = []
count = 0
for i in range(1, 11):  # 10大组
    for j in range(1, 6):  # 每大组包含5小组
        labels = excel_data.loc[count, '50HZ']  # 只使用50HZ的标签
        label_groups.extend([labels] * 200)  # 将一组标签重复200次
        count+=1

# 转换为NumPy数组
labels = np.array(label_groups)

# 定义图像的高度、宽度和通道数
image_height = 64  # 减小图像高度
image_width = 64  # 减小图像宽度
num_channels = 1

# 创建数据生成器
def data_generator(images, labels, batch_size):
    num_samples = len(images)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_images = images[batch_indices]
            batch_labels = labels[batch_indices]
            yield batch_images, batch_labels

# 载入图像数据
image_groups = []

for group in range(1, 11):
    group_images = []
    for image_num in range(1, 6):
        folder_name = f'circle(340x344)/trail{group:01d}_{image_num:02d}'  # 修改 "trail" 为 "item"
        folder_path = f'data/{folder_name}/'

        image_group = []
        for i in range(200):
            filename = f'{folder_path}/layer_{i + 1:02d}.jpg'
            image = cv2.imread(filename)
            image = cv2.resize(image, (image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_group.append(image)

        group_images.extend(image_group)

    image_groups.extend(group_images)

# 转换为NumPy数组
images = np.array(image_groups)

# 将数据拆分为训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 创建数据生成器实例
batch_size = 8
train_data_generator = data_generator(x_train, y_train, batch_size)
val_data_generator = data_generator(x_val, y_val, batch_size)

# 定义较小的Transformer模型
def smaller_transformer_model(image_height, image_width, num_channels):
    inputs = keras.Input(shape=(image_height, image_width, num_channels))
    x = layers.Reshape((image_height * image_width, num_channels))(inputs)
    
    # Transformer Encoder
    for _ in range(2):  # 减少Transformer Encoder的层数
        x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)  # 减少头数和key_dim
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv1D(filters=32, kernel_size=1, activation='relu')(x)  # 减小输出通道数
    
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)  # 减小Dense层的单元数
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    return model

# 创建较小的Transformer模型
smaller_model = smaller_transformer_model(image_height, image_width, num_channels)

# 编译模型
smaller_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 训练模型
steps_per_epoch = len(x_train) // batch_size
validation_steps = len(x_val) // batch_size
epochs = 400

history = smaller_model.fit(
    train_data_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_generator,
    validation_steps=validation_steps
    # callbacks=[PredictionCallback((x_val, y_val))]  # 添加回调函数
)

# 保存模型权重
smaller_model.save_weights('smaller_transformer_model_weights.h5')
