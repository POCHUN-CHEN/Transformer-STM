import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')

# 提取物理量標簽數據並轉換為NumPy數組，存儲為列表
label_groups = []
count = 0
for i in range(1, 11):  # 10大組
    for j in range(1, 6):  # 每大組包含5小組
        labels = excel_data.loc[count, '50HZ']  # 只使用50HZ的標簽
        label_groups.extend([labels] * 200)  # 將一組標簽重復200次
        count += 1

# 轉換為NumPy數組
labels = np.array(label_groups)

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

# 載入圖像數據
image_groups = []

for group in range(1, 11):
    group_images = []
    for image_num in range(1, 6):
        folder_name = f'circle(340x344)/trail{group:01d}_{image_num:02d}'
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

# 轉換為NumPy數組
images = np.array(image_groups)

# 將數據拆分為訓練集和驗證集
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# 創建數據生成器
batch_size = 8

train_data_generator = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
val_data_generator = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

# 定義Convolutional Transformer模型
def complex_convolutional_transformer_model(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    # Convolutional Blocks
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Transformer Blocks
    num_heads = 8
    key_dim = 64
    value_dim = 64
    for _ in range(4):  # 增加Transformer Blocks的數量
        x = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Flatten and Dense Layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)  # 增加Dense層的神經元數量
    x = keras.layers.Dropout(0.5)(x)  # 添加Dropout層
    outputs = keras.layers.Dense(1)(x)

    model = keras.models.Model(inputs, outputs)
    return model

# 創建複雜的Convolutional Transformer模型
input_shape = (image_height, image_width, num_channels)
complex_conv_transformer_model = complex_convolutional_transformer_model(input_shape)

# 編譯模型
optimizer = keras.optimizers.Adam(learning_rate=0.01)  # 設定學習率
complex_conv_transformer_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# 訓練模型
epochs = 500

history = complex_conv_transformer_model.fit(
    train_data_generator,
    validation_data=val_data_generator,
    epochs=epochs
)

# 保存模型權重
complex_conv_transformer_model.save_weights('complex_convolutional_transformer_model_weights.h5')
