import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# # 創建自定義回調函數
# class PredictionCallback(keras.callbacks.Callback):
#     def __init__(self, test_data):
#         self.test_data = test_data  # 傳入驗證集數據

#     def on_epoch_end(self, epoch, logs=None):
#         # 在每個訓練周期結束時進行預測
#         x_test, y_test = self.test_data
#         predictions = self.model.predict(x_test)

#         # 在這裡可以印出預測結果
#         print("Sample predictions: ", predictions[:5])
#         # 打印預測結果
#         print(f"Predictions at end of epoch {epoch}: {predictions}")

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')

# 提取物理量標簽數據並轉換為NumPy數組，存儲為列表
label_groups = []
count = 0
for i in range(1, 11):  # 10大組
    for j in range(1, 6):  # 每大組包含5小組
        labels = excel_data.loc[count, '50HZ']  # 只使用50HZ的標簽
        label_groups.extend([labels] * 200)  # 將一組標簽重復200次
        count+=1
        
# 轉換為NumPy數組
labels = np.array(label_groups)

# 定義圖像的高度、寬度和通道數
image_height = 64  # 減小圖像高度
image_width = 64  # 減小圖像寬度
num_channels = 1

# 創建數據生成器
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

# 載入圖像數據
image_groups = []

for group in range(1, 11):
    group_images = []
    for image_num in range(1, 6):
        folder_name = f'circle(340x344)/trail{group:01d}_{image_num:02d}'  # 修改 "trail" 為 "item"
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

# 創建數據生成器實例
batch_size = 8
train_data_generator = data_generator(x_train, y_train, batch_size)
val_data_generator = data_generator(x_val, y_val, batch_size)

# 定義較小的Transformer模型
def smaller_transformer_model(image_height, image_width, num_channels):
    inputs = keras.Input(shape=(image_height, image_width, num_channels))
    x = layers.Reshape((image_height * image_width, num_channels))(inputs)
    
    # Transformer Encoder
    for _ in range(2):  # 減少Transformer Encoder的層數
        x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)  # 減少頭數和key_dim
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Conv1D(filters=32, kernel_size=1, activation='relu')(x)  # 減小輸出通道數
    
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)  # 減小Dense層的單元數
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs, outputs)
    return model

# 創建較小的Transformer模型
smaller_model = smaller_transformer_model(image_height, image_width, num_channels)

# 編譯模型
smaller_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 訓練模型
steps_per_epoch = len(x_train) // batch_size
validation_steps = len(x_val) // batch_size
epochs = 2000

history = smaller_model.fit(
    train_data_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_data_generator,
    validation_steps=validation_steps
    # callbacks=[PredictionCallback((x_val, y_val))]  # 添加回調函數
)

# 保存模型權重
smaller_model.save_weights('smaller_transformer_model_weights.h5')
