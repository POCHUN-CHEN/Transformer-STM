import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from sklearn.metrics import r2_score

# 定義圖像的高度、寬度和通道數
image_height = 64
image_width = 64
num_channels = 1

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
model = smaller_transformer_model(image_height, image_width, num_channels)


# 載入模型權重
model.load_weights('smaller_transformer_model_weights.h5')

# 載入要進行預測的圖像數據
image_paths = ['Test_images/layer_63_1_1.jpg','Test_images/layer_70_2_1.jpg','Test_images/layer_57_5_3.jpg']
images = []

for path in image_paths:
    image = cv2.imread(path)
    image = cv2.resize(image, (image_width, image_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)

images = np.array(images)

# 進行預測
predictions = model.predict(images)

# 實際目標值（如果有的話，這裡假設你有實際目標值）
# actual_targets = [335.82,332.34,314.1]
actual_targets = [335.82,335.82,335.82]

# 計算 R^2 值
r2 = r2_score(actual_targets, predictions)

# 打印 R^2 值
print(f'R^2: {r2}')

# 打印預測值
print(f'Predictions: {predictions}')

model.summary()