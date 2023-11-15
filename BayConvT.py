import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import keras_tuner as kt

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

# 定義Convolution Transformer模型建構函數
def build_model(hp):
    inputs = keras.Input(shape=(image_height, image_width, num_channels))

    # Convolutional Blocks
    x = keras.layers.Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=32), 
                            (3, 3), activation='relu')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(hp.Int('conv_2_filter', min_value=64, max_value=256, step=64), 
                            (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(hp.Int('conv_3_filter', min_value=128, max_value=512, step=128), 
                            (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    # Transformer Blocks
    for _ in range(hp.Int('num_transformer_blocks', 1, 5)):
        num_heads = hp.Choice('num_heads', values=[4, 8, 16])
        key_dim = hp.Int('key_dim', min_value=32, max_value=128, step=32)
        x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Flatten and Dense Layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu')(x)
    x = keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1))(x)
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# 設置貝葉斯優化
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mae',
    max_trials=10,
    num_initial_points=2,
    directory='my_dir',
    project_name='bayesian_opt_conv_transformer'
)

# 開始搜索
tuner.search(train_data_generator, epochs=10, validation_data=val_data_generator)

# 獲取最佳超參數
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# 使用最佳超參數創建模型
model = build_model(best_hps)
model.fit(train_data_generator, epochs=150, validation_data=val_data_generator)

# 保存模型權重
model.save_weights('bayesian_conv_transformer_model_weights.h5')