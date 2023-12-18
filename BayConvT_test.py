import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import r2_score
import keras_tuner as kt
import matplotlib.pyplot as plt

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

# 定義的Convolution Transformer模型
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

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')

# 提取不同頻率的標簽數據
frequencies = ['50HZ', '200HZ', '400HZ', '800HZ']
labels_dict = {}
for freq in frequencies:
    label_groups = []
    count = 0
    for i in range(1, 11):  # 10大組
        for j in range(1, 6):  # 每大組包含5小組
            labels = excel_data.loc[count, freq]
            label_groups.extend([labels] * 200)
            count += 1
    labels_dict[freq] = np.array(label_groups)

# 對於每個頻率進行模型載入、預測和評估
for freq in ['50HZ', '200HZ', '400HZ', '800HZ']:
    # 創建與原始搜索相同配置的Keras Tuner實例
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_mae',
        max_trials=10,
        num_initial_points=2,
        directory='my_dir',
        project_name=f'bayesian_opt_conv_transformer_{freq}'
    )
    
    # 獲取當前頻率的標簽
    current_labels = labels_dict[freq]

    # 重新加載最佳超參數
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 構建模型
    model = build_model(best_hps)

    # 載入模型權重
    model.load_weights(f'Weight/bayesian_conv_transformer_model_weights_{freq}.h5')

    # 進行預測
    predictions = model.predict(images)

    # 計算 R^2 值
    r2 = r2_score(current_labels, predictions)

    # 列印結果
    print(f'Frequency: {freq}')
    print(f'Predictions: {predictions.flatten()}')
    print(f'Actual: {current_labels}')
    print(f'R^2: {r2}\n')

    # # 將預測值和實際值繪製成點圖（R^2圖）
    # plt.scatter(current_labels, predictions.flatten())
    # plt.title('Predictions vs Actual')
    # plt.xlabel('Actual Values')
    # plt.ylabel('Predicted Values')
    # plt.show()

    # 預測值與實際值分成兩條線顯示
    # 生成圖片編號
    image_numbers = np.arange(1, len(predictions) + 1)

    # 繪製實際值和預測值的兩條線
    plt.plot(image_numbers, current_labels, label='Actual', marker='o')
    plt.plot(image_numbers, predictions.flatten(), label='Predicted', marker='x')

    # 添加標籤和標題
    plt.xlabel('Image Number')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted')
    plt.legend()  # 顯示圖例
    plt.show()

# 列印模型簡報
model.summary()
