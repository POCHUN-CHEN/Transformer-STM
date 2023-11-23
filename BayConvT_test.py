import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from sklearn.metrics import r2_score
import keras_tuner as kt

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

# 載入要進行預測的圖像數據
image_paths = ['Test_images/layer_81_1_3.jpg','Test_images/layer_15_6_1.jpg','Test_images/layer_183_9_5.jpg']

# 實際目標值
actual_targets_dict = {
    '50HZ': [335.17, 306.52, 315.02],  # 填入50HZ對應的目標值
    '200HZ': [330.62, 301.98, 311.4],  # 填入200HZ對應的目標值
    '400HZ': [298.66, 278.43, 288.77], # 填入400HZ對應的目標值
    '800HZ': [215.62, 210.2, 221.59]   # 填入800HZ對應的目標值
}

# 載入和處理影像
images = []
for path in image_paths:
    image = cv2.imread(path)
    image = cv2.resize(image, (image_width, image_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image)
images = np.array(images)

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

    # 重新加載最佳超參數
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 構建模型
    model = build_model(best_hps)

    # 載入模型權重
    model.load_weights(f'Weight/bayesian_conv_transformer_model_weights_{freq}.h5')

    # 進行預測
    predictions = model.predict(images)

    # 計算 R^2 值
    r2 = r2_score(actual_targets_dict[freq], predictions)

    # 列印結果
    print(f'Frequency: {freq}')
    print(f'Predictions: {predictions.flatten()}')
    print(f'Actual: {actual_targets_dict[freq]}')
    print(f'R^2: {r2}\n')

# # 列印模型簡報
# model.summary()