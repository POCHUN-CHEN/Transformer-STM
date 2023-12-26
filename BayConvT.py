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

# 提取不同頻率的標簽數據
frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
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

# 初始化 DataFrame 以存儲記錄
records = pd.DataFrame()

# 設置 epoch 數目
train_epochs = 150

# 對於每個頻率進行模型訓練和保存
for freq in frequencies:
    # 獲取當前頻率的標簽
    current_labels = labels_dict[freq]

    # 拆分數據集
    x_train, x_val, y_train, y_val = train_test_split(images, current_labels, test_size=0.2, random_state=42)

    # 數據生成器
    batch_size = 8
    train_data_generator = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_data_generator = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    # 設置貝葉斯優化
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_mae',
        max_trials=20,
        num_initial_points=2,
        directory='my_dir',
        project_name=f'bayesian_opt_conv_transformer_{freq}'
    )

    # # 定義回調函數以保存最佳模型權重
    # best_model_weights_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'Weight/bayesian_conv_transformer_model_weights_{freq}.h5',
    #     save_best_only=True,
    #     save_weights_only=True,
    #     monitor='val_mae',
    #     mode='min',
    #     verbose=1
    # )

    # 開始搜索
    # tuner.search(train_data_generator, epochs=10, validation_data=val_data_generator,callbacks=[best_model_weights_callback])
    tuner.search(train_data_generator, epochs=10, validation_data=val_data_generator)

    # 獲取最佳超參數並創建模型
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_model(best_hps)

    # 訓練模型
    print(f'Frequency: {freq}')
    model.fit(train_data_generator, epochs=train_epochs, validation_data=val_data_generator)

    # 獲取整體 epochs 的記錄
    current_records = pd.DataFrame(model.history.history)

    # 添加 epoch 列
    current_records.insert(0, 'epoch', range(1, train_epochs + 1))

    # 將當前記錄添加到整體記錄中
    records = pd.concat([records, current_records], ignore_index=True)

    # 將 DataFrame 寫入 Excel 檔案
    records.to_excel(f'Records/bayesian_conv_transformer_records_{freq}.xlsx', index=False)

    # 清除先前的訓練記錄
    records = records.iloc[0:0]  # 刪除所有行，得到一個空的 DataFrame

    # 保存模型權重
    model.save_weights(f'Weight/bayesian_conv_transformer_model_weights_{freq}.h5')