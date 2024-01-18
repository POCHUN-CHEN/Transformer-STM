import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import keras_tuner as kt

# 定義Convolution Transformer模型建構函數
def build_model(hp):
    image_inputs = keras.Input(shape=(image_height, image_width, num_channels))
    process_inputs = keras.Input(shape=(process_params.shape[1],))

    # Convolutional Blocks
    x = keras.layers.Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=32), 
                            (3, 3), activation='relu')(image_inputs)
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
    x = keras.layers.concatenate([x, process_inputs])  # 將製程參數與其他特徵串聯
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=[image_inputs, process_inputs], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# 設定想要訓練的頻率標籤
N = '50HZ_Bm'  # 可以改成 '50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv'

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')
excel_process = pd.read_excel('Process_parameters.xlsx')

################################################################################
################################## 工件材料性質 ##################################
################################################################################

# 載入材料數據標簽
label_groups = []
count = 0
for i in range(1, 11):
    for j in range(1, 6):  # 每大組包含5小組
        # 添加條件判斷，跳過標籤第5小組
        if j not in [5]:
            labels = excel_data.loc[count, N]  # 使用N指定的頻率標簽
            label_groups.extend([labels] * 200)  # 將一組標簽重復200次
        count += 1

# 轉換為NumPy數組
labels = np.array(label_groups)


#################################################################################
#################################### 製程參數 ####################################
#################################################################################

# 載入製程參數
Process_parameters = ['氧濃度', '雷射掃描速度', '雷射功率', '線間距', '能量密度']
proc_groups = []  # 儲存全部大組製程參數

for i in range(1, 11):
    group_procs = []  # 每大組的製程參數
    parameters_group = []
    for para in Process_parameters:
        parameters = excel_process.loc[i-1, para]
        parameters_group.append(parameters)

    for j in range(1, 5):  # 每大組包含5小組，取4大組
        group_procs.extend([parameters_group] * 200)

    proc_groups.extend(group_procs)

# 轉換為NumPy數組
process_params = np.array(proc_groups)
# process_params = process_params.reshape(-1, 1)  # 調整形狀


#################################################################################
#################################### 積層影像 ####################################
#################################################################################

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

# 載入圖像數據
image_groups = []

for group in range(1, 11):
    group_images = []
    for image_num in range(1, 5):
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


#################################################################################
#################################### 訓練模型 ####################################
#################################################################################

# 設置 epoch 數目
train_epochs = 200

# 設置貝葉斯優化
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mae',
    max_trials=20,
    num_initial_points=2,
    directory='my_dir',
    project_name=f'bayesian_opt_conv_transformer_par_{N}'
)

# 將數據拆分為訓練集和驗證集
x_train, x_val, y_train, y_val, proc_train, proc_val = train_test_split(images, labels, process_params, test_size=0.25, random_state=42)

# 創建數據生成器
batch_size = 8
train_data_generator = tf.data.Dataset.from_tensor_slices(((x_train, proc_train), y_train)).batch(batch_size)
val_data_generator = tf.data.Dataset.from_tensor_slices(((x_val, proc_val), y_val)).batch(batch_size)

# 開始搜索
tuner.search(train_data_generator, epochs=10, validation_data=val_data_generator)

# 獲取最佳超參數
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# 使用最佳超參數創建模型
model = build_model(best_hps)
print(f'Frequency: {N}')
model.fit(train_data_generator, epochs=train_epochs, validation_data=val_data_generator)

# 初始化 DataFrame 以存儲記錄（抓取訓練過程的趨勢資料）
records = pd.DataFrame()

# 獲取整體 epochs 的記錄
current_records = pd.DataFrame(model.history.history)

# 添加 epoch 列
current_records.insert(0, 'epoch', range(1, train_epochs + 1))

# 將當前記錄添加到整體記錄中
records = pd.concat([records, current_records], ignore_index=True)

# 將 DataFrame 寫入 Excel 檔案
records.to_excel(f'Records/bayesian_conv_transformer_par_records_{N}.xlsx', index=False)

# 清除先前的訓練記錄
records = records.iloc[0:0]  # 刪除所有行，得到一個空的 DataFrame

# 保存模型權重
model.save_weights(f'Weight/bayesian_conv_transformer_par_model_weights_{N}.h5')