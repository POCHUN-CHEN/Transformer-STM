import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
import keras_tuner as kt
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # 標準化製程參數
# from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 提取不同頻率
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['50HZ_μa', '200HZ_μa', '400HZ_μa', '800HZ_μa']
# frequencies = ['50HZ_μa']

frequencies = ['50HZ_μa', '400HZ_μa', '800HZ_μa']

# 定義範圍
group_start = 1
group_end = 40
piece_num_start = 1
piece_num_end = 5

# 定義其他相關範圍或常數
image_layers = 200  # 每顆影像的層數

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

proc_features_dim = 5  # 假設制程參數維度為5
num_classes = 1  # 回歸任務

# # 設置貝葉斯優化 epoch 數目
# max_trials=20

k_fold_splits = 1

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')
excel_process = pd.read_excel('Process_parameters.xlsx')


################################################################################
##################################### 定義 #####################################
################################################################################

# # 定義Convolution Transformer模型建構函數
# def build_model(hp):
#     image_inputs = keras.Input(shape=(image_height, image_width, num_channels), dtype='float32') # 更改數值精度
#     process_inputs = keras.Input(shape=(proc_dict[freq].shape[1],), dtype='float32')

#     # Convolutional Blocks
#     x = keras.layers.Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=32), 
#                             (3, 3), activation='relu', kernel_regularizer=l2(0.00))(image_inputs) # 添加 L2 正則化
#     x = keras.layers.MaxPooling2D((2, 2))(x)
#     x = keras.layers.Conv2D(hp.Int('conv_2_filter', min_value=64, max_value=256, step=64), 
#                             (3, 3), activation='relu')(x)
#     x = keras.layers.MaxPooling2D((2, 2))(x)
#     x = keras.layers.Conv2D(hp.Int('conv_3_filter', min_value=128, max_value=512, step=128), 
#                             (3, 3), activation='relu')(x)
#     x = keras.layers.MaxPooling2D((2, 2))(x)

#     # Transformer Blocks
#     for _ in range(hp.Int('num_transformer_blocks', 1, 5)):
#         num_heads = hp.Choice('num_heads', values=[4, 8, 16])
#         key_dim = hp.Int('key_dim', min_value=32, max_value=128, step=32)
#         x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
#         x = layers.LayerNormalization(epsilon=1e-6)(x)

#     # Flatten and Dense Layers
#     x = keras.layers.Flatten()(x)
#     x = keras.layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu', kernel_regularizer=l2(0.00))(x) # 添加 L2 正則化
#     x = keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1))(x)
#     x = keras.layers.concatenate([x, process_inputs])  # 將製程參數與其他特徵串聯
#     outputs = keras.layers.Dense(1)(x)

#     model = keras.Model(inputs=[image_inputs, process_inputs], outputs=outputs)
#     model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-3, 1e-2, sampling='log')),
#                   loss='mean_squared_error', metrics=['mae'])
#     return model

# 定義Convolution Block
def ConvBlock(x, filters, kernel_size, strides):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu')(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# 定義Transformer Block
def TransformerBlock(x, embed_dim, num_heads, ff_dim):
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dense(embed_dim)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

# 建立CvT模型的函數
def build_cvt_model(image_height, image_width, num_channels, proc_features_dim, num_classes):
    inputs_img = keras.Input(shape=(image_height, image_width, num_channels), dtype='float32') # 更改數值精度
    inputs_process = keras.Input(shape=(proc_dict[freq].shape[1],), dtype='float32')

    # 根據PyTorch示例直接指定的超參數
    # 假設有三個階段，每個階段的參數如下
    stages_filters = [64, 128, 256]  # 每個階段的filters數量
    stages_kernel_sizes = [7, 3, 3]  # 每個階段的kernel size
    stages_strides = [4, 2, 2]  # 每個階段的strides
    embed_dims = [64, 128, 256]  # 每個階段的embed_dim
    num_heads = [1, 2, 4]  # 每個階段的num_heads
    ff_dims = [128, 256, 512]  # 每個階段的ff_dim

    x = inputs_img
    for stage in range(3):  # 有三個階段
        x = ConvBlock(x, stages_filters[stage], stages_kernel_sizes[stage], stages_strides[stage])
        x = TransformerBlock(x, embed_dims[stage], num_heads[stage], ff_dims[stage])
    
    # 全局平均池化層
    x = layers.GlobalAveragePooling2D()(x)

    # 處理制程參數路徑
    y = layers.Dense(64, activation="relu")(inputs_process)  # 假設制程參數的Dense層為64個單元

    # 合並圖像和制程參數的路徑
    combined = layers.concatenate([x, y])

    # 回歸輸出層
    outputs = layers.Dense(num_classes, activation="linear")(combined)

    # 建立和編譯模型
    model = keras.Model(inputs=[inputs_img, inputs_process], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # 根據需要調整學習率
                  loss='mean_squared_error', 
                  metrics=['mae'])

    return model


################################################################################
################################## 工件材料性質 ##################################
################################################################################

# 載入材料數據標簽
labels_dict = {}
valid_dict = {}

# 假設每個大組有5個小組，並且每小組對應一行Excel數據
start_index = (group_start - 1) * (piece_num_end - piece_num_start + 1)
end_index = group_end * ((piece_num_end - piece_num_start + 1))

for freq in frequencies:
    valid_indices = []  # 用於記錄有效圖像的索引
    label_groups = []
    count = 0
    for i in range(1, group_end + 1):
        for j in range(piece_num_start, piece_num_end + 1):  # 每大組包含5小組
            labels = excel_data.loc[count, str(freq)]
            # 使用 pd.isnull() 檢查 labels 是否為 NaN
            if not pd.isnull(labels):  # 如果 labels 不為 NaN，則添加到 label_groups
                if start_index <= count < end_index:
                    label_groups.extend([labels] * image_layers)
                valid_indices.append(count)  # 添加有效圖像的索引
            count += 1
        
    labels_dict[freq] = np.array(label_groups)  # 轉換為NumPy數組

    # 過濾 valid_indices，只保留新範圍內的索引
    valid_indices = [index for index in valid_indices if start_index <= index < end_index]

    valid_dict[freq] = np.array(valid_indices)  # 轉換為NumPy數組


#################################################################################
#################################### 製程參數 ####################################
#################################################################################

# 載入製程參數
Process_parameters = ['氧濃度', '雷射掃描速度', '雷射功率', '線間距', '能量密度']
proc_dict = {}  # 儲存所有頻率全部大組製程參數
proc_dict_scaled = {}

for freq in frequencies:
    valid_proc_groups = []  # 儲存有效的製程參數組
    for index in valid_dict[freq]:  # 只遍歷有效的索引
        group_procs = []  # 每顆的製程參數們
        parameters_group = []
        group_index = index // (piece_num_end - piece_num_start + 1)

        for para in Process_parameters:
            parameters = excel_process.loc[group_index, para]
            parameters_group.append(parameters)
        
        group_procs.extend([parameters_group] * image_layers)

        # 每個有效索引都對應到一組製程參數
        valid_proc_groups.extend(group_procs)

    # 轉換為NumPy數組
    proc_dict[freq] = np.array(valid_proc_groups)
    
    # 初始化 StandardScaler
    scaler = StandardScaler()
    
    # 標準化製程參數
    proc_dict_scaled[freq] = scaler.fit_transform(proc_dict[freq])


#################################################################################
#################################### 積層影像 ####################################
#################################################################################

valid_images = []  # 用於儲存有效的圖像

for index in valid_dict[freq]:  # 只遍歷有效的索引
    group_index = index // (piece_num_end - piece_num_start + 1) + 1
    image_num = index % (piece_num_end - piece_num_start + 1) + 1

    folder_name = f'circle(340x345)/trail{group_index:01d}_{image_num:02d}'
    folder_path = f'data/{folder_name}/'
    
    for i in range(image_layers):
        filename = f'{folder_path}/layer_{i + 1:02d}.jpg'
        image = cv2.imread(filename)
        image = cv2.resize(image, (image_width, image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image / 255.0  # 歸一化
        valid_images.append(image)

# 轉換為NumPy數組
images = np.array(valid_images)


#################################################################################
#################################### 測試模型 ####################################
#################################################################################

# 對於每個頻率進行模型載入、預測和評估
for freq in frequencies:
    print(f"Testing for frequency {freq}")
    # 定義訓練集和驗證集
    x_val, y_val, proc_val = [], [], []

    # 初始化每組的第一個有效索引列表
    first_valid_indices_per_group = []

    # 按原始數據的分組規則遍歷
    for d in range(0, count, 5):  # 原始數據，每五個一組
        for j in range(d, d + 5):
            if j in valid_dict[freq]:
                first_valid_indices_per_group.append(j)
                break  # 找到每組的第一個有效索引後，跳出內層循环，繼續下一組

    # 遍歷每個有效的索引而不是固定的範圍
    for i in range(len(valid_dict[freq])):
        index = i * image_layers

        # 決定是否將當前索引用作驗證集
        if valid_dict[freq][i] in first_valid_indices_per_group :
            # 驗證集
            x_val.extend(images[index:index + image_layers])
            y_val.extend(labels_dict[freq][index:index + image_layers])
            proc_val.extend(proc_dict_scaled[freq][index:index + image_layers])

    # 轉換為 NumPy 數組
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    proc_val = np.array(proc_val)

    # # 設置貝葉斯優化
    # tuner = kt.BayesianOptimization(
    #     build_cvt_model,
    #     objective='val_mae',
    #     max_trials=max_trials,
    #     num_initial_points=2,
    #     directory='my_dir/Images & Parameters/',
    #     project_name=f'bayesian_opt_conv_transformer_par_{freq}'
    # )

    # 重新加載最佳超參數
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 構建模型
    # model = build_cvt_model(best_hps)
    model = build_cvt_model(image_height, image_width, num_channels, proc_features_dim, num_classes)

    # 載入模型權重
    model.load_weights(f'Weight/Images & Parameters/bayesian_conv_transformer_par_model_weights_{freq}.h5')

    # 進行預測
    predictions = model.predict([x_val, proc_val])

    # 計算 R^2 值
    r2 = r2_score(y_val, predictions.flatten())
    mse = mean_squared_error(y_val, predictions.flatten())
    mae = mean_absolute_error(y_val, predictions.flatten())

    # 列印結果
    print(f'Frequency: {freq}')
    print(f'Predictions: {predictions.flatten()}')
    print(f'Actual: {y_val}')
    print(f'R^2: {r2:.3f}')
    print(f'MSE: {mse:.3f}')
    print(f'MAE: {mae:.3f}\n')

    # 將預測值和實際值繪製成點圖（R^2圖）
    plt.scatter(y_val, predictions.flatten())
    plt.title(f'R^2 - {freq}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    # plt.text(0.0, 0.0,f'R^2: {r2:.3f}\nMSE: {mse:.3f}\nMAE: {mae:.3f}', 
    #          ha='left', va='center', transform=plt.gca().transAxes)
    plt.savefig(f'Plots/Images & Parameters/R^2_par_{freq}.png')  # 儲存圖片
    plt.show()

    # 預測值與實際值分成兩條線顯示(實際與預測畫在一起)
    # 生成圖片編號
    image_numbers = np.arange(1, len(predictions) + 1)

    # 繪製實際值和預測值的兩條線
    plt.plot(image_numbers, y_val, label='Actual', marker='o')
    plt.plot(image_numbers, predictions.flatten(), label='Predicted', marker='x')

    # 添加標籤和標題
    plt.xlabel('Image Number')
    plt.ylabel('Values')
    plt.title(f'Actual vs Predicted - {freq}')
    plt.legend()  # 顯示圖例
    plt.savefig(f'Plots/Images & Parameters/Actual_vs_Predicted_par_{freq}.png')  # 儲存圖片
    plt.show()

# 列印模型簡報
# tuner.results_summary() #查看 BayesianOptimization 最佳解（Val_mae 分數越低越好）
model.summary()
