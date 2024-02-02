import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
import keras_tuner as kt

from sklearn.preprocessing import StandardScaler # 標準化製程參數
# from sklearn.preprocessing import MinMaxScaler # 歸一化製程參數
# from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping

# from sklearn.model_selection import train_test_split

# 提取不同頻率
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['50HZ_μa', '200HZ_μa', '400HZ_μa', '800HZ_μa']
frequencies = ['50HZ_μa']

# 定義範圍
group_start = 1
group_end = 10
piece_num_start = 1
piece_num_end = 5

# 定義其他相關範圍或常數
image_layers = 200  # 每顆影像的層數

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

# 批次大小
batch_size = 8

# 設置 epoch 數目
train_epochs = 500

# 設置貝葉斯優化 epoch 數目
max_trials=20
trials_epochs=10

k_fold_splits = 1

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')
excel_process = pd.read_excel('Process_parameters.xlsx')


################################################################################
##################################### 定義 #####################################
################################################################################

# 定義Convolution Transformer模型建構函數
def build_model(hp):
    image_inputs = keras.Input(shape=(image_height, image_width, num_channels), dtype='float32') # 更改數值精度
    process_inputs = keras.Input(shape=(proc_dict[freq].shape[1],), dtype='float32')

    # Convolutional Blocks
    x = keras.layers.Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=32), 
                            (3, 3), activation='relu', kernel_regularizer=l2(0.00))(image_inputs) # 添加 L2 正則化
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
    x = keras.layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu', kernel_regularizer=l2(0.00))(x) # 添加 L2 正則化
    x = keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1))(x)
    x = keras.layers.concatenate([x, process_inputs])  # 將製程參數與其他特徵串聯
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=[image_inputs, process_inputs], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-3, 1e-2, sampling='log')),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# # 定義 R² 損失函數
# def r_squared_loss(y_true, y_pred):
#     SS_res =  tf.reduce_sum(tf.square(y_true - y_pred))
#     SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
#     return 1 - (SS_res / (SS_tot + tf.keras.backend.epsilon()))

# 定義學習率調整函數
def lr_scheduler(epoch, lr):
    # 每 100 個 epochs 學習率下降為原來的十分之一
    if epoch > 0 and epoch % 50 == 0:
        return lr * 0.8
    return lr

################################################################################
################################## 工件材料性質 ##################################
################################################################################

# 載入材料數據標簽
labels_dict = {}
for freq in frequencies:
    label_groups = []
    count = 0
    for i in range(group_start, group_end + 1):
        for j in range(piece_num_start, piece_num_end + 1):  # 每大組包含5小組
            labels = excel_data.loc[count, freq]
            label_groups.extend([labels] * image_layers)
            count += 1
        
    labels_dict[freq] = np.array(label_groups)  # 轉換為NumPy數組


#################################################################################
#################################### 製程參數 ####################################
#################################################################################

# 載入製程參數
Process_parameters = ['氧濃度', '雷射掃描速度', '雷射功率', '線間距', '能量密度']
proc_dict = {}  # 儲存所有頻率全部大組製程參數
proc_dict_scaled = {}
for freq in frequencies:
    proc_groups = []  # 儲存全部大組製程參數
    for i in range(group_start, group_end + 1):
        group_procs = []  # 每大組的製程參數
        parameters_group = []
        for para in Process_parameters:
            parameters = excel_process.loc[i-1, para]
            parameters_group.append(parameters)

        for j in range(piece_num_start, piece_num_end + 1):  # 每大組包含5小組
            group_procs.extend([parameters_group] * image_layers)

        proc_groups.extend(group_procs)

    # 轉換為NumPy數組
    proc_dict[freq] = np.array(proc_groups)
    
    # 初始化 StandardScaler
    scaler = StandardScaler()
    
    # 標準化製程參數
    proc_dict_scaled[freq] = scaler.fit_transform(proc_dict[freq])


#################################################################################
#################################### 積層影像 ####################################
#################################################################################

# 載入圖像數據
image_groups = []

for group in range(group_start, group_end + 1):
    group_images = []
    for image_num in range(piece_num_start, piece_num_end + 1):
        folder_name = f'circle(340x345)/trail{group:01d}_{image_num:02d}'
        folder_path = f'data/{folder_name}/'

        image_group = []
        for i in range(image_layers):
            filename = f'{folder_path}/layer_{i + 1:02d}.jpg'
            image = cv2.imread(filename)
            image = cv2.resize(image, (image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image / 255.0  # 歸一化
            image_group.append(image)

        group_images.extend(image_group)

    image_groups.extend(group_images)

# 轉換為NumPy數組
images = np.array(image_groups)


#################################################################################
#################################### 測試模型 ####################################
#################################################################################

# 對於每個頻率進行模型訓練和保存
for freq in frequencies:
    for fold in range(1, k_fold_splits + 1):
        print(f"Training on fold {fold}/{k_fold_splits} for frequency {freq}")
        # 定義訓練集和驗證集
        x_train, y_train, proc_train = [], [], []
        x_val, y_val, proc_val = [], [], []

        for group in range(group_start, group_end + 1):
            for image_num in range(piece_num_start, piece_num_end + 1):
                # 計算在 labels_dict 和 proc_dict 中的索引
                index = (group - 1) * piece_num_end * image_layers + (image_num - 1) * image_layers

                # K-折交叉驗證
                if image_num == fold:
                    x_val.extend(images[index:index + image_layers])
                    y_val.extend(labels_dict[freq][index:index + image_layers])
                    proc_val.extend(proc_dict_scaled[freq][index:index + image_layers])

                else:
                    x_train.extend(images[index:index + image_layers])
                    y_train.extend(labels_dict[freq][index:index + image_layers])
                    proc_train.extend(proc_dict_scaled[freq][index:index + image_layers])

        # 轉換為 NumPy 數組
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        proc_train = np.array(proc_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)
        proc_val = np.array(proc_val)

        # print(y_val)
        # print(f'\n')
        # print(y_train)
                    
        # print(proc_val)
        # print(f'\n')
        # print(proc_train)
                    
        # print(len(images))
        # print(len(labels_dict[freq]))
        # print(len(proc_dict_scaled[freq]))
        
        # 設置貝葉斯優化
        tuner = kt.BayesianOptimization(
            build_model,
            objective='val_mae',
            max_trials=max_trials,
            num_initial_points=2,
            directory='my_dir/Images & Parameters/',
            project_name=f'bayesian_opt_conv_transformer_par_{freq}_fold_{fold}'
        )
        
        # 獲取當前頻率的標簽
        # current_labels = labels_dict[freq]
        # current_proc = proc_dict[freq]

        # for fold, (train_index, val_index) in enumerate(kf.split(images)):
        #     print(f"Training on fold {fold+1}/{k_fold_splits} for frequency {freq}")

        #     # 分割數據集
        #     x_train, x_val = images[train_index], images[val_index]
        #     y_train, y_val = current_labels[train_index], current_labels[val_index]
        #     proc_train, proc_val = current_proc[train_index], current_proc[val_index]
        #     print(train_index)
        #     print(val_index)

        #     # 轉換數據為DataFrame
        #     # df_x_train = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
        #     # df_x_val = pd.DataFrame(x_val.reshape(x_val.shape[0], -1))
        #     df_y_train = pd.DataFrame(y_train)
        #     df_y_val = pd.DataFrame(y_val)
        #     # df_proc_train = pd.DataFrame(proc_train)
        #     # df_proc_val = pd.DataFrame(proc_val)

        #     # 為每個組合 (freq, fold) 創建並保存Excel檔案
        #     file_name = f'data_{freq}_fold_{fold+1}.xlsx'
        #     with pd.ExcelWriter(file_name) as writer:
        #         # df_x_train.to_excel(writer, sheet_name='X_Train', index=False)
        #         # df_x_val.to_excel(writer, sheet_name='X_Val', index=False)
        #         df_y_train.to_excel(writer, sheet_name='Y_Train', index=False)
        #         df_y_val.to_excel(writer, sheet_name='Y_Val', index=False)
        #         # df_proc_train.to_excel(writer, sheet_name='Proc_Train', index=False)
        #         # df_proc_val.to_excel(writer, sheet_name='Proc_Val', index=False)

        # 將數據拆分為訓練集和驗證集
        # x_train, x_val, y_train, y_val, proc_train, proc_val = train_test_split(images, labels_dict[freq], proc_dict[freq], test_size=0.25, random_state=42)

        # 數據生成器
        train_data_generator = tf.data.Dataset.from_tensor_slices(((x_train, proc_train), y_train)).batch(batch_size)
        val_data_generator = tf.data.Dataset.from_tensor_slices(((x_val, proc_val), y_val)).batch(batch_size)
        
        # 開始搜索
        tuner.search(train_data_generator, epochs=trials_epochs, validation_data=val_data_generator)

        # 獲取最佳超參數並創建模型
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] # num_trials=1 表示只獲取一組最佳參數，而索引 [0] 表示從可能的最佳參數列表中獲取第一組。

        # 早期停止功能
        # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=100, verbose=1)

        # 學習率調整
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        # 創建 TensorBoard 回調
        tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)

        # 使用最佳超參數創建模型
        model = build_model(best_hps)
        print(f'Frequency: {freq}')
        # model.fit(train_data_generator, epochs=train_epochs, validation_data=val_data_generator, callbacks=[early_stopping, tensorboard_callback, lr_callback])
        model.fit(train_data_generator, epochs=train_epochs, validation_data=val_data_generator, callbacks=[tensorboard_callback, lr_callback])

        # 初始化 DataFrame 以存儲記錄（抓取訓練過程的趨勢資料）
        records = pd.DataFrame()
        
        # 獲取整體 epochs 的記錄
        current_records = pd.DataFrame(model.history.history)

        # 當前紀錄的實際長度
        actual_length = len(current_records)

        # 添加 epoch 列
        current_records.insert(0, 'epoch', range(1, actual_length + 1))

        # 將當前記錄添加到整體記錄中
        records = pd.concat([records, current_records], ignore_index=True)

        # 將 DataFrame 寫入 Excel 檔案
        # records.to_excel(f'Records/bayesian_conv_transformer_par_records_{freq}.xlsx', index=False)
        records.to_excel(f'Records/Images & Parameters/bayesian_conv_transformer_par_records_{freq}_fold_{fold}.xlsx', index=False)

        # 清除先前的訓練記錄
        records = records.iloc[0:0]  # 刪除所有行，得到一個空的 DataFrame

        # 保存模型權重
        # model.save_weights(f'Weight/bayesian_conv_transformer_par_model_weights_{freq}.h5')
        model.save_weights(f'Weight/Images & Parameters/bayesian_conv_transformer_par_model_weights_{freq}_fold_{fold}.h5')