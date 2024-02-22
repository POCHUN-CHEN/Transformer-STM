import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
import keras_tuner as kt
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler # 標準化製程參數
from tensorflow.keras.regularizers import l2

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 提取不同頻率
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['50HZ_μa', '200HZ_μa', '400HZ_μa', '800HZ_μa']
frequencies = ['50HZ_μa']

# 定義範圍
group_start = 11
group_end = 20
piece_num_start = 1
piece_num_end = 5

# 定義其他相關範圍或常數
image_layers = 200  # 每顆影像的層數

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

num_classes = 1  # 回歸任務

# 設置貝葉斯優化 epoch 數目
max_trials=20

k_fold_splits = 1

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')


################################################################################
##################################### 定義 #####################################
################################################################################

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
def build_cvt_model(image_height, image_width, num_channels, num_classes):
    inputs_img = keras.Input(shape=(image_height, image_width, num_channels), dtype='float32')

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

    # 回歸輸出層
    outputs = layers.Dense(num_classes, activation="linear")(x)

    # 建立和編譯模型
    model = keras.Model(inputs=inputs_img, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mean_squared_error', 
                  metrics=['mae'])

    return model


################################################################################
################################## 工件材料性質 ##################################
################################################################################

# 載入材料數據標簽
labels_dict = {}
for freq in frequencies:
    label_groups = []
    count = 0
    for i in range(1, group_end + 1):
        for j in range(piece_num_start, piece_num_end + 1):  # 每大組包含5小組
            labels = excel_data.loc[count, freq]
            label_groups.extend([labels] * image_layers)
            count += 1
        
    labels_dict[freq] = np.array(label_groups)  # 轉換為NumPy數組


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

# 對於每個頻率進行模型載入、預測和評估
for freq in frequencies:
    for fold in range(1, k_fold_splits + 1):
        print(f"Testing on fold {fold}/{k_fold_splits} for frequency {freq}")
        # 定義訓練集和驗證集
        x_val, y_val = [], []

        for group in range(group_start, group_end + 1):
            for image_num in range(piece_num_start, piece_num_end + 1):
                # 計算在 labels_dict 和 proc_dict 中的索引
                images_index = ((group - 1) * piece_num_end * image_layers + (image_num - 1) * image_layers)%((group_end + 1 - group_start) * (piece_num_end + 1 - piece_num_start) * image_layers)
                index = (group - 1) * piece_num_end * image_layers + (image_num - 1) * image_layers

                # K-折交叉驗證
                if image_num == fold:
                    x_val.extend(images[images_index:images_index + image_layers])
                    y_val.extend(labels_dict[freq][index:index + image_layers])

        # 轉換為 NumPy 數組
        x_val = np.array(x_val)
        y_val = np.array(y_val)

        # # 設置貝葉斯優化
        # tuner = kt.BayesianOptimization(
        #     build_model,
        #     objective='val_mae',
        #     max_trials=max_trials,
        #     num_initial_points=2,
        #     directory='my_dir/Images/',
        #     project_name=f'bayesian_opt_conv_transformer_{freq}_fold_{fold}'
        # )

        # # 重新加載最佳超參數
        # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # 構建模型
        # model = build_model(best_hps)
        model = build_cvt_model(image_height, image_width, num_channels, num_classes)

        # 載入模型權重
        model.load_weights(f'Weight/Images/bayesian_conv_transformer_model_weights_{freq}_fold_{fold}.h5')

        # 進行預測
        predictions = model.predict(x_val)

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
        plt.title(f'R^2 - {freq} - Fold {fold}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.savefig(f'Plots/Images/R^2_{freq}_fold_{fold}.png')  # 儲存圖片
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
        plt.title(f'Actual vs Predicted - {freq} - Fold {fold}')
        plt.legend()  # 顯示圖例
        plt.savefig(f'Plots/Images/Actual_vs_Predicted_{freq}_fold_{fold}.png')  # 儲存圖片
        plt.show()

# 列印模型簡報
# tuner.results_summary() #查看 BayesianOptimization 最佳解（Val_mae 分數越低越好）
model.summary()
