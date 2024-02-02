import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import keras_tuner as kt
from PIL import Image
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2

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

# 設置貝葉斯優化 epoch 數目
max_trials=20
trials_epochs=10

# 讀取Excel文件中的標簽數據
excel_process = pd.read_excel('Process_parameters.xlsx')


################################################################################
##################################### 定義 #####################################
################################################################################

# 定義Convolution Transformer模型建構函數
def build_model(hp):
    image_inputs = keras.Input(shape=(image_height, image_width, num_channels))
    process_inputs = keras.Input(shape=(proc_dict[freq].shape[1],))

    # Convolutional Blocks
    x = keras.layers.Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=32), 
                            (3, 3), activation='relu', kernel_regularizer=l2(0.01))(image_inputs) # 添加 L2 正則化
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
    x = keras.layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu', kernel_regularizer=l2(0.01))(x) # 添加 L2 正則化
    x = keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1))(x)
    x = keras.layers.concatenate([x, process_inputs])  # 將製程參數與其他特徵串聯
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs=[image_inputs, process_inputs], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# Grad-CAM 函數
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


#################################################################################
#################################### 製程參數 ####################################
#################################################################################

# 載入製程參數
Process_parameters = ['氧濃度', '雷射掃描速度', '雷射功率', '線間距', '能量密度']
proc_dict = {}  # 儲存所有頻率全部大組製程參數
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
            image = image / 255.0  # 標準化
            image_group.append(image)

        group_images.extend(image_group)

    image_groups.extend(group_images)

# 轉換為NumPy數組
images = np.array(image_groups)


#################################################################################
##################################### 熱力圖 #####################################
#################################################################################

# 對於每個頻率進行模型載入、預測和評估
for freq in frequencies:
    for fold in range(5):
        # 設置貝葉斯優化
        tuner = kt.BayesianOptimization(
            build_model,
            objective='val_mae',
            max_trials=max_trials,
            num_initial_points=2,
            directory='my_dir',
            project_name=f'bayesian_opt_conv_transformer_par_{freq}_fold_{fold+1}'
        )

        # 獲取當前頻率的標簽
        current_proc = proc_dict[freq]

        # 重新加載最佳超參數
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # 構建模型
        model = build_model(best_hps)

        # 載入模型權重
        model.load_weights(f'Weight/bayesian_conv_transformer_par_model_weights_{freq}_fold_{fold+1}.h5')

        # 圖像正則化
        images /= 255.0

        # 生成 Grad-CAM 熱力圖
        last_conv_layer_name = 'conv2d_2'  # 使用模型的最後一個卷積層名稱
        heatmap = make_gradcam_heatmap(images, model, last_conv_layer_name)

        # 將熱力圖轉換為8位整數格式
        heatmap = np.uint8(255 * heatmap)

        # 應用顏色映射
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # # 將 PIL 圖像轉換為 NumPy 數組
        # img_np = np.array(img)

        # 將單通道灰度圖像轉換為三通道圖像
        img_np_color = cv2.cvtColor(images, cv2.COLOR_GRAY2BGR)

        # 將熱力圖調整為原始圖像的大小並應用到原圖
        heatmap_resized = cv2.resize(heatmap_colored, (images.shape[1], images.shape[0]))
        superimposed_img = heatmap_resized * 0.4 + img_np_color

        # # 顯示結果
        # plt.figure(figsize=(10, 10))
        # plt.imshow(superimposed_img / 255.0)
        # plt.axis('off')
        # plt.show()

        # 創建一個圖形和三個子圖
        plt.figure(figsize=(18, 6))

        # 顯示第一張圖像
        plt.subplot(1, 4, 1)  # 1列2行的第1個
        plt.imshow(img_np_color / 255.0)
        plt.axis('off')
        plt.title('Original Image')

        # 顯示第二張圖像
        plt.subplot(1, 3, 2)  # 1列2行的第2個
        plt.imshow(heatmap_resized / 255.0)
        plt.axis('off')
        plt.title('Heatmap')

        # 顯示第三張圖像
        plt.subplot(1, 3, 3)  # 1列2行的第3個
        plt.imshow(superimposed_img / 255.0)
        plt.axis('off')
        plt.title('Superimposed')

        # 顯示圖形
        plt.show()