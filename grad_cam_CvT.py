import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
import collections
from itertools import repeat

from tensorflow.keras.preprocessing import image


import matplotlib.pyplot as plt
import keras_tuner as kt
from PIL import Image
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
from tqdm import tqdm


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 動態記憶體分配
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 提取不同頻率
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['50HZ_μa', '200HZ_μa', '400HZ_μa', '800HZ_μa']
frequencies = ['50HZ_μa']

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

num_classes = 1  # 回歸任務

# # 批次大小
# batch_size = 128

# # 設置 epoch 數目
# train_epochs = 1000

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')
excel_process = pd.read_excel('Process_parameters.xlsx')


################################################################################
##################################### 定義 #####################################
################################################################################

# 定義 Projection 層
class Projection(layers.Layer):
    def __init__(self, dim, kernel_size, strides, padding='same', method='dw_bn', name=None):
        super().__init__(name=name)
        self.dim = dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.method = method

        if method == 'dw_bn':
            self.conv = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)
            self.bn = layers.BatchNormalization()
        elif method == 'avg':
            self.avg_pool = layers.AveragePooling2D(pool_size=kernel_size, strides=strides, padding='same')
        elif method == 'linear':
            self.proj = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def call(self, inputs):
        if self.method == 'dw_bn':
            x = self.conv(inputs)
            x = self.bn(x)
        elif self.method == 'avg':
            x = self.avg_pool(inputs)
        elif self.method == 'linear':
            x = self.proj(inputs)

        # print("留言密碼測試")
        # print("Projection x:", x.shape)
        
        return x

# 定義 ConvAttention 層
class ConvAttention(layers.Layer):
    def __init__(self, dim, num_heads, kernel_size, strides, padding, qkv_method='dw_bn', attn_drop=0.1, proj_drop=0.1, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.qkv_method = qkv_method

        # 創建Q、K、V的卷積投影
        self.q_proj = Projection(dim, kernel_size, strides, padding, 'linear' if qkv_method == 'avg' else qkv_method, name='q_proj')
        self.k_proj = Projection(dim, kernel_size, strides, padding, method=qkv_method, name='k_proj')
        self.v_proj = Projection(dim, kernel_size, strides, padding, method=qkv_method, name='v_proj')

        # 創建Q、K、V的線性投影（非必要）
        self.proj_q = layers.Dense(dim)
        self.proj_k = layers.Dense(dim)
        self.proj_v = layers.Dense(dim)

        # 創建多頭注意力機制
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)

        # 添加Dropout層
        self.attn_dropout = layers.Dropout(attn_drop)
        self.proj_dropout = layers.Dropout(proj_drop)
        self.proj = layers.Dense(dim)

    def call(self, inputs):
        # print("留言密碼測試")
        # print("Shape of inputs:", inputs.shape)
        # 計算 query, key, value
        # 執行卷積投影
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        # 執行線性投影（非必要）
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        # # print 語句來印出形狀
        # print("Shape of q:", q.shape)
        # print("Shape of k:", k.shape)
        # print("Shape of v:", v.shape)

        _, h, w, c = q.shape
        # print("Before reshape - h:", h, ", w:", w, ", c:", c)
        q = tf.reshape(q, [-1, h * w, c])
        k = tf.reshape(k, [-1, h * w, c])
        v = tf.reshape(v, [-1, h * w, c])

        # # 重塑操作後印出形狀
        # print("After reshape - Shape of q:", q.shape)
        # print("After reshape - Shape of k:", k.shape)
        # print("After reshape - Shape of v:", v.shape)

        # 注意力機制操作
        attn_output = self.attention(q, v, k)
        # print("Shape of attn_output:", attn_output.shape)
        attn_output = self.attn_dropout(attn_output) # （非必要）

        # 將輸出的形狀從 (batch_size, height * width, channels) 轉變回原始的形狀
        attn_output = tf.reshape(attn_output, [-1, h, w, c])
        # 重塑回原始形狀後印出形狀
        # print("After reshape back - Shape of attn_output:", attn_output.shape)

        # 線性變換並應用dropout（非必要）
        output = self.proj(attn_output)
        output = self.proj_dropout(output)

        return output

# def _ntuple(n):
#     def parse(x):
#         if isinstance(x, collections.abc.Iterable):
#             return x
#         return tuple(repeat(x, n))
#     return parse

# to_2tuple = _ntuple(2)
# to_3tuple = _ntuple(3)
# to_4tuple = _ntuple(4)
# to_ntuple = _ntuple

# 定義 ConvEmbed 層
class ConvEmbed(layers.Layer):
    def __init__(self, embed_dim=64, patch_size=7, stride=4, padding="same", norm_layer=None):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride
        self.padding = padding
        self.norm_layer_name = norm_layer
        
        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=stride,
            padding=padding
        )
        # self.norm = norm_layer(epsilon=1e-6) if norm_layer else layers.LayerNormalization(epsilon=1e-6)
        self.norm = layers.LayerNormalization(axis=-1) if norm_layer == "LayerNormalization" else None

    def call(self, inputs):
        # print("inputs:", inputs.shape)
        x = self.proj(inputs) # 投影處理
        # print("x:", x.shape)
        if self.norm:
            # 在 TensorFlow 中，需要將張量從 [B, H, W, C] 重塑為 [B, H*W, C] 進行層標準化，
            # 但由於 TensorFlow 的 LayerNormalization 直接在最後一維上工作，因此這裡不需要重塑。
            x = self.norm(x)
        # print("ConvEmbed x:", x.shape)
        # print("留言密碼測試")
        return x
    
    def get_config(self):
        config = super(ConvEmbed, self).get_config()
        config.update({
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "stride": self.stride,
            "padding": self.padding,
            "norm_layer": self.norm_layer_name,  # 僅保存 norm_layer 的名稱
        })
        return config


# # 定義 ConvBlock 層
# class ConvBlock(layers.Layer):
#     def __init__(self, filters, kernel_size, strides, name=None):
#         super().__init__(name=name)
#         self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu')
#         self.norm = layers.LayerNormalization(epsilon=1e-6)

#     def call(self, inputs):
#         x = self.conv(inputs)
#         x = self.norm(x)
#         return x

# 定義 ConvTransformerBlock 層
class ConvTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, kernel_size, strides, padding, qkv_method='dw_bn', ffn_dim_factor=4, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.qkv_method = qkv_method
        self.ffn_dim_factor = ffn_dim_factor  # 控制隱藏層大小的倍數

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = ConvAttention(dim, num_heads, kernel_size, strides, padding, qkv_method=qkv_method)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        #  Mlp 實現
        self.ffn = keras.Sequential([
            layers.Dense(dim * self.ffn_dim_factor, activation=tf.nn.gelu),  # 可配置的隐藏层大小
            layers.Dropout(dropout_rate),  # 加入Dropout层
            layers.Dense(dim),
        ])
        # print("Shape of CTdim:", dim)
        # 假設 dim 是目標通道數，我們需要確保調整層的輸出通道數也是 dim
        # self.adjust_channels = layers.Conv2D(dim, kernel_size=1, padding='same', use_bias=False)  # 添加 use_bias=False 以匹配您的其他卷積層設置
        self.output_conv = layers.Conv2D(dim, kernel_size=1)

    def call(self, inputs):
        x = self.norm1(inputs)
        attn_output = self.attn(x)
        x = attn_output + inputs

        # # 檢查並調整通道數
        # if attn_output.shape[-1] != inputs.shape[-1]:
        #     adjusted_inputs = self.adjust_channels(inputs)
        # else:
        #     adjusted_inputs = inputs

        # x = attn_output + adjusted_inputs  # 現在形狀相匹配，可以相加

        y = self.norm2(x)
        # y = self.ffn(y)
        ffn_output = self.ffn(y)
        ffn_output = self.output_conv(ffn_output) # 調整 FFN 輸出維度
        # ffn_output = tf.reshape(ffn_output, tf.shape(x))  # 使用 tf.reshape 調整 FFN 輸出維度
        return x + ffn_output

# 建立CvT模型
def create_cvt_model(image_height, image_width, num_channels, proc_dim, num_classes):
    image_inputs = keras.Input(shape=(image_height, image_width, num_channels))
    proc_inputs = keras.Input(shape=(proc_dim,))

    x = image_inputs

    # Stage 1
    x = ConvEmbed(embed_dim=64, patch_size=7, stride=4, norm_layer=layers.LayerNormalization)(x)
    x = ConvTransformerBlock(64, num_heads=1, kernel_size=3, strides=1, padding='same', qkv_method='dw_bn', name='stage1_transformer')(x)

    # Stage 2
    x = ConvEmbed(embed_dim=128, patch_size=3, stride=2, norm_layer=layers.LayerNormalization)(x)
    x = ConvTransformerBlock(128, num_heads=2, kernel_size=3, strides=1, padding='same', qkv_method='dw_bn', name='stage2_transformer')(x)

    # Stage 3
    x = ConvEmbed(embed_dim=256, patch_size=3, stride=2, norm_layer=layers.LayerNormalization)(x)
    x = ConvTransformerBlock(256, num_heads=4, kernel_size=3, strides=1, padding='same', qkv_method='dw_bn', name='stage3_transformer')(x)


    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # 處理製程參數
    proc_features = layers.Dense(256, activation='relu')(proc_inputs)
    proc_features = layers.Dense(256, activation='relu')(proc_features)

    # 將圖像特徵和製程參數特徵連接起來
    concatenated = layers.concatenate([x, proc_features])

    # 輸出層
    outputs = layers.Dense(num_classes, activation='linear')(concatenated)

    # 創建模型
    model = keras.Model(inputs=[image_inputs, proc_inputs], outputs=outputs)
    return model

# 數據預處理函數
def preprocess_data(excel_data, excel_process, group_start, group_end, piece_num_start, piece_num_end, image_layers, image_height, image_width):
    # 載入材料數據標簽
    labels_dict = []
    valid_dict = []

    start_index = (group_start - 1) * (piece_num_end - piece_num_start + 1)
    end_index = group_end * ((piece_num_end - piece_num_start + 1))

    valid_indices = []
    label_groups = []
    count = 0
    for i in range(1, group_end + 1):
        for j in range(piece_num_start, piece_num_end + 1):
            labels = excel_data.loc[count, str(freq)]
            if not pd.isnull(labels):
                if start_index <= count < end_index:
                    label_groups.extend([labels] * image_layers)
                valid_indices.append(count)
            count += 1

    labels_dict = np.array(label_groups)
    valid_indices = [index for index in valid_indices if start_index <= index < end_index]
    valid_dict = np.array(valid_indices)

    # 載入製程參數
    Process_parameters = ['氧濃度', '雷射掃描速度', '雷射功率', '線間距', '能量密度']
    proc_dict = []
    valid_proc_groups = []
    for index in valid_dict:
        group_procs = []
        parameters_group = []
        group_index = index // (piece_num_end - piece_num_start + 1)

        for para in Process_parameters:
            parameters = excel_process.loc[group_index, para]
            parameters_group.append(parameters)

        group_procs.extend([parameters_group] * image_layers)
        valid_proc_groups.extend(group_procs)

    proc_dict = np.array(valid_proc_groups)

    # 標準化製程參數
    scaler = StandardScaler()
    proc_dict_scaled = scaler.fit_transform(proc_dict)

    # 處理積層影像
    valid_images = []
    for index in valid_dict:
        group_index = index // (piece_num_end - piece_num_start + 1) + 1
        image_num = index % (piece_num_end - piece_num_start + 1) + 1

        folder_name = f'circle(340x345)/trail{group_index:01d}_{image_num:02d}'
        folder_path = f'data/{folder_name}/'

        for i in range(image_layers):
            filename = f'{folder_path}/layer_{i + 1:02d}.jpg'
            image = cv2.imread(filename)
            image = cv2.resize(image, (image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image / 255.0
            valid_images.append(image)

    images = np.array(valid_images)
    print("images :", images.shape)

    return labels_dict, proc_dict_scaled, images, valid_dict, count

# Grad-CAM 函數
def make_gradcam_heatmap(img_array, proc_array, model, last_conv_layer_name, pred_index=None, batch_size=4):
    img_dataset = tf.data.Dataset.from_tensor_slices((img_array, proc_array))
    img_dataset = img_dataset.batch(batch_size)

    # print("img_array :", img_array.shape)
    # print("proc_array :", proc_array.shape)
    gpu_memory_usage = tf.config.experimental.get_memory_usage("/gpu:0")
    print(f"GPU memory usage 1: {gpu_memory_usage} bytes")

    heatmaps = []
    # for img_batch, proc_batch in tqdm(img_dataset, desc="處理 Grad-CAM 圖像進度"):
    #     # 調整輸入張量的形狀
    #     if len(img_batch.shape) == 3:
    #         img_batch = tf.expand_dims(img_batch, axis=-1)
        
    #     # 轉換數據類型為 float32
    #     img_batch = tf.cast(img_batch, tf.float16)
    #     proc_batch = tf.cast(proc_batch, tf.float16)

    #     # grad_model 不僅輸出原始模型的最終輸出，也輸出指定的一個中間層的輸出。
    #     grad_model = tf.keras.models.Model(
    #         model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    #     )
        
    #     # TensorFlow 提供的一種自動微分函式，用於記錄在其作用域下執行的所有操作的梯度。
    #     with tf.GradientTape() as tape:
    #         last_conv_layer_output, preds = grad_model([img_batch, proc_batch])
    #         if pred_index is None:
    #             pred_index = tf.argmax(preds[0])
    #         class_channel = preds[:, pred_index]

    #     grads = tape.gradient(class_channel, last_conv_layer_output)
    #     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    #     last_conv_layer_output = last_conv_layer_output[0]
    #     heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    #     heatmap = tf.squeeze(heatmap)

    #     # 對熱力圖進行後處理，使其更適合可視化
    #     heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    #     heatmaps.append(heatmap.numpy())

    # # 將所有熱力圖堆疊到一個數組中
    # heatmaps = np.concatenate(heatmaps, axis=0)
    # return heatmaps


    for img_batch, proc_batch in tqdm(img_dataset, desc="處理 Grad-CAM 圖像進度"):
        # 調整輸入張量的形狀
        if len(img_batch.shape) == 3:
            img_batch = tf.expand_dims(img_batch, axis=-1)
        
        # 轉換數據類型為 float32
        img_batch = tf.cast(img_batch, tf.float16)
        proc_batch = tf.cast(proc_batch, tf.float16)

        # 創建一個新的模型，輸出最後卷積層的輸出和最終模型的預測
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            # 使用 grad_model 進行預測，這會提供我們需要的中間層輸出和最終預測
            last_conv_layer_output, predictions = grad_model([img_batch, proc_batch])
            tape.watch(last_conv_layer_output)  # 確保這里的 last_conv_layer_output 是被監控的張量

            # 假設是回歸問題，我們關注於模型的輸出
            # predicted = predictions[..., 0]  # 取決於你的模型輸出結構
            predicted = predictions[:, 0]

        # 計算關於最後卷積層輸出的梯度
        grads = tape.gradient(predicted, last_conv_layer_output)

        # 計算梯度的全局平均，以便得到權重
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 打印预测和梯度
        print("Predictions:", predictions.numpy())
        print("Gradients shape:", grads.shape)
        print("Pooled gradients:", pooled_grads.numpy())

        for i in range(predictions.shape[0]):
            # 使用權重加權最後的卷積層特征圖，以生成熱力圖
            heatmap = tf.reduce_sum(pooled_grads * last_conv_layer_output[i], axis=-1)
            heatmap = tf.squeeze(heatmap)

            # 對熱力圖進行後處理，使其更適合可視化
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmaps.append(heatmap.numpy())

    heatmaps = np.concatenate(heatmaps, axis=0)
    # 返回所有熱力圖
    return heatmaps

# 測試模型並保存結果
def test_and_save_results(freq, labels_dict, proc_dict_scaled, images, valid_dict, count):
    # 定義驗證集
    x_val, y_val, proc_val = [], [], []

    first_valid_indices_per_group = []
    for d in range(0, count, 5):
        for j in range(d, d + 5):
            if j in valid_dict:
                first_valid_indices_per_group.append(j)
                break

    for i in range(len(valid_dict)):
        index = i * image_layers

        if valid_dict[i] in first_valid_indices_per_group:
            x_val.extend(images[index:index + image_layers])
            y_val.extend(labels_dict[index:index + image_layers])
            proc_val.extend(proc_dict_scaled[index:index + image_layers])

    x_val = np.array(x_val)
    y_val = np.array(y_val)
    proc_val = np.array(proc_val)

    # 構建模型
    model = create_cvt_model(image_height, image_width, num_channels, proc_dict_scaled.shape[1], num_classes)
    # model.summary()
    
    # 載入模型權重
    model.load_weights(f'Weight/Images & Parameters/cvt_model_weights_{freq}.h5')

    # 生成 Grad-CAM 熱力圖
    last_conv_layer_name = 'stage3_transformer'  # 使用模型的最後一個卷積層名稱
    heatmap = make_gradcam_heatmap(x_val, proc_val, model, last_conv_layer_name)

    # 將熱力圖轉換為8位整數格式
    heatmap = np.uint8(255 * heatmap)

    # 應用顏色映射
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 將單通道灰度圖像轉換為三通道圖像
    img_np_color = []
    for image in x_val:
        # 將圖像數據類型轉換為 8 位無符號整數
        image = np.uint8(image * 255)
        img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_np_color.append(img_color)
    img_np_color = np.array(img_np_color)

    # 將熱力圖調整為原始圖像的大小並應用到原圖
    heatmap_resized = cv2.resize(heatmap_colored, (x_val.shape[2], x_val.shape[1]))
    heatmap_resized = np.expand_dims(heatmap_resized, axis=0)
    heatmap_resized = np.repeat(heatmap_resized, x_val.shape[0], axis=0)
    superimposed_img = heatmap_resized * 0.4 + img_np_color

    # 將 superimposed_img 的值範圍縮放到 [0, 255],並轉換為 uint8 類型
    superimposed_img = (superimposed_img / np.max(superimposed_img) * 255.0).astype(np.uint8)

    print("Original Image data type:", img_np_color.dtype)
    print("Original Image min value:", np.min(img_np_color))
    print("Original Image max value:", np.max(img_np_color))

    print("Heatmap data type:", heatmap_resized.dtype)
    print("Heatmap min value:", np.min(heatmap_resized))
    print("Heatmap max value:", np.max(heatmap_resized))

    print("Superimposed data type:", superimposed_img.dtype)
    print("Superimposed min value:", np.min(superimposed_img))
    print("Superimposed max value:", np.max(superimposed_img))



    # 逐個顯示多個圖像
    for image_index in tqdm(range(len(img_np_color)), desc="處理圖像進度"):
        # 創建一個圖形和三個子圖
        plt.figure(figsize=(18, 6))

        # 顯示原始圖像
        plt.subplot(1, 4, 1)  # 1列2行的第1個
        plt.imshow(img_np_color[image_index])
        plt.axis('off')
        plt.title('Original Image')

        # 顯示熱力圖
        plt.subplot(1, 3, 2)  # 1列2行的第2個
        plt.imshow(heatmap_resized[image_index])
        plt.axis('off')
        plt.title('Heatmap')

        # 顯示疊加圖像
        plt.subplot(1, 3, 3)  # 1列2行的第3個
        plt.imshow(superimposed_img[image_index])
        plt.axis('off')
        plt.title('Superimposed')

        # 顯示圖形
        # plt.show()
        plt.savefig(f'Plots/Heatmap/Heatmap_{image_index+1}.png')
        plt.clf() # 清理圖形，避免圖形重疊
        plt.close() # 關閉當前圖形，釋放資源

        # print(f'儲存第_{image_index+1}張圖片，共{len(img_np_color)}張。')


    # 進行預測
    predictions = model.predict([x_val, proc_val])

    # 計算評估指標
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

#################################################################################
##################################### 熱力圖 #####################################
#################################################################################
# 主程序
for freq in frequencies:
    print(f"Testing for frequency {freq}")

    labels_dict, proc_dict_scaled, images, valid_dict, count = preprocess_data(excel_data, excel_process, group_start, group_end, piece_num_start, piece_num_end, image_layers, image_height, image_width)

    test_and_save_results(freq, labels_dict, proc_dict_scaled, images, valid_dict, count)

        