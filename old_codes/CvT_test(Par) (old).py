import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 提取不同頻率
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['400HZ_Pcv']
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

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')
excel_process = pd.read_excel('Process_parameters.xlsx')

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

    return labels_dict, proc_dict_scaled, images, valid_dict, count

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

    # 載入模型權重
    model.load_weights(f'Weight/Images & Parameters/cvt_model_weights_{freq}.h5')

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

    # 繪製 R^2 圖
    plt.scatter(y_val, predictions.flatten())
    plt.title(f'R^2 - {freq}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.savefig(f'Plots/Images & Parameters/R^2_{freq}.png')
    plt.clf()

    # 繪製實際值與預測值的線圖
    image_numbers = np.arange(1, len(predictions) + 1)
    plt.plot(image_numbers, y_val, label='Actual', marker='o')
    plt.plot(image_numbers, predictions.flatten(), label='Predicted', marker='x')
    plt.xlabel('Image Number')
    plt.ylabel('Values')
    plt.title(f'Actual vs Predicted - {freq}')
    plt.legend()
    plt.savefig(f'Plots/Images & Parameters/Actual_vs_Predicted_{freq}.png')
    plt.clf()

# 主程序
for freq in frequencies:
    print(f"Testing for frequency {freq}")

    labels_dict, proc_dict_scaled, images, valid_dict, count = preprocess_data(excel_data, excel_process, group_start, group_end, piece_num_start, piece_num_end, image_layers, image_height, image_width)

    test_and_save_results(freq, labels_dict, proc_dict_scaled, images, valid_dict, count)