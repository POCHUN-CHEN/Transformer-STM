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
frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['50HZ_Hc']
# frequencies = ['50HZ_μa']

# 投影方式 (dw_bn/avg/linear)
projection_method = 'dw_bn'

# cls_token 是否打開 (True/False)
cls_token_switch = True

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

# Spec 定義
# ConvEmbed {embed_dim，patch_size, stride}
# ConvTransformerBlock {embed_dim, num_heads, kernel_size, strides, qkv_method, with_cls_token}
spec = {
    'stages': [
        {'embed_dim': 64, 'patch_size': 7, 'stride': 4, 'num_heads': 1, 'kernel_size': 3, 'strides': 1, 'qkv_method': projection_method, 'with_cls_token': False},
        {'embed_dim': 128, 'patch_size': 3, 'stride': 2, 'num_heads': 2, 'kernel_size': 3, 'strides': 1, 'qkv_method': projection_method, 'with_cls_token': False},
        {'embed_dim': 256, 'patch_size': 3, 'stride': 2, 'num_heads': 4, 'kernel_size': 3, 'strides': 1, 'qkv_method': projection_method, 'with_cls_token': cls_token_switch},
    ]
}

# 定義一個提取圖像的高度、寬度和通道數的函數
def extract_dimensions(x):
    batch_size = tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    num_channels = tf.shape(x)[3]
    return batch_size, height, width, num_channels

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
            # self.proj = layers.Dense(dim)
            self.proj = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def call(self, inputs):
        if self.method == 'dw_bn':
            # 確保輸入 DepthwiseConv2D 張量的維度為 4
            x = self.conv(inputs)
            x = self.bn(x)
        elif self.method == 'avg':
            x = self.avg_pool(inputs)
        elif self.method == 'linear':
            # x = self.proj(inputs)
            x = inputs
        
        return x

# 定義 ConvAttention 層
class ConvAttention(layers.Layer):
    def __init__(self, dim, num_heads, kernel_size, strides, padding, qkv_method='dw_bn', attn_drop=0.1, proj_drop=0.1, with_cls_token=True, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.qkv_method = qkv_method
        self.with_cls_token = with_cls_token

        # 創建Q、K、V的卷積投影
        self.q_proj = Projection(dim, kernel_size, strides, padding, 'linear' if qkv_method == 'avg' else qkv_method, name='q_proj')
        self.k_proj = Projection(dim, kernel_size, strides, padding, method=qkv_method, name='k_proj')
        self.v_proj = Projection(dim, kernel_size, strides, padding, method=qkv_method, name='v_proj')

        # 創建Q、K、V的線性投影
        self.proj_q = layers.Dense(dim)
        self.proj_k = layers.Dense(dim)
        self.proj_v = layers.Dense(dim)

        # 創建多頭注意力機制
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)

        # 添加Dropout層
        self.attn_dropout = layers.Dropout(attn_drop)
        self.proj_dropout = layers.Dropout(proj_drop)
        self.proj = layers.Dense(dim)
        
    def call(self, inputs, height, width):
        # 這邊不能直接取用 input 的 shape ，因為有加入 Cls_token 只能由外部傳入。
        batch_size = tf.shape(inputs)[0]
        num_channels = tf.shape(inputs)[2]
        
        if self.with_cls_token:
            cls_tokens, inputs = tf.split(inputs, [1, height * width], axis=1)
            inputs = tf.reshape(inputs, [batch_size, height, width, num_channels])
        else:
            inputs = tf.reshape(inputs, [batch_size, height, width, num_channels])
            
        # 計算 query, key, value
        # 執行卷積投影
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        if self.with_cls_token:            
            # 確保輸入 attention 張量的維度為 3
            q = tf.reshape(q, [batch_size, height * width, num_channels])
            k = tf.reshape(k, [batch_size, height * width, num_channels])
            v = tf.reshape(v, [batch_size, height * width, num_channels])

            # 把 cls_tokens 串接到 qkv 之前
            q = tf.concat([cls_tokens, q], axis=1)
            k = tf.concat([cls_tokens, k], axis=1)
            v = tf.concat([cls_tokens, v], axis=1)
            # print("進入cls_token！")
        else:
            # 確保輸入 attention 張量的維度為 3
            q = tf.reshape(q, [batch_size, height * width, num_channels])
            k = tf.reshape(k, [batch_size, height * width, num_channels])
            v = tf.reshape(v, [batch_size, height * width, num_channels])
            # print("不進入cls_token！")

        # 執行線性投影
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)
        
        # 注意力機制操作
        attn_output = self.attention(q, v, k)

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
    def __init__(self, embed_dim=64, patch_size=7, stride=4, padding="same", norm_layer=None, name=None):
        super().__init__(name=name)
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
        x = self.proj(inputs) # 投影處理
        if self.norm:
            # 在 TensorFlow 中，需要將張量從 [B, H, W, C] 重塑為 [B, H*W, C] 進行層標準化，
            # 但由於 TensorFlow 的 LayerNormalization 直接在最後一維上工作，因此這裡不需要重塑。
            x = self.norm(x)
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

# 定義 ConvTransformerBlock 層
class ConvTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, kernel_size, strides, padding, qkv_method='dw_bn', Mlp_dim_factor=4, dropout_rate=0.1, with_cls_token=True, name=None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.qkv_method = qkv_method
        self.with_cls_token = with_cls_token
        self.Mlp_dim_factor = Mlp_dim_factor  # 控制隱藏層大小的倍數

        # 初始化 cls_token
        if with_cls_token:
            self.cls_token = self.add_weight(shape=(1, 1, 1, embed_dim), initializer='zeros', trainable=True, name='cls_token') # 四維度符合圖像處理

        # [未加入Dim_in/Dim_out不同]
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = ConvAttention(embed_dim, num_heads, kernel_size, strides, padding, qkv_method=qkv_method, with_cls_token=with_cls_token)
        # self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Mlp (Multi-Layered Perceptrons)實現
        # a.k.a. Feed Forward Neural Networks (FFNNs)
        self.Mlp = keras.Sequential([
            layers.Dense(embed_dim * self.Mlp_dim_factor, activation=tf.nn.gelu),  # 可配置的隱藏層大小
            layers.Dropout(dropout_rate),  # 加入Dropout層
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate),  # 加入Dropout層
        ])
        self.output_conv = layers.Conv2D(embed_dim, kernel_size=1)
        
    def call(self, inputs):
        batch_size, height, width, num_channels = extract_dimensions(inputs)
        # batch_size = tf.shape(inputs)[0]
        # height = tf.shape(inputs)[1]
        # width = tf.shape(inputs)[2]
        # num_channels = tf.shape(inputs)[3]

        if self.with_cls_token:
            cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1, 1])
            cls_tokens = tf.reshape(cls_tokens, [batch_size, 1, num_channels])
            inputs = tf.reshape(inputs, [batch_size, height * width, num_channels])
            inputs = tf.concat([cls_tokens, inputs], axis=1)
        else:
            inputs = tf.reshape(inputs, [batch_size, height * width, num_channels])
        
        x = self.norm1(inputs)
        attn_output = self.attn(x, height, width)
        x = attn_output + inputs

        # [未加入DropPath]

        y = self.norm1(x)
        Mlp_output = self.Mlp(y)
        
        output = x + Mlp_output
        
        if self.with_cls_token:
            cls_tokens, output = tf.split(output, [1, height * width], axis=1)
            output = tf.reshape(output, [batch_size, height, width, num_channels])
            return output, cls_tokens
        else:
            output = tf.reshape(output, [batch_size, height, width, num_channels])
            return output

# 建立CvT模型
def create_cvt_model(image_height, image_width, num_channels, num_classes):
    image_inputs = keras.Input(shape=(image_height, image_width, num_channels), name='Image_inputs')
    # proc_inputs = keras.Input(shape=(proc_dim,), name='Proc_inputs')

    x = image_inputs
    _,height, width, _ = extract_dimensions(x)
    # height = tf.shape(x)[1]
    # width = tf.shape(x)[2]

    # 使用spec來動態創建多個階段
    for i, stage_spec in enumerate(spec['stages']):
        x = ConvEmbed(embed_dim=stage_spec['embed_dim'], 
                                  patch_size=stage_spec['patch_size'], 
                                  stride=stage_spec['stride'], 
                                  norm_layer=layers.LayerNormalization, 
                                  name=f'stage{i+1}_ConvEmbed')(x)
        if stage_spec['with_cls_token']:
            x, cls_tokens = ConvTransformerBlock(embed_dim=stage_spec['embed_dim'], 
                                                 num_heads=stage_spec['num_heads'],
                                                 kernel_size=stage_spec['kernel_size'], 
                                                 strides=stage_spec['strides'], 
                                                 padding='same', 
                                                 qkv_method=stage_spec['qkv_method'], 
                                                 with_cls_token=stage_spec['with_cls_token'], 
                                                 name=f'stage{i+1}_transformer')(x)
        else:
            x = ConvTransformerBlock(embed_dim=stage_spec['embed_dim'], 
                                    num_heads=stage_spec['num_heads'],
                                    kernel_size=stage_spec['kernel_size'], 
                                    strides=stage_spec['strides'], 
                                    padding='same', 
                                    qkv_method=stage_spec['qkv_method'], 
                                    with_cls_token=stage_spec['with_cls_token'], 
                                    name=f'stage{i+1}_transformer')(x)

    # 最後處理cls_tokens如果它們存在
    if stage_spec['with_cls_token']:
        print("cls_tokens")
        x = layers.LayerNormalization(epsilon=1e-6)(cls_tokens)
        x = tf.squeeze(x, axis=1)
        
    else:
        print("No cls_tokens")
        _,height, width, num_channels = extract_dimensions(x)
        # height = tf.shape(x)[1]
        # width = tf.shape(x)[2]
        # num_channels = tf.shape(x)[3]
        x = tf.reshape(x, [-1, height * width, num_channels])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.reduce_mean(x, axis=1)

    # 輸出層
    outputs = layers.Dense(num_classes, activation='linear',name='Final_Dense')(x)

    # 創建模型
    model = keras.Model(inputs=[image_inputs], outputs=outputs)
    return model

# 數據預處理函數
def preprocess_data(excel_data, group_start, group_end, piece_num_start, piece_num_end, image_layers, image_height, image_width):
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

    return labels_dict, images, valid_dict, count

# 測試模型並保存結果
def test_and_save_results(freq, labels_dict, images, valid_dict, count):
    # 定義驗證集
    x_val, y_val = [], []

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

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # 構建模型
    model = create_cvt_model(image_height, image_width, num_channels, num_classes)

    # 載入模型權重
    model.load_weights(f'Weight/Images/cvt_model_weights_{freq}_{projection_method}_cls{cls_token_switch}.h5')

    # 進行預測
    predictions = model.predict([x_val])

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
    plt.savefig(f'Plots/Images/CvT_R^2_{freq}_{projection_method}_cls{cls_token_switch}.png')
    plt.clf()

    # 繪製實際值與預測值的線圖
    image_numbers = np.arange(1, len(predictions) + 1)
    plt.plot(image_numbers, y_val, label='Actual', marker='o')
    plt.plot(image_numbers, predictions.flatten(), label='Predicted', marker='x')
    plt.xlabel('Image Number')
    plt.ylabel('Values')
    plt.title(f'Actual vs Predicted - {freq}')
    plt.legend()
    plt.savefig(f'Plots/Images/CvT_Actual_vs_Predicted_{freq}_{projection_method}_cls{cls_token_switch}.png')
    plt.clf()

# 主程序
for freq in frequencies:
    print(f"Testing for frequency {freq}")

    labels_dict, images, valid_dict, count = preprocess_data(excel_data, group_start, group_end, piece_num_start, piece_num_end, image_layers, image_height, image_width)

    test_and_save_results(freq, labels_dict, images, valid_dict, count)