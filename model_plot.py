import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard

# 投影方式 (dw_bn/avg/linear)
projection_method = 'dw_bn'

# cls_token 是否打開 (True/False)
cls_token_switch = False

# 批次大小
batch_size = 128

# Spec 定義
spec = {
    'stages': [
        {'embed_dim': 64, 'patch_size': 7, 'stride': 4, 'num_heads': 1, 'kernel_size': 3, 'strides': 1, 'qkv_method': 'linear', 'with_cls_token': False},
        {'embed_dim': 128, 'patch_size': 3, 'stride': 2, 'num_heads': 2, 'kernel_size': 3, 'strides': 1, 'qkv_method': 'linear', 'with_cls_token': False},
        {'embed_dim': 256, 'patch_size': 3, 'stride': 2, 'num_heads': 4, 'kernel_size': 3, 'strides': 1, 'qkv_method': 'linear', 'with_cls_token': cls_token_switch},
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
def create_cvt_model(image_height, image_width, num_channels, proc_dim, num_classes):
    image_inputs = keras.Input(shape=(image_height, image_width, num_channels), name='Image_inputs')
    proc_inputs = keras.Input(shape=(proc_dim,), name='Proc_inputs')

    x = image_inputs
    _,height, width, _ = extract_dimensions(x)

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
    # if stage_spec['with_cls_token']:
    #     print("cls_tokens")
    #     x = layers.LayerNormalization(epsilon=1e-6)(cls_tokens)
    #     x = tf.squeeze(x, axis=1)
        
    # else:
    #     print("No cls_tokens")
    #     _,height, width, num_channels = extract_dimensions(x)
    #     x = tf.reshape(x, [-1, height * width, num_channels])
    #     x = layers.LayerNormalization(epsilon=1e-6)(x)
    #     x = tf.reduce_mean(x, axis=1)

    if stage_spec['with_cls_token']:
        x = tf.squeeze(cls_tokens, axis=1)
    else:
        x = tf.reduce_mean(x, axis=[1, 2])
    
    # 處理製程參數
    proc_features = layers.Dense(256, activation='relu', name='Proc_Dense_1')(proc_inputs)
    proc_features = layers.Dense(256, activation='relu', name='Proc_Dense_2')(proc_features)

    # 將圖像特徵和製程參數特徵連接起來
    concatenated = layers.concatenate([x, proc_features])

    # 輸出層
    outputs = layers.Dense(num_classes, activation='linear',name='Final_Dense')(concatenated)

    # 創建模型
    model = keras.Model(inputs=[image_inputs, proc_inputs], outputs=outputs)
    return model

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

# 假設 proc_dim 和 num_classes
proc_dim = 5  # 假設處理參數維度為5
num_classes = 1  # 回歸任務

# 使用模型建立函數來創建模型
model = create_cvt_model(image_height, image_width, num_channels, proc_dim, num_classes)

# 使用 plot_model 函數畫出模型架構
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
