from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1


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
def build_cvt_model(image_height, image_width, num_channels, proc_dict, num_classes):
    inputs_img = keras.Input(shape=(image_height, image_width, num_channels), dtype='float32') # 更改數值精度
    inputs_process = keras.Input(shape=(proc_dict.shape[1],), dtype='float32')

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

     # 加強製程參數處理部分
    y = layers.Dense(128, activation="relu")(inputs_process)
    y = layers.Dense(128, activation="relu")(y)  # 增加額外的Dense層
    y = layers.Attention()([y, y])  # 引入自注意力層增強特徵

    # 合並圖像和制程參數的路徑
    combined = layers.concatenate([x, y])
    combined = layers.Dense(64, activation="relu")(combined)  # 融合前進行進一步處理

    # 回歸輸出層
    outputs = layers.Dense(num_classes, activation="linear")(combined)

    # 建立和編譯模型
    model = keras.Model(inputs=[inputs_img, inputs_process], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # 根據需要調整學習率
                  loss='mean_squared_error', 
                  metrics=['mae'])

    return model

# 這裡隨便定義一下 proc_dict_scaled 和 num_classes 作為範例
proc_dict_scaled = np.random.rand(10, 5)  # 假設 proc_dict_scaled 是一個形狀為 (10, 5) 的數組
num_classes = 1

# 使用你的模型建立函數來創建模型
model = build_cvt_model(image_height, image_width, num_channels, proc_dict_scaled, num_classes)

# 使用 plot_model 函數畫出模型架構
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)