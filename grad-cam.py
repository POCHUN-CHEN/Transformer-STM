import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras_tuner as kt
from PIL import Image

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

# 定義的Convolution Transformer模型
def build_model(hp):
    inputs = keras.Input(shape=(image_height, image_width, num_channels))

    # Convolutional Blocks
    x = keras.layers.Conv2D(hp.Int('conv_1_filter', min_value=32, max_value=128, step=32), 
                            (3, 3), activation='relu')(inputs)
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
    outputs = keras.layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error', metrics=['mae'])
    return model

# 創建與原始搜索相同配置的Keras Tuner實例
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_mae',
    max_trials=10,
    num_initial_points=2,
    directory='my_dir',
    project_name=f'bayesian_opt_conv_transformer_50HZ_μa'
)

# 重新加載最佳超參數
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# best_hps = tuner.get_best_hyperparameters()
# if best_hps:
#     print("找到最佳超參數。")
# else:
#     print("沒有找到最佳超參數，請檢查超參數調整過程。")

# 構建模型
model = build_model(best_hps)

# 載入模型權重
model.load_weights(f'Weight/bayesian_conv_transformer_model_weights_50HZ_μa.h5')

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

# 加載並預處理圖像
img_path = 'Test_images/layer_15_6_1.jpg'  # 替換為您的圖像路徑
img = image.load_img(img_path, target_size=(128, 128), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# 生成 Grad-CAM 熱力圖
last_conv_layer_name = 'conv2d_2'  # 使用模型的最後一個卷積層名稱
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# 將熱力圖轉換為8位整數格式
heatmap = np.uint8(255 * heatmap)

# 應用顏色映射
heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 將 PIL 圖像轉換為 NumPy 數組
img_np = np.array(img)

# 將單通道灰度圖像轉換為三通道圖像
img_np_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

# 將熱力圖調整為原始圖像的大小並應用到原圖
heatmap_resized = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]))
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