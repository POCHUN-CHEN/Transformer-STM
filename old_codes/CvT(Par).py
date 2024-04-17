import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard

# 動態記憶體分配
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 提取不同頻率
frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br']
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv']
# frequencies = ['800HZ_Pcv']

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

# 批次大小
batch_size = 128

# 設置 epoch 數目
train_epochs = 1000

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')
excel_process = pd.read_excel('Process_parameters.xlsx')

# 定義 ConvProjection 層
class ConvProjection(layers.Layer):
    def __init__(self, dim, kernel_size, strides, padding, method='dw_bn', name=None):
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
        else:
            raise ValueError(f"Unknown method: {method}")
        
    def call(self, inputs):
        if self.method == 'dw_bn':
            x = self.conv(inputs)
            x = self.bn(x)
        elif self.method == 'avg':
            x = self.avg_pool(inputs)
        
        return x

# 定義 ConvAttention 層
class ConvAttention(layers.Layer):
    def __init__(self, dim, num_heads, kernel_size, strides, padding, qkv_method='dw_bn', name=None):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.qkv_method = qkv_method

        self.q_proj = ConvProjection(dim, kernel_size, strides, padding, method=qkv_method, name='q_proj')
        self.k_proj = ConvProjection(dim, kernel_size, strides, padding, method=qkv_method, name='k_proj')
        self.v_proj = ConvProjection(dim, kernel_size, strides, padding, method=qkv_method, name='v_proj')

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.reshape = layers.Reshape((-1, dim))

    def call(self, inputs):
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)

        _, h, w, _ = q.shape
        q = self.reshape(q)
        k = self.reshape(k)
        v = self.reshape(v)

        attn_output = self.attention(q, v, k)
        attn_output = layers.Reshape((h, w, self.dim))(attn_output)
        
        return attn_output

# 定義 ConvTransformerBlock 層
class ConvTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, kernel_size, strides, padding, qkv_method='dw_bn', name=None):
        super().__init__(name=name)
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.qkv_method = qkv_method

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = ConvAttention(dim, num_heads, kernel_size, strides, padding, qkv_method=qkv_method)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(dim * 4, activation="relu"),
            layers.Dense(dim),
        ])

    def call(self, inputs):
        x = self.norm1(inputs)
        x = self.attn(x)
        x = x + inputs

        y = self.norm2(x)
        y = self.ffn(y)
        return x + y

# 定義 ConvBlock 層
class ConvBlock(layers.Layer):
    def __init__(self, filters, kernel_size, strides, name=None):
        super().__init__(name=name)
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', activation='relu')
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        return x

# 建立CvT模型
def create_cvt_model(image_height, image_width, num_channels, proc_dim, num_classes):
    image_inputs = keras.Input(shape=(image_height, image_width, num_channels))
    proc_inputs = keras.Input(shape=(proc_dim,))

    x = image_inputs

    # Stage 1
    x = ConvBlock(64, 7, 4, name='stage1_conv')(x)
    x = ConvTransformerBlock(64, num_heads=1, kernel_size=3, strides=1, padding='same', qkv_method='dw_bn', name='stage1_transformer')(x)

    # Stage 2
    x = ConvBlock(128, 3, 2, name='stage2_conv')(x)
    x = ConvTransformerBlock(128, num_heads=2, kernel_size=3, strides=1, padding='same', qkv_method='dw_bn', name='stage2_transformer')(x)

    # Stage 3
    x = ConvBlock(256, 3, 2, name='stage3_conv')(x)
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

# 定義學習率調整函數
def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 50 == 0:
        return lr * 0.8
    return lr

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

# 訓練模型並保存結果
def train_and_save_model(freq, labels_dict, proc_dict_scaled, images, valid_dict, count):
    # 定義訓練集和驗證集
    x_train, y_train, proc_train = [], [], []
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
        else:
            x_train.extend(images[index:index + image_layers])
            y_train.extend(labels_dict[index:index + image_layers])
            proc_train.extend(proc_dict_scaled[index:index + image_layers])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    proc_train = np.array(proc_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    proc_val = np.array(proc_val)

    # 創建模型
    model = create_cvt_model(image_height, image_width, num_channels, proc_dict_scaled.shape[1], num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mean_squared_error',
                  metrics=['mae'])

    # 學習率調整
    lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)

    # 創建 TensorBoard 回調
    tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)

    # 訓練模型
    model.fit([x_train, proc_train], y_train, epochs=train_epochs, batch_size=batch_size,
              validation_data=([x_val, proc_val], y_val), callbacks=[tensorboard_callback, lr_callback])

    # 保存模型權重
    model.save_weights(f'Weight/Images & Parameters/cvt_model_weights_{freq}.h5')

    # 初始化 DataFrame 以存儲記錄
    records = pd.DataFrame(model.history.history)
    records.insert(0, 'epoch', range(1, len(records) + 1))
    records.to_excel(f'Records/Images & Parameters/cvt_records_{freq}.xlsx', index=False)

# 主程序
for freq in frequencies:
    print(f"Training for frequency {freq}")

    labels_dict, proc_dict_scaled, images, valid_dict, count = preprocess_data(excel_data, excel_process, group_start, group_end, piece_num_start, piece_num_end, image_layers, image_height, image_width)

    train_and_save_model(freq, labels_dict, proc_dict_scaled, images, valid_dict, count)

