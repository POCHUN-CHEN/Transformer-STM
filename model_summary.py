import tensorflow as tf
from tensorflow import keras

# 創建一個與原始模型相同架構的新模型
model = keras.Sequential([
    # 在此添加與原始模型相同的層
])

# 加載權重
model.load_weights('smaller_transformer_model_weights.h5')

# 打印模型摘要
model.summary()
