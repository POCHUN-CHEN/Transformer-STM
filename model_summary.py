import tensorflow as tf
from tensorflow import keras

# 创建一个与原始模型相同架构的新模型
model = keras.Sequential([
    # 在此添加与原始模型相同的层
])

# 加载权重
model.load_weights('smaller_transformer_model_weights.h5')

# 打印模型摘要
model.summary()
