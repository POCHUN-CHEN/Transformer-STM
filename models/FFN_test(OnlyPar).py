import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 提取不同頻率
frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']

# frequencies = ['800HZ_Hc']

# 定義範圍
group_start = 1
group_end = 40
piece_num_start = 1
piece_num_end = 5

# 定義其他相關範圍或常數
image_layers = 200  # 每顆影像的層數

num_classes = 1  # 回歸任務

# 獲取當前腳本所在的目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

# 確定文件的相對路徑
Circle_test_path = os.path.join(script_dir, '../Excel/Processed_Circle_test.xlsx')
Process_parameters_path = os.path.join(script_dir, '../Excel/Process_parameters.xlsx')

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel(Circle_test_path)
excel_process = pd.read_excel(Process_parameters_path)

# 建立CvT模型
def create_cvt_model(proc_dim, num_classes):
    proc_inputs = keras.Input(shape=(proc_dim,))
    
    # 處理製程參數
    proc_features = layers.Dense(256, activation='relu')(proc_inputs)
    proc_features = layers.Dense(256, activation='relu')(proc_features)

    # 輸出層
    outputs = layers.Dense(num_classes, activation='linear')(proc_features)

    # 創建模型
    model = keras.Model(inputs=[proc_inputs], outputs=outputs)
    return model

# 數據預處理函數
def preprocess_data(excel_data, excel_process, group_start, group_end, piece_num_start, piece_num_end, image_layers):
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

    return labels_dict, proc_dict_scaled, valid_dict, count

# 測試模型並保存結果
def test_and_save_results(freq, labels_dict, proc_dict_scaled, valid_dict, count):
    # 定義驗證集
    y_val, proc_val = [], []

    first_valid_indices_per_group = []
    for d in range(0, count, 5):
        for j in range(d, d + 5):
            if j in valid_dict:
                first_valid_indices_per_group.append(j)
                break

    for i in range(len(valid_dict)):
        index = i * image_layers

        if valid_dict[i] in first_valid_indices_per_group:
            y_val.extend(labels_dict[index:index + image_layers])
            proc_val.extend(proc_dict_scaled[index:index + image_layers])

    y_val = np.array(y_val)
    proc_val = np.array(proc_val)

    # 構建模型
    model = create_cvt_model(proc_dict_scaled.shape[1], num_classes)

    # 載入模型權重
    model.load_weights(os.path.join(script_dir, f'../Result/Weight/Parameters/Vit_model_weights_{freq}.h5'))

    # 檢查並建立資料夾
    Plots_folder = os.path.join(script_dir, '../Result/Plots/Parameters')
    
    if not os.path.exists(Plots_folder):
        os.makedirs(Plots_folder)

    # 進行預測
    predictions = model.predict([proc_val])

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
    plt.savefig(os.path.join(Plots_folder, f'Vit_R^2_{freq}.png'))
    plt.clf()

    # 繪製實際值與預測值的線圖
    image_numbers = np.arange(1, len(predictions) + 1)
    plt.plot(image_numbers, y_val, label='Actual', marker='o')
    plt.plot(image_numbers, predictions.flatten(), label='Predicted', marker='x')
    plt.xlabel('Image Number')
    plt.ylabel('Values')
    plt.title(f'Actual vs Predicted - {freq}')
    plt.legend()
    plt.savefig(os.path.join(Plots_folder, f'Vit_Actual_vs_Predicted_{freq}.png'))
    plt.clf()

# 主程序
if __name__ == "__main__":
    for freq in frequencies:
        print(f"Testing for frequency {freq}")

        labels_dict, proc_dict_scaled, valid_dict, count = preprocess_data(excel_data, excel_process, group_start, group_end, piece_num_start, piece_num_end, image_layers)

        test_and_save_results(freq, labels_dict, proc_dict_scaled, valid_dict, count)