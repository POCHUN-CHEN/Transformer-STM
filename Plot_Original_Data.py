import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # 標準化製程參數

# 提取不同頻率
# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
# frequencies = ['50HZ_μa', '200HZ_μa', '400HZ_μa', '800HZ_μa']
frequencies = ['50HZ_μa']

# 定義範圍
group_start = 1
group_end = 10
piece_num_start = 1
piece_num_end = 5

# 定義其他相關範圍或常數
image_layers = 200  # 每顆影像的層數

# 定義圖像的高度、寬度和通道數
image_height = 128
image_width = 128
num_channels = 1

k_fold_splits = 5

# 讀取Excel文件中的標簽數據
excel_data = pd.read_excel('Circle_test.xlsx')
excel_process = pd.read_excel('Process_parameters.xlsx')


################################################################################
################################## 工件材料性質 ##################################
################################################################################

# 載入材料數據標簽
labels_dict = {}
for freq in frequencies:
    label_groups = []
    count = 0
    for i in range(group_start, group_end + 1):
        for j in range(piece_num_start, piece_num_end + 1):  # 每大組包含5小組
            labels = excel_data.loc[count, freq]
            label_groups.extend([labels] * image_layers)
            count += 1
        
    labels_dict[freq] = np.array(label_groups)  # 轉換為NumPy數組s


#################################################################################
#################################### 製程參數 ####################################
#################################################################################

# 載入製程參數
Process_parameters = ['氧濃度', '雷射掃描速度', '雷射功率', '線間距', '能量密度']
proc_dict = {}  # 儲存所有頻率全部大組製程參數
proc_dict_scaled = {}
for freq in frequencies:
    proc_groups = []  # 儲存全部大組製程參數
    for i in range(group_start, group_end + 1):
        group_procs = []  # 每大組的製程參數
        parameters_group = []
        for para in Process_parameters:
            parameters = excel_process.loc[i-1, para]
            parameters_group.append(parameters)

        for j in range(piece_num_start, piece_num_end + 1):  # 每大組包含5小組
            group_procs.extend([parameters_group] * image_layers)

        proc_groups.extend(group_procs)

    # 轉換為NumPy數組
    proc_dict[freq] = np.array(proc_groups)
    
    # 初始化 StandardScaler
    scaler = StandardScaler()
    
    # 標準化製程參數
    proc_dict_scaled[freq] = scaler.fit_transform(proc_dict[freq])


#################################################################################
#################################### 積層影像 ####################################
#################################################################################

# 載入圖像數據
image_groups = []

for group in range(group_start, group_end + 1):
    group_images = []
    for image_num in range(piece_num_start, piece_num_end + 1):
        folder_name = f'circle(340x345)/trail{group:01d}_{image_num:02d}'
        folder_path = f'data/{folder_name}/'

        image_group = []
        for i in range(image_layers):
            filename = f'{folder_path}/layer_{i + 1:02d}.jpg'
            image = cv2.imread(filename)
            image = cv2.resize(image, (image_width, image_height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image / 255.0  # 標準化
            image_group.append(image)

        group_images.extend(image_group)

    image_groups.extend(group_images)

# 轉換為NumPy數組
images = np.array(image_groups)


#################################################################################
#################################### 繪製資料 ####################################
#################################################################################


# 對於每個頻率進行模型載入、預測和評估
for freq in frequencies:
    
    # 列印結果
    print(f'Frequency: {freq}')


    # 生成圖片編號
    image_numbers = np.arange(1, len(images) + 1)


    #################################### 製程參數 ####################################    

    # # 定義不同的顏色
    # colors = ['red', 'green', 'blue', 'purple', 'orange']
    # # 定義製程參數的英文標籤ㄋ
    # parameter_labels = ['Oxygen Concentration', 'Laser Scanning Speed', 'Laser Power', 'Layer Spacing', 'Energy Density']

    # # 創建一個新的figure和axes
    # fig, ax1 = plt.subplots()

    # # 繪製標籤數據
    # ax1.set_xlabel('Image Number')
    # ax1.set_ylabel('Labels', color='tab:blue')
    # ax1.plot(image_numbers, labels_dict[freq], label='Labels', marker='o', color='tab:blue')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # # 創建第二個axes，共享相同的x軸
    # ax2 = ax1.twinx()

    # # 繪製每個製程參數的線
    # for i, label in enumerate(parameter_labels):
    #     ax2.plot(image_numbers, proc_dict_scaled[freq][:, i], label=label, marker='x', color=colors[i])

    # ax2.set_ylabel('Parameters', color='tab:red')
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    # # 合併兩個圖例
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()

    # # 使用bbox_to_anchor自定義圖例位置
    # ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)

    # # 添加標題
    # plt.title(f'Labels vs Parameters - {freq}')

    # # 顯示圖表
    # plt.show()



    #################################### 標籤值 ####################################
    # 計算每小組的平均值並創建一個與原始數據相同長度的平均值數組
    group_averages_repeated = np.empty(len(labels_dict[freq]))  # 創建一個空數組

    for i in range(group_start, group_end + 1):
        start_idx = (i - 1) * piece_num_end * image_layers
        end_idx = start_idx + piece_num_end * image_layers
        group_avg = np.mean(labels_dict[freq][start_idx:end_idx])
        group_averages_repeated[start_idx:end_idx] = group_avg  # 將這個範圍內的所有值設置為平均值

    # 繪製實際標籤值
    plt.plot(image_numbers, labels_dict[freq], label='Actual', marker='o')

    # 繪製每小組的平均值線
    plt.plot(image_numbers, group_averages_repeated, label='Group Average', color='red', linestyle='--')

    # 添加標籤和標題
    plt.xlabel('Image Number')
    plt.ylabel('Values')
    plt.title(f'Actual vs Group Average - {freq}')
    plt.legend()

    # 顯示圖表
    plt.show()