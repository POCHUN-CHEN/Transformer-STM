import openpyxl
import matplotlib.pyplot as plt
import numpy as np


# frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']
frequencies = ['50HZ_μa', '200HZ_μa', '400HZ_μa', '800HZ_μa']
# frequencies = ['50HZ_μa']

for freq in frequencies:
    for fold in range(5):
        # 從Excel文件讀取數據
        # wb = openpyxl.load_workbook(f'bayesian_conv_transformer_par_model_weights_{freq}_fold_{fold+1}.xlsx')
        wb = openpyxl.load_workbook(f'bayesian_par_model_weights_{freq}_fold_{fold+1}.xlsx')
        # wb = openpyxl.load_workbook(f'bayesian_conv_transformer_par_records_{freq}.xlsx')
        ws = wb.active

        # 提取數據並存儲在字典中，以便按列訪問
        headers = ["epoch", "loss", "mae", "val_loss", "val_mae"]
        data = {header: [] for header in headers}

        for row in ws.iter_rows(min_row=2, values_only=True):
            for header, value in zip(headers, row):
                data[header].append(value)

        # 創建圖表
        plt.figure(figsize=(10, 6))

        # 繪制每個物理量的圖線
        epochs = [int(epoch) for epoch in data["epoch"]]  # 將 Epoch 值轉換為整數
        for header in headers[1:]:  # 從第二列開始，忽略Epoch列
            plt.plot(epochs, data[header], label=header)

        # 添加圖例和標簽
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Metrics Over epochs")

        # 設定 x 軸刻度間隔，這裡設為每 10 個 epoch 顯示一個刻度
        plt.xticks(np.arange(min(epochs)-1, max(epochs), 50))

        # 設定 X 軸範圍
        plt.xlim(0, 200)  # 将Y轴范围限制在0到100之间

        # 設定 Y 軸範圍
        plt.ylim(0, 2000)  # 将Y轴范围限制在0到100之间

        # 顯示圖表
        plt.grid(True)
        plt.show()
