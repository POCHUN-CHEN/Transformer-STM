import openpyxl
import matplotlib.pyplot as plt
import numpy as np

# 從Excel文件讀取數據
wb = openpyxl.load_workbook('output.xlsx')
ws = wb.active

# 提取數據並存儲在字典中，以便按列訪問
headers = ["Epoch", "Loss", "MAE", "Val_Loss", "Val_MAE"]
data = {header: [] for header in headers}

for row in ws.iter_rows(min_row=2, values_only=True):
    for header, value in zip(headers, row):
        data[header].append(value)

# 創建圖表
plt.figure(figsize=(10, 6))

# 繪制每個物理量的圖線
epochs = [int(epoch) for epoch in data["Epoch"]]  # 將 Epoch 值轉換為整數
for header in headers[1:]:  # 從第二列開始，忽略Epoch列
    plt.plot(epochs, data[header], label=header)

# 添加圖例和標簽
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Metrics Over Epochs")

# 設定 x 軸刻度間隔，這裡設為每 10 個 Epoch 顯示一個刻度
plt.xticks(np.arange(min(epochs)-1, max(epochs), 50))

# 設定 Y 軸範圍
plt.ylim(0, 10)  # 将Y轴范围限制在0到100之间

# 顯示圖表
plt.grid(True)
plt.show()
