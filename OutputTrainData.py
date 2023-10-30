import openpyxl

# 創建一個新的Excel工作簿和工作表
wb = openpyxl.Workbook()
ws = wb.active

# 設置列標題
headers = ["Epoch", "Loss", "MAE", "Val_Loss", "Val_MAE"]
ws.append(headers)

# 讀取文本文件
with open('terminal_epoch_data.txt', 'r') as file:
    lines = file.readlines()

data = []

# 用於跟蹤當前的Epoch
current_epoch = None

# 逐行讀取文本文件並提取數據
for line in lines:
    # 如果行包含"Epoch"，則提取Epoch數值
    if line.startswith("Epoch "):
        parts = line.split()
        if len(parts) >= 2:
            current_epoch = parts[1].split('/')[0]
    # 如果行包含"loss"，則提取相關物理參數
    elif "loss:" in line:
        parts = line.split()
        if len(parts) >= 6:
            loss = float(parts[parts.index("loss:") + 1])  # 將文本轉換為浮點數
            mae_raw = parts[parts.index("mae:") + 1]  # 提取原始的mae字符串
            mae = float(mae_raw.split('/')[0])  # 提取mae的數字部分
            val_loss = float(parts[parts.index("val_loss:") + 1])
            val_mae_raw = parts[parts.index("val_mae:") + 1]  # 提取原始的val_mae字符串
            val_mae = float(val_mae_raw.split('/')[0])  # 提取val_mae的數字部分
            
            data.append([current_epoch, loss, mae, val_loss, val_mae])

# 將數據添加到Excel工作表中
for row in data:
    ws.append(row)

# 保存Excel工作簿
wb.save('output.xlsx')

print("數據已保存到output.xlsx")
