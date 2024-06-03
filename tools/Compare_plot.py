import pandas as pd
import matplotlib.pyplot as plt
import os

# 提取不同頻率
frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']

# 獲取當前腳本所在的目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

def plot_predictions(file1, file2, freq):
    # 讀取第一個Excel文件的數據
    data1 = pd.read_excel(file1)
    predictions1 = data1['Predictions']
    actual1 = data1['Actual']
    r21 = data1['R2 Score'].iloc[0]
    mse1 = data1['MSE'].iloc[0]
    mae1 = data1['MAE'].iloc[0]

    # 讀取第二個Excel文件中對應頻率的sheet
    data2 = pd.read_excel(file2, sheet_name=freq)
    predictions2 = data2['prediction']
    actual2 = data2['true']
    r22 = data2['R2 score'].iloc[0]
    mse2 = data2['MSE'].iloc[0]
    mae2 = data2['MAE'].iloc[0]

    # 繪製散點圖
    plt.figure(figsize=(10, 8))
    plt.scatter(actual2, predictions2, color='orange',  label='LightGBM', alpha=0.8, s=10)
    plt.scatter(actual1, predictions1, color='blue', label='Cvt', alpha=0.4, s=3)
    
    # 添加圖例、標題和軸標籤
    plt.title(f'Actual vs Predicted - {freq}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    # 在圖上添加 R^2, MSE, MAE 文字
    xlim = plt.xlim()
    ylim = plt.ylim()
    text_x = xlim[0] + (xlim[1] - xlim[0]) * 0.019  # 計算x軸上的文本位置
    text_y = ylim[1] - (ylim[1] - ylim[0]) * 0.1  # 計算y軸上的文本位置

    plt.text(text_x, ylim[1] - (ylim[1] - ylim[0]) * 0.2, f'LightGBM\nR2: {r22:.2f}\nMSE: {mse2:.2f}\nMAE: {mae2:.2f}',
         fontsize=9, color='orange',
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='orange', boxstyle='square,pad=0.5'))
    plt.text(text_x, ylim[1] - (ylim[1] - ylim[0]) * 0.33, f'Cvt\nR2: {r21:.2f}\nMSE: {mse1:.2f}\nMAE: {mae1:.2f}',
         fontsize=9, color='blue',
         bbox=dict(facecolor='white', alpha=0.5, edgecolor='blue', boxstyle='square,pad=0.5'))

    

    # 保存圖片
    compare_folder = os.path.join(script_dir, '../Result/Plots/Compare')

    # 檢查並建立儲存資料夾
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)
    
    # 儲存圖片
    save_path = os.path.join(compare_folder, f'actual_vs_predicted_{freq}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Plot for {freq} saved.")

# 主程序
if __name__ == "__main__":
    for freq in frequencies:
        # 空列表來存儲分割後的頻率和屬性
        hz = []
        proper = []

        parts = freq.split('_')  # 使用'_'來分割字符串
        # hz = parts[0] # 頻率部分
        proper = parts[1] # 屬性部分
        
        file1_position = os.path.join(script_dir, f'../Result/Excel/Images & Parameters/Predictions_Metrics_{freq}.xlsx')
        file2_position = os.path.join(script_dir, f'../Result/Excel/glcm/{proper}_lightgbm.xlsx')
        
        plot_predictions(file1_position, file2_position, freq)
