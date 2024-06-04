import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

# 提取不同頻率
frequencies = ['50HZ_Bm', '50HZ_Hc', '50HZ_μa', '50HZ_Br', '50HZ_Pcv', '200HZ_Bm', '200HZ_Hc', '200HZ_μa', '200HZ_Br', '200HZ_Pcv', '400HZ_Bm', '400HZ_Hc', '400HZ_μa', '400HZ_Br', '400HZ_Pcv', '800HZ_Bm', '800HZ_Hc', '800HZ_μa', '800HZ_Br', '800HZ_Pcv']

# 獲取當前腳本所在的目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

def collect_data_and_plot(models, properties, frequencies):
    results = {prop: {model: [] for model in models} for prop in properties}
    base_frequencies = list(OrderedDict.fromkeys(f.split('_')[0] for f in frequencies))

    # 收集數據
    for prop in properties:
        for freq in base_frequencies:
            freq_prop = f"{freq}_{prop}"
            # 特定模型（CvT）的文件讀取
            filecvt_position = os.path.join(script_dir, f'../Result/Excel/Images & Parameters/Predictions_Metrics_{freq_prop}.xlsx')
            data_cvt = pd.read_excel(filecvt_position, sheet_name='Sheet1')
            r2_score_cvt = data_cvt['R2 Score'].iloc[0]
            results[prop]['CvT'].append(r2_score_cvt)

            # 其他模型的文件讀取
            for model in models:
                if model != 'CvT':  # 確保不重複處理 CvT
                    file_path = os.path.join(script_dir, f'../Result/Excel/glcm/{prop}_{model}.xlsx')
                    data = pd.read_excel(file_path, sheet_name=freq_prop)
                    r2_score = data['R2 score'].iloc[0]
                    results[prop][model].append(r2_score)

    # 繪圖
    for prop, model_scores in results.items():
        plt.figure(figsize=(10, 8))
        # 繪製 CvT 數據
        plt.plot(base_frequencies, model_scores['CvT'], marker='o', label="CvT", color='red')
        # 繪製其他模型數據
        for model, scores in model_scores.items():
            if model != 'CvT':
                plt.plot(base_frequencies, scores, marker='o', label=model)
        plt.title(f'R² Score for {prop}')
        plt.xlabel('Frequency')
        plt.ylabel('R² Score')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=6)
        plt.grid(True)
        # plt.show()

        # 保存圖片
        compare_folder = os.path.join(script_dir, '../Result/Plots/Compare_R')

        # 檢查並建立儲存資料夾
        if not os.path.exists(compare_folder):
            os.makedirs(compare_folder)
        
        # 儲存圖片
        save_path = os.path.join(compare_folder, f'{prop}.png')
        plt.savefig(save_path)
        plt.close()

        print(f"Plot for {prop} saved.")

# 模型和屬性定義
models = ['lightgbm', 'xgboost', 'svr', 'logistic', 'linear', 'CvT']  # 包含 CvT 在內的模型列表
properties = ['Bm', 'Hc', 'μa', 'Br', 'Pcv']

if __name__ == "__main__":
    collect_data_and_plot(models, properties, frequencies)