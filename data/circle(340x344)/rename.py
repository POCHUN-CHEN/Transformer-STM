import os

# 獲取當前工作目錄
folder_path = os.getcwd()


# 遍歷文件夾
for item_number in range(1, 51):
    old_folder_name = f'item{item_number:01d}'
    
    # 計算新的文件夾名
    trail_number = (item_number - 1) // 5 + 1
    new_folder_name = f'trail{trail_number:01d}_{(item_number - 1) % 5 + 1:02d}'
    
    # 構建完整的文件夾路徑
    old_folder_path = os.path.join(folder_path, old_folder_name)
    new_folder_path = os.path.join(folder_path, new_folder_name)
    
    # 使用 os.rename 進行重命名
    os.rename(old_folder_path, new_folder_path)
    
    print(f'Renamed folder {old_folder_name} to {new_folder_name}')
