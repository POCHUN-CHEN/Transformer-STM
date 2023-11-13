import os

# 獲取當前工作目錄
folder_path = os.getcwd()

# 定義新的資料夾順序映射
new_order_mapping = {
    'item1': 'item36',
    'item2': 'item37',
    'item3': 'item38',
    'item4': 'item39',
    'item5': 'item40',
    'item6': 'item31',
    'item7': 'item32',
    'item8': 'item33',
    'item9': 'item34',
    'item10': 'item35',
    'item11': 'item45',
    'item12': 'item50',
    'item13': 'item26',
    'item14': 'item27',
    'item15': 'item28',
    'item16': 'item29',
    'item17': 'item30',
    'item18': 'item44',
    'item19': 'item49',
    'item20': 'item21',
    'item21': 'item22',
    'item22': 'item23',
    'item23': 'item24',
    'item24': 'item25',
    'item25': 'item43',
    'item26': 'item48',
    'item27': 'item16',
    'item28': 'item17',
    'item29': 'item18',
    'item30': 'item19',
    'item31': 'item20',
    'item32': 'item42',
    'item33': 'item47',
    'item34': 'item11',
    'item35': 'item12',
    'item36': 'item13',
    'item37': 'item14',
    'item38': 'item15',
    'item39': 'item41',
    'item40': 'item46',
    'item41': 'item6',
    'item42': 'item7',
    'item43': 'item8',
    'item44': 'item9',
    'item45': 'item10',
    'item46': 'item1',
    'item47': 'item2',
    'item48': 'item3',
    'item49': 'item4',
    'item50': 'item5',
}

# 遍歷文件夾
for old_folder_name, new_folder_name in new_order_mapping.items():
    old_folder_path = os.path.join(folder_path, old_folder_name)
    new_folder_path = os.path.join(folder_path, new_folder_name)

    # 如果目標資料夾已經存在，則添加後綴數字，直到找到一個可用的名稱
    while os.path.exists(new_folder_path):
        new_folder_name += "_1"
        new_folder_path = os.path.join(folder_path, new_folder_name)

     # 使用 os.rename 進行重命名
    os.rename(old_folder_path, new_folder_path)

    print(f'Renamed folder {old_folder_name} to {new_folder_name}')

# 移除所有資料夾名稱中的後綴數字
for folder_name in os.listdir(folder_path):
    if folder_name.startswith('item'):
        new_folder_name = folder_name.split('_')[0]
        os.rename(os.path.join(folder_path, folder_name), os.path.join(folder_path, new_folder_name))

