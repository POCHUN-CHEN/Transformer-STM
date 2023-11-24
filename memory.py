import psutil
import os
import time

while True:
    # 清除終端機輸出
    os.system('cls' if os.name == 'nt' else 'clear')

    # 獲取記憶體使用情況
    memory = psutil.virtual_memory()

    # 列印詳細的記憶體資訊
    print("總記憶體 (Total):", memory.total, "bytes")
    print("可用記憶體 (Available):", memory.available, "bytes")
    print("已使用記憶體 (Used):", memory.used, "bytes")
    print("記憶體使用率 (Usage):", memory.percent, "%")

    # 暫停一秒
    time.sleep(1)