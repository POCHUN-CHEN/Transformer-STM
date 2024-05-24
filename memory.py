import psutil
import os
import time
import subprocess

def get_gpu_memory():
    # 使用 subprocess 執行 nvidia-smi 指令
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)

    # 解析輸出結果
    gpu_memory_info = result.stdout.strip().split(', ')
    
    # 將資訊轉換成字典
    gpu_memory = {
        'used': int(gpu_memory_info[0]),
        'free': int(gpu_memory_info[1]),
        'total': int(gpu_memory_info[2])
    }

    return gpu_memory

def get_gpu_utilization():
    # 使用 subprocess 執行 nvidia-smi 指令
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, text=True)

    # 解析輸出結果
    gpu_utilization = float(result.stdout.strip())

    return gpu_utilization


while True:
    # 清除終端機輸出
    os.system('cls' if os.name == 'nt' else 'clear')

    # 獲取CPU使用情況
    cpu_usage = psutil.cpu_percent()

    # 列印CPU使用情況
    print("CPU 使用率 (Usage):", cpu_usage, "%")

    # 獲取系統記憶體使用情況
    system_memory = psutil.virtual_memory()

    # 列印系統記憶體資訊
    print("\nMemory 使用情況:")
    print("Memory (Total):", round(system_memory.total / (1024 ** 3), 2), "GB")
    print("Memory (Used):", round(system_memory.used / (1024 ** 3), 2), "GB")
    print("Memory (Free):", round(system_memory.available / (1024 ** 3), 2), "GB")
    print("Memory 使用率 (Usage):", system_memory.percent, "%")


    # 獲取顯卡計算資源使用率
    gpu_utilization = get_gpu_utilization()

    # 列印顯卡計算資源使用率
    print("\nGPU 使用率 (Usage):", gpu_utilization, "%")

    # 獲取顯卡記憶體使用情況
    gpu_memory = get_gpu_memory()

    # 列印顯卡記憶體資訊
    print("\n顯卡記憶體使用情況:")
    print("顯卡總記憶體 (Total):", round(gpu_memory['total'] / 1024, 2), "GB")
    print("顯卡已使用記憶體 (Used):", round(gpu_memory['used'] / 1024, 2), "GB")
    print("顯卡可用記憶體 (Free):", round(gpu_memory['free'] / 1024, 2), "GB")
    print("顯卡記憶體使用率 (Usage):", round(gpu_memory['used'] / gpu_memory['total']*100, 2), "%")

    # 暫停一秒
    time.sleep(1)
