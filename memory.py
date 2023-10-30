import psutil

# 获取内存使用情况
memory = psutil.virtual_memory()

# 打印内存信息
print("总内存 (Total):", memory.total, "bytes")
print("可用内存 (Available):", memory.available, "bytes")
print("已使用内存 (Used):", memory.used, "bytes")
print("内存使用率 (Usage):", memory.percent, "%")