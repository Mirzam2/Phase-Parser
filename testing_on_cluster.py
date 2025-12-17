import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os

# 1. Проверяем, передано ли название скрипта
if len(sys.argv) < 2:
    print("Ошибка: Не указано название запускаемого скрипта.")
    print(f"Использование: python {sys.argv[0]} <название_скрипта.py>")
    sys.exit(1)

# Берем название скрипта из аргументов
script_name = sys.argv[1]

# Путь к картинке (оставил жестким, как в оригинале)
image_path = "/home/ebychkov/Teach/HPPL/image.jpg"

cores_list = np.array([1, 2, 4, 8, 12, 24])
repeats = 5                

# 2. Формируем название для Excel-файла
# os.path.basename берет только имя файла (без путей), splitext убирает .py
base_name = os.path.splitext(os.path.basename(script_name))[0]
output_excel_name = f"{base_name}_test_results.xlsx"

results = []
for cores in cores_list:
    print(f"Testing {cores} cores (script: {script_name})", end=" ")

    current_times = []
    current_memory_peak_rank0_mb = []
    current_memory_peak_worker_avg_mb = []
    current_memory_total_cluster_mb = []
    for i in range(repeats):
        # 3. Вставляем script_name и image_path в команду
        cmd = f"module load mpi/openmpi-x86_64 && mpirun -n {cores} --oversubscribe python {script_name}"
        
        try:
            process = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            output_lines = process.stdout.strip().split('\n')
            json_line = next((line for line in output_lines if line.startswith('{')), None)
            if json_line:
                data = json.loads(json_line)
                t = data['time_sec']
                current_times.append(t)
                current_memory_peak_rank0_mb.append(data['memory_peak_rank0_mb'])
                current_memory_peak_worker_avg_mb.append(data['memory_peak_worker_avg_mb'])
                current_memory_total_cluster_mb.append(data['memory_total_cluster_mb'])
        except subprocess.CalledProcessError as e:
            # Можно раскомментировать для отладки, если что-то упадет
            # print(f"\nError: {e.stderr}")
            pass

    if current_times:
        avg_time = np.mean(current_times)
        std_dev = np.std(current_times)
        avg_mem_root = np.mean(current_memory_peak_rank0_mb)
        avg_mem_workers = np.mean(current_memory_peak_worker_avg_mb)
        avg_mem_total = np.mean(current_memory_total_cluster_mb)
        results.append({
            "cores": cores,
            "avg_time": avg_time,
            "std_dev": std_dev,
            "Mem root": avg_mem_root,
            "Memory workers": avg_mem_workers,
            "Memory total": avg_mem_total,
        })
        print(f" Done! Avg: {avg_time:.4f}s (±{std_dev:.4f})")
    else:
        print(" Failed (no successful runs)")

df = pd.DataFrame(results)
df.to_excel(output_excel_name)
print(f"Результаты сохранены в: {output_excel_name}")