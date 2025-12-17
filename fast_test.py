    
import os
# Вставляем это в САМОЕ НАЧАЛО скрипта, до импорта numpy/dscribe
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"



import time
import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from dscribe.descriptors import SOAP

def run_benchmark():
    # --- 2. ГЕНЕРАЦИЯ ДАННЫХ ---
    print("Генерация данных...")
    # Создаем одну структуру среднего размера (например, 108 атомов)
    # Al (fcc) 4 атома * 3*3*3 = 108 атомов
    base_atom = bulk('Al', 'fcc', a=4.05)
    structure_medium = base_atom * (20, 10, 10) 
    
    # СОЗДАЕМ СПИСОК СТРУКТУР
    # 96 штук. Это даст хорошую загрузку:
    # При 1 ядре: 96 задач очереди
    # При 24 ядрах: по 4 задачи на брата
    n_structures = 96
    structure_list = [structure_medium.copy() for _ in range(n_structures)]
    
    print(f"Список создан: {len(structure_list)} структур по {len(structure_medium)} атомов.")

    # --- 3. НАСТРОЙКА SOAP ---
    # Параметры "средней тяжести", чтобы расчет не был мгновенным
    soap = SOAP(
        species=["Al"],
        periodic=True,
        r_cut=6.0,
        n_max=8,
        l_max=6,
        sparse=False, # Плотные матрицы быстрее считаются, если влезают в память
        compression={'mode': 'mu2'}
    )

    # --- 4. ТЕСТ ---
    cores_list = [1, 2, 4, 6, 12, 24] # Список ядер для проверки
    times = []

    print(f"{'Cores':<10} | {'Time (s)':<10} | {'Speedup':<10}")
    print("-" * 35)

    base_time = None

    for n_jobs in cores_list:
        start_time = time.perf_counter()
        
        # Самое важное: передаем СПИСОК (structure_list)
        soap.create(structure_list, n_jobs=n_jobs)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        times.append(elapsed)
        
        # Считаем ускорение (Speedup)
        if base_time is None:
            base_time = elapsed
            speedup = 1.0
        else:
            speedup = base_time / elapsed

        print(f"{n_jobs:<10} | {elapsed:<10.4f} | {speedup:<10.2f}x")

    # --- 5. ГРАФИК ---
    plt.figure(figsize=(10, 6))
    plt.plot(cores_list, times, 'o-', linewidth=2, label="Measured Time")
    
    # Рисуем идеальную кривую (1/x) для сравнения
    ideal_times = [times[0] / n for n in cores_list]
    plt.plot(cores_list, ideal_times, '--', label="Ideal Scaling (1/x)", alpha=0.6)

    plt.title(f"DScribe SOAP Scaling ({n_structures} structures list)")
    plt.xlabel("Number of Cores (n_jobs)")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()
    
    # Сохраняем и показываем
    plt.savefig("final_benchmark.png")
    print("\nГрафик сохранен как final_benchmark.png")
    plt.show()

if __name__ == "__main__":
    run_benchmark()