    
import os
# Ограничиваем количество потоков для библиотек линейной алгебры до 1
# Это нужно делать ДО импорта numpy и dscribe
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

  
from ase.io import read
from pathlib import Path
from dscribe.descriptors import SOAP, ACSF
import timeit
import logging
import matplotlib.pyplot as plt
import numpy as np
global_path = Path('/home/ebychkov/Teach/HPPL_project/')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(global_path/ 'tests'/ 'app_fix.log'),
        logging.StreamHandler()  # также выводит в консоль
    ]
)
logger = logging.getLogger(__name__)
logger.info('Отладка началась')

structure_file = global_path / 'data' / 'HfB2_5gr_p1.dat'
structure = read(structure_file, format='lammps-data')
logger.info("Структура считана успешно")
species_str = ["Ti", "Zr", "Ta", "Hf", "Nb", "B"]
R_cut = 8
n_max = 6
l_max = 3
soap = SOAP(
    species=species_str,
    periodic=True,
    r_cut=R_cut,
    n_max=n_max,
    l_max=l_max,
    compression = {'mode':'mu2'})
acsf = ACSF(
    species=species_str,        # Список всех возможных элементов в твоем датасете
    r_cut=R_cut,                  # Радиус сферы окружения (в Ангстремах)
    g2_params=[[1, 2], [1, 4]], # Параметры радиальных функций [[eta, Rs], ...]
    g4_params=[[1, 2, 1]],      # Параметры угловых функций [[eta, zeta, lambda], ...]
    periodic=True,             # Важно: False для молекул, True для кристаллов
    sparse=False                # Возвращать обычный numpy массив (быстрее для GPU/Torch)
)

logger.info('Запускаем расчет для маленькой структуры')
n_tests = 5
time_arr = []
for n_jobs in range(1, 24):
    logger.debug(f'Small {n_jobs}')
    total_time = timeit.timeit(lambda: soap.create(structure, n_jobs=n_jobs), number=n_tests)
    time_arr.append(total_time/ n_tests)
logger.info('Закончили расчет для маленькой структуры')
plt.plot(np.arange(1, 24), time_arr)
plt.xlabel("Number of cores")
plt.ylabel("Time, s")
plt.savefig(global_path/ 'tests' / 'check_time_small_str.png')
plt.show()
logger.info('Сохранили файл расчет для маленькой структуры')
supercell = structure * (5, 5, 4)
n_tests = 3
time_arr_big = []
n_cores_big = [1, 2, 4, 8, 12, 24]
logger.info('Запускаем расчет для большой структуры')
for n_jobs in n_cores_big:
    logger.debug(f'Big {n_jobs}')
    total_time = timeit.timeit(lambda: soap.create(supercell, n_jobs=n_jobs), number=n_tests)
    time_arr_big.append(total_time/ n_tests)
logger.info('Закончили расчет для большой структуры')
plt.plot(n_cores_big, time_arr_big)
plt.xlabel("Number of cores")
plt.ylabel("Time, s")
plt.savefig(global_path/ 'tests' / 'check_time_big_str.png')
plt.show()
