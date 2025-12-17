import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


from ase.io import read
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from dscribe.descriptors import SOAP, ACSF
import timeit
import logging
import matplotlib.pyplot as plt
import numpy as np
import umap

global_path = Path('/home/ebychkov/Teach/Phase-Parser/')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(global_path/ 'tests'/ 'app_UMAP.log'),
        logging.StreamHandler()  # также выводит в консоль
    ]
)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('umap').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING) 
logger = logging.getLogger(__name__)
logger.info('Отладка началась')

structure_file = global_path / 'data' / 'HfB2_5gr_p1.dat'
structure = read(structure_file, format='lammps-data')
unique_symbols = set(structure.symbols)
species_str = np.array(list(unique_symbols))
logger.info("Структура считана успешно")
species_str = ["Ti", "Zr", "Ta", "Hf", "Nb", "B"]
R_cut = 8
n_max = 8
l_max = 4
soap = SOAP(
    species=species_str,
    periodic=False,
    r_cut=R_cut,
    n_max=n_max,
    l_max=l_max,
    compression = {'mode':'mu2'})
local_enviroments = soap.create(structure, n_jobs = 1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(local_enviroments)
logger.info('Дискрипторы посчитаны')

def UMAP_working(df_scaled, n_jobs = -1):
    reducer = umap.UMAP(
    n_neighbors=300,
    n_components=10,  # Сжимаем до 10 измерений для кластеризации
    min_dist=0.0,
    metric='cosine',
    verbose=False,
    n_jobs = n_jobs# Для высокой размерности cosine часто лучше euclidean
    )
    embedding = reducer.fit_transform(df_scaled)

logger.info('Запускаем расчет для маленькой структуры')
n_tests = 10
cores_arr = np.array([1,2,4, 8, 12, 24])
time_arr = np.ones_like(cores_arr)

for i, n_jobs in enumerate(cores_arr):
    total_time = timeit.timeit(lambda: UMAP_working(df_scaled, n_jobs=n_jobs), number=n_tests)
    logger.debug(f'n_cores = {n_jobs}, total time = {total_time / n_tests}, Speedup = {total_time / n_tests/ time_arr[0]}')
    time_arr[i] = total_time/ n_tests
logger.info('Закончили расчет для маленькой структуры')
plt.plot(np.arange(1, 6), time_arr)
plt.xlabel("Number of cores")
plt.ylabel("Time, s")
plt.savefig(global_path/ 'tests' / 'check_time_UMAP.png')
plt.show()
logger.info('Сохранили файл расчет для маленькой структуры')
