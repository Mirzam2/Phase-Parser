from mpi4py import MPI
import numpy as np
import sys
import os
import json
import resource

# Инициализация базового коммуникатора
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    from ase import io, Atoms
    from scipy.spatial import cKDTree
except ImportError as e:
    # print(f"[{rank}] ERROR: Missing library! {e}", flush=True)
    sys.exit(1)

comm.Barrier()
t_start = MPI.Wtime()

# ==========================================
# ПАРАМЕТРЫ
# ==========================================
FILENAME = "data/HfB2_X.dat" 
CUTOFF = 3.5 

# ==========================================
# 1. НАСТРОЙКА 3D ДЕКОМПОЗИЦИИ
# ==========================================

# 1. Вычисляем оптимальное разбиение и превращаем в NUMPY массив
dims = np.array(MPI.Compute_dims(size, 3))

# Создаем Декартов коммуникатор
cart_comm = comm.Create_cart(dims.tolist(), periods=[True, True, True])

# 2. Получаем координаты и превращаем в NUMPY массив (ВАЖНО!)
my_coords = np.array(cart_comm.Get_coords(rank))

if rank == 0:
    pass
    # print(f"Total ranks: {size}. Grid decomposition: {dims}", flush=True)

# ==========================================
# 2. ЧТЕНИЕ И РАСПРЕДЕЛЕНИЕ АТОМОВ
# ==========================================

if not os.path.exists(FILENAME):
    if rank == 0: print(f"ERROR: File {FILENAME} not found")
    sys.exit(1)

full_system = io.read(FILENAME, format='lammps-data', atom_style='atomic')
cell = full_system.get_cell()
pbc = full_system.get_pbc()

# Размеры ящика
L = np.array([cell[0][0], cell[1][1], cell[2][2]])

# Вычисляем границы локального домена
# Теперь dims - это массив numpy, деление пройдет корректно
domain_len = L / dims

# Начало и конец моего "кубика"
# Теперь my_coords - это массив, умножение пройдет поэлементно
my_start = my_coords * domain_len
my_end = (my_coords + 1) * domain_len

pos = full_system.get_positions()

# Создаем маску для 3D объема
mask = (
    (pos[:, 0] >= my_start[0]) & (pos[:, 0] < my_end[0]) &
    (pos[:, 1] >= my_start[1]) & (pos[:, 1] < my_end[1]) &
    (pos[:, 2] >= my_start[2]) & (pos[:, 2] < my_end[2])
)

# Коррекция правой границы для крайних процессов
for i in range(3):
    if my_coords[i] == dims[i] - 1:
        mask &= (pos[:, i] <= L[i] + 0.0001)

local_atoms = full_system[mask]
local_atoms.set_cell(cell)
local_atoms.set_pbc(pbc)

# print(f"[{rank}] Coords {my_coords}: Atoms {len(local_atoms)} | Bounds X[{my_start[0]:.1f}:{my_end[0]:.1f}]", flush=True)
comm.Barrier()

# ==========================================
# 3. ОБМЕН ПРИЗРАКАМИ (GHOST EXCHANGE) В 3D
# ==========================================
current_atoms = local_atoms.copy()
all_ghosts = [] 

for axis in range(3): # 0=X, 1=Y, 2=Z
    source_rank, dest_rank = cart_comm.Shift(direction=axis, disp=1)
    
    pos = current_atoms.get_positions()
    
    # Подготовка данных для отправки направо
    right_boundary = my_end[axis]
    mask_to_right = pos[:, axis] > (right_boundary - CUTOFF)
    atoms_to_right = current_atoms[mask_to_right]

    # Подготовка данных для отправки налево
    left_boundary = my_start[axis]
    mask_to_left = pos[:, axis] < (left_boundary + CUTOFF)
    atoms_to_left = current_atoms[mask_to_left]
    
    # ОБМЕН
    recv_from_left = cart_comm.sendrecv(sendobj=atoms_to_right, dest=dest_rank, source=source_rank)
    recv_from_right = cart_comm.sendrecv(sendobj=atoms_to_left, dest=source_rank, source=dest_rank)
    
    # КОРРЕКЦИЯ PBC
    if my_coords[axis] == 0 and recv_from_left is not None and len(recv_from_left) > 0:
        p = recv_from_left.get_positions()
        p[:, axis] -= L[axis]
        recv_from_left.set_positions(p)
        
    if my_coords[axis] == dims[axis] - 1 and recv_from_right is not None and len(recv_from_right) > 0:
        p = recv_from_right.get_positions()
        p[:, axis] += L[axis]
        recv_from_right.set_positions(p)

    # Добавляем полученных
    ghosts_step = []
    if recv_from_left and len(recv_from_left) > 0:
        ghosts_step.append(recv_from_left)
    if recv_from_right and len(recv_from_right) > 0:
        ghosts_step.append(recv_from_right)
    
    for g in ghosts_step:
        current_atoms += g
        all_ghosts.append(g)

final_atoms = current_atoms

# ==========================================
# 4. РАСЧЕТ
# ==========================================
tree = cKDTree(final_atoms.get_positions())
pairs = tree.query_pairs(r=CUTOFF)

# Фильтрация пар: считаем пару, только если хотя бы один атом СВОЙ
count_local_pairs = 0
local_indices_count = len(local_atoms)

valid_pairs = 0
for i, j in pairs:
    if i < local_indices_count or j < local_indices_count:
        valid_pairs += 1

# print(f"[{rank}] DONE. Valid pairs: {valid_pairs}", flush=True)
from dscribe.descriptors import SOAP, ACSF
unique_symbols = set(current_atoms.symbols)
# print(unique_symbols)
len(current_atoms)
species_str = np.array(list(unique_symbols))
R_cut = 6
n_max = 8
l_max = 4
soap_test = SOAP(
    species=species_str,
    periodic=False,
    r_cut=5,
    n_max=n_max,
    l_max=l_max,
    compression = {'mode':'mu2'})
local_enviroments = soap_test.create(current_atoms, n_jobs = 1)
local_enviroments.shape

comm.Barrier()

usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
usage_mb = usage_kb / 1024.0
all_usages = comm.gather(usage_mb, root=0)
if rank == 0:
    result_data = {
        "cores": size,
        "time_sec" :MPI.Wtime() - t_start,
        "memory_peak_rank0_mb": all_usages[0],           # Сколько съел Мастер
        "memory_peak_worker_avg_mb": np.mean(all_usages[1:]) if size > 1 else 0, # Среднее по рабочим
        "memory_total_cluster_mb": sum(all_usages) 
    }
    print(json.dumps(result_data))
    # print(f">>> SUCCESS. Total time: {MPI.Wtime() - t_start:.4f} sec <<<", flush=True)