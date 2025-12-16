from mpi4py import MPI
import numpy as np
import sys
import os

# Инициализация MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    from ase import io, Atoms
    from scipy.spatial import cKDTree
except ImportError as e:
    print(f"[{rank}] ERROR: Missing library! {e}", flush=True)
    sys.exit(1)

# Синхронизация перед стартом
comm.Barrier()
t_start_global = MPI.Wtime()

# --- ПАРАМЕТРЫ ---
FILENAME = "HfB2_X.dat" 
CUTOFF = 3.5 

# --- ЭТАП 1: ЧТЕНИЕ И НАРЕЗКА (Только Rank 0) ---
local_atoms = None
my_x_range = None
total_cell_x = 0.0

t_io_start = MPI.Wtime()

if rank == 0:
    print(f"[{rank}] Reading file via ASE...", flush=True)
    if not os.path.exists(FILENAME):
        print(f"[{rank}] ERROR: File {FILENAME} not found!", flush=True)
        comm.Abort(1)

    # Используем atom_style='atomic' для правильного чтения LAMMPS
    full_system = io.read(FILENAME, format='lammps-data', atom_style='atomic')
    
    cell = full_system.get_cell()
    pbc = full_system.get_pbc()
    pos = full_system.get_positions()
    total_cell_x = cell[0][0]
    
    print(f"[{rank}] File read. System size: {len(full_system)} atoms. Cell X: {total_cell_x:.2f}", flush=True)

    # Готовим массив для Scatter
    # Используем numpy array object вместо списка для эффективности
    scatter_data = np.empty(size, dtype=object)

    print(f"[{rank}] Slicing data for {size} processors...", flush=True)
    
    slice_width = total_cell_x / size
    
    for r in range(size):
        x_start = r * slice_width
        x_end = (r + 1) * slice_width
        
        # Векторизованная маска NumPy (быстро)
        if r == size - 1:
            # Последний кусок берет всё до конца + запас
            mask = (pos[:, 0] >= x_start) & (pos[:, 0] <= total_cell_x + 10.0)
        else:
            mask = (pos[:, 0] >= x_start) & (pos[:, 0] < x_end)
            
        # ASE при slicing сохраняет типы, массы и заряды атомов
        sub_atoms = full_system[mask]
        sub_atoms.set_cell(cell)
        sub_atoms.set_pbc(pbc)
        
        # Упаковываем в массив
        scatter_data[r] = (sub_atoms, (x_start, x_end), total_cell_x)
    
    print(f"[{rank}] Scatter data prepared. Starting distribution...", flush=True)
else:
    scatter_data = None

if rank == 0:
    print(f"[{rank}] TIME: IO & Prep took {MPI.Wtime() - t_io_start:.4f} sec", flush=True)


# --- ЭТАП 2: SCATTER (Рассылка данных) ---
t_scatter_start = MPI.Wtime()

try:
    # Самая опасная операция на Windows. 
    # Если висит - значит проблема с pickle/сетью.
    data_packet = comm.scatter(scatter_data, root=0)
    
    local_atoms = data_packet[0]
    my_x_range = data_packet[1]
    total_cell_x = data_packet[2]
    # print(f"[{rank}] Received {len(local_atoms)} atoms.", flush=True)
except Exception as e:
    print(f"[{rank}] SCATTER FAILED: {e}", flush=True)
    sys.exit(1)

comm.Barrier()
t_scatter_end = MPI.Wtime()
print(f"[{rank}] TIME: Scatter took {t_scatter_end - t_scatter_start:.4f} sec", flush=True)


# --- ЭТАП 3: GHOST EXCHANGE (Обмен границами) ---
t_ghost_start = MPI.Wtime()

left_neigh = (rank - 1) % size
right_neigh = (rank + 1) % size
pos = local_atoms.get_positions()

# Выбираем атомы для соседей
left_mask = pos[:, 0] < (my_x_range[0] + CUTOFF)
data_to_left = local_atoms[left_mask]

right_mask = pos[:, 0] > (my_x_range[1] - CUTOFF)
data_to_right = local_atoms[right_mask]

# Безопасный обмен (SendRecv)
ghosts_from_right = comm.sendrecv(sendobj=data_to_left, dest=left_neigh, source=right_neigh)
ghosts_from_left = comm.sendrecv(sendobj=data_to_right, dest=right_neigh, source=left_neigh)

# Коррекция PBC (Периодические граничные условия)
if rank == 0:
    p = ghosts_from_left.get_positions()
    p[:, 0] -= total_cell_x
    ghosts_from_left.set_positions(p)

if rank == size - 1:
    p = ghosts_from_right.get_positions()
    p[:, 0] += total_cell_x
    ghosts_from_right.set_positions(p)

# Объединение (Свои + Призраки)
final_atoms = local_atoms.copy()
final_atoms += ghosts_from_left
final_atoms += ghosts_from_right

t_ghost_end = MPI.Wtime()
print(f"[{rank}] TIME: Ghost Exchange took {t_ghost_end - t_ghost_start:.4f} sec. Total atoms: {len(final_atoms)}", flush=True)


# --- ЭТАП 4: РАСЧЕТ (cKDTree) ---
t_calc_start = MPI.Wtime()

# Строим дерево и ищем пары
tree = cKDTree(final_atoms.get_positions())
pairs = tree.query_pairs(r=CUTOFF)

t_calc_end = MPI.Wtime()
print(f"[{rank}] TIME: Calc took {t_calc_end - t_calc_start:.4f} sec. Pairs found: {len(pairs)}", flush=True)

# --- ФИНАЛ ---
comm.Barrier()
t_end_global = MPI.Wtime()

if rank == 0:
    print("-" * 40, flush=True)
    print(f">>> ALL DONE SUCCESSFULLY <<<")
    print(f">>> TOTAL EXECUTION TIME: {t_end_global - t_start_global:.4f} sec <<<")
    print("-" * 40, flush=True)