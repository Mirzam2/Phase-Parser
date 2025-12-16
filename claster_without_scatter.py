from mpi4py import MPI
import numpy as np
import sys
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

try:
    from ase import io, Atoms
    from scipy.spatial import cKDTree
except ImportError as e:
    print(f"[{rank}] ERROR: Missing library! {e}", flush=True)
    sys.exit(1)

comm.Barrier()
t_start = MPI.Wtime()

# ПАРАМЕТРЫ
FILENAME = "HEB2_5gr_p1.data" 
CUTOFF = 3.5 

# ==========================================
# ИЗМЕНЕНИЕ: ВМЕСТО SCATTER КАЖДЫЙ ЧИТАЕТ САМ
# Это надежнее на Windows, чтобы избежать зависаний сети
# ==========================================

print(f"[{rank}] Reading file independently...", flush=True)

if not os.path.exists(FILENAME):
    if rank == 0: print("ERROR: File not found")
    sys.exit(1)


full_system = io.read(FILENAME, format='lammps-data', atom_style='atomic')
cell = full_system.get_cell()
total_cell_x = cell[0][0]

# SLICING
slice_width = total_cell_x / size
x_start = rank * slice_width
x_end = (rank + 1) * slice_width

pos = full_system.get_positions()

# Маска для выбора своих атомов
if rank == size - 1:
    mask = (pos[:, 0] >= x_start) & (pos[:, 0] <= total_cell_x + 10.0)
else:
    mask = (pos[:, 0] >= x_start) & (pos[:, 0] < x_end)

# Создаем свои локальные атомы
local_atoms = full_system[mask]
local_atoms.set_cell(cell)
local_atoms.set_pbc(full_system.get_pbc())
my_x_range = (x_start, x_end)

print(f"[{rank}] I selected my slice. Atoms count: {len(local_atoms)}", flush=True)
comm.Barrier()


#ОБЫЧНЫЙ ОБМЕН (GHOST EXCHANGE)
print(f"[{rank}] Starting Ghost Exchange...", flush=True)
left_neigh = (rank - 1) % size
right_neigh = (rank + 1) % size
pos = local_atoms.get_positions()

# Подготовка данных
left_mask = pos[:, 0] < (my_x_range[0] + CUTOFF)
data_to_left = local_atoms[left_mask]
right_mask = pos[:, 0] > (my_x_range[1] - CUTOFF)
data_to_right = local_atoms[right_mask]

# Обмен
try:
    ghosts_from_right = comm.sendrecv(sendobj=data_to_left, dest=left_neigh, source=right_neigh)
    ghosts_from_left = comm.sendrecv(sendobj=data_to_right, dest=right_neigh, source=left_neigh)
except Exception as e:
    print(f"[{rank}] GHOST EXCHANGE FAILED (Firewall block?): {e}", flush=True)
    sys.exit(1)

# Коррекция PBC
if rank == 0:
    p = ghosts_from_left.get_positions()
    p[:, 0] -= total_cell_x
    ghosts_from_left.set_positions(p)

if rank == size - 1:
    p = ghosts_from_right.get_positions()
    p[:, 0] += total_cell_x
    ghosts_from_right.set_positions(p)

# Объединение
final_atoms = local_atoms.copy()
final_atoms += ghosts_from_left
final_atoms += ghosts_from_right



# Расчет
tree = cKDTree(final_atoms.get_positions())
pairs = tree.query_pairs(r=CUTOFF)

print(f"[{rank}] DONE. Pairs found: {len(pairs)}", flush=True)

comm.Barrier()
if rank == 0:
    print(f">>> SUCCESS. Total time: {MPI.Wtime() - t_start:.4f} sec <<<", flush=True)