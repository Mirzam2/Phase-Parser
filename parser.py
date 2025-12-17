from mpi4py import MPI
import numpy as np
import sys
import os
import json
#import resource
from dscribe.descriptors import SOAP
from sklearn.preprocessing import StandardScaler

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
FILENAME = "data/multiphase.dat" 
CUTOFF = 5

REFERENCE_CONFIGURATIONS = ['reference_data/BN.cif', 'reference_data/HfB2.cif', 'reference_data/TiB2.cif']

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

full_system = io.read(FILENAME, format='lammps-data')
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
N = len(current_atoms)
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
unique_symbols = set(full_system.symbols)
# print(unique_symbols)
len(current_atoms)
species_str = np.array(list(unique_symbols))
R_cut = 4
n_max = 8
l_max = 4
soap_test = SOAP(
    species=species_str,
    periodic=False,
    r_cut=R_cut,
    n_max=n_max,
    l_max=l_max,
    compression = {'mode':'mu2'})
local_enviroments = soap_test.create(current_atoms, n_jobs = 1)
local_enviroments.shape

comm.Barrier()

# usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# usage_mb = usage_kb / 1024.0
# all_usages = comm.gather(usage_mb, root=0)
# if rank == 0:
#     result_data = {
#         "cores": size,
#         "time_sec" :MPI.Wtime() - t_start,
#         "memory_peak_rank0_mb": all_usages[0],           # Сколько съел Мастер
#         "memory_peak_worker_avg_mb": np.mean(all_usages[1:]) if size > 1 else 0, # Среднее по рабочим
#         "memory_total_cluster_mb": sum(all_usages) 
#     }
#     print(json.dumps(result_data))        
                                            
# ... (Previous code ends at json output) ...
                                            
# ==========================================
# 5. COMPOSITE CLUSTERING & REFINEMENT
# ==========================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import hdbscan

# ---------------------------------------------------------
# A. PREPARE DATA & REFERENCES
# ---------------------------------------------------------

# 1. Process Reference Data
ref_descriptors = []
ref_labels = []

for idx, ref_file in enumerate(REFERENCE_CONFIGURATIONS):
    try:
        ref_atoms_struct = io.read(ref_file)
        # Calculate SOAP for reference
        d = soap_test.create(ref_atoms_struct, n_jobs=1)
        ref_descriptors.append(d)
        ref_labels.extend([idx] * len(d))
    except Exception as e:
        if rank == 0: print(f"Warning: Failed to process reference {ref_file}: {e}", flush=True)

if not ref_descriptors:
    if rank == 0: print("ERROR: No reference descriptors calculated.")
    sys.exit(1)

X_ref = np.vstack(ref_descriptors)
y_ref = np.array(ref_labels)

# 2. Global Scaling & PCA (Local + Ghosts + References)
# We scale everything based on the LOCAL distribution to preserve local variance features
scaler = StandardScaler()

# local_enviroments contains [Local Atoms (0..N) | Ghost Atoms (N..end)]
# Fit scaler on Local atoms only (indices < N)
X_local_raw = local_enviroments[:N]
scaler.fit(X_local_raw)

# Transform everything
X_full_scaled = scaler.transform(local_enviroments) # Size: N + Ghosts
X_ref_scaled = scaler.transform(X_ref)

# PCA - Reduce dimensionality
pca = PCA(n_components=10)
# Fit on local data
pca.fit(X_full_scaled[:N]) 

X_full_pca = pca.transform(X_full_scaled) # Local + Ghosts projected
X_ref_pca = pca.transform(X_ref_scaled)   # References projected

# ---------------------------------------------------------
# B. STAGE 1: UNSUPERVISED CLUSTERING + CENTROID LABELING
# ---------------------------------------------------------

# 1. HDBSCAN on Local Atoms (indices < N)
X_local_pca = X_full_pca[:N]
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, prediction_data=True)
hdbscan_labels = clusterer.fit_predict(X_local_pca)

# 2. Train KNN for Centroids (Reference Matching)
knn_ref = KNeighborsClassifier(n_neighbors=1)
knn_ref.fit(X_ref_pca, y_ref)

# 3. Assign Labels to Clusters
local_labels = np.full(N, -1, dtype=np.int32) # Default -1 (Noise)
unique_clusters = np.unique(hdbscan_labels)

if rank == 0:
    print(f"[{rank}] HDBSCAN found {len(unique_clusters)} clusters.", flush=True)

for cluster_id in unique_clusters:
    if cluster_id == -1: continue # Skip noise for now
    
    # Extract points and calculate centroid
    mask_c = (hdbscan_labels == cluster_id)
    centroid = np.mean(X_local_pca[mask_c], axis=0).reshape(1, -1)
    
    # Predict label for the centroid
    ref_idx = knn_ref.predict(centroid)[0]
    
    # Assign to all atoms in this cluster
    local_labels[mask_c] = ref_idx

# ---------------------------------------------------------
# C. STAGE 2: GHOST LABEL EXCHANGE
# ---------------------------------------------------------
# We need to populate labels for the ghost atoms (indices >= N)
# so we can use them as neighbors for the noise refinement.
# We must replicate the exact geometric logic used in Section 3.

# Re-initialize temp arrays to simulate the ghost buildup process
temp_atoms = local_atoms.copy()
current_labels_list = local_labels.copy().tolist() # Easier to append to list

# We reuse variables defined in Section 1 (dims, my_coords, L, CUTOFF)
for axis in range(3):
    source_rank, dest_rank = cart_comm.Shift(direction=axis, disp=1)
    
    pos = temp_atoms.get_positions()
    labels_arr = np.array(current_labels_list, dtype=np.int32)
    
    # 1. Prepare Data to Send (Same logic as atoms)
    # Right
    right_boundary = my_end[axis]
    mask_to_right = pos[:, axis] > (right_boundary - CUTOFF)
    labels_to_right = labels_arr[mask_to_right]

    # Left
    left_boundary = my_start[axis]
    mask_to_left = pos[:, axis] < (left_boundary + CUTOFF)
    labels_to_left = labels_arr[mask_to_left]
    
    # 2. Exchange Labels
    # Note: We send labels as integers.
    recv_labels_left = cart_comm.sendrecv(sendobj=labels_to_right, dest=dest_rank, source=source_rank)
    recv_labels_right = cart_comm.sendrecv(sendobj=labels_to_left, dest=source_rank, source=dest_rank)
    
    # 3. Append received labels to our list
    # Important: The order of appending MUST match the order atoms were added in Section 3
    if recv_labels_left is not None and len(recv_labels_left) > 0:
        current_labels_list.extend(recv_labels_left)
    
    if recv_labels_right is not None and len(recv_labels_right) > 0:
        current_labels_list.extend(recv_labels_right)
        
    # 4. Update temp_atoms geometry (Required so next axis masks are correct!)
    # We must append the ghost ATOMS to temp_atoms exactly as done before
    # We can retrieve them from 'all_ghosts' or re-exchange. 
    # Since 'all_ghosts' is a flat list of atoms objects added sequentially, we can reconstruct.
    # However, it's safer to just re-use the 'all_ghosts' list we built in Section 3.
    # To do this correctly without re-communicating atoms, we need to know how many ghosts were added per step.
    # SIMPLIFICATION: Since we ran the EXACT same mask logic, the counts match.
    # We just need to know how many ghosts were added in this specific axis iteration to skip ahead in 'all_ghosts'?
    # Actually, simpler: just run the atom exchange again. It's fast compared to SOAP.
    
    # ... Re-running atom exchange to keep 'pos' array consistent ...
    # (Copy-paste of minimal atom exchange logic from Sec 3 for geometry update)
    recvd_atoms_left = cart_comm.sendrecv(sendobj=temp_atoms[mask_to_right], dest=dest_rank, source=source_rank)
    recvd_atoms_right = cart_comm.sendrecv(sendobj=temp_atoms[mask_to_left], dest=source_rank, source=dest_rank)
    
    # Correct PBC (geometry only)
    if my_coords[axis] == 0 and recvd_atoms_left:
        p = recvd_atoms_left.get_positions(); p[:, axis] -= L[axis]; recvd_atoms_left.set_positions(p)
    if my_coords[axis] == dims[axis]-1 and recvd_atoms_right:
        p = recvd_atoms_right.get_positions(); p[:, axis] += L[axis]; recvd_atoms_right.set_positions(p)

    if recvd_atoms_left: temp_atoms += recvd_atoms_left
    if recvd_atoms_right: temp_atoms += recvd_atoms_right

# Now 'current_labels_list' has size matching 'final_atoms' / 'X_full_pca'
full_labels = np.array(current_labels_list, dtype=np.int32)

# Sanity check
if len(full_labels) != len(X_full_pca):
    if rank == 0: print(f"ERROR: Label sync mismatch. Labels: {len(full_labels)}, Descriptors: {len(X_full_pca)}")
    sys.exit(1)

# ---------------------------------------------------------
# D. STAGE 3: REFINEMENT (KNN ON NOISE)
# ---------------------------------------------------------

# Identify Noise atoms (Label -1) among LOCAL atoms
mask_local_noise = (full_labels[:N] == -1)
count_noise = np.sum(mask_local_noise)

if count_noise > 0:
    # 1. Training Set: All atoms (Local + Ghosts) that have a valid label (>= 0)
    mask_valid = (full_labels != -1)
    
    if np.sum(mask_valid) > 5: # Need enough points to train
        X_train_refine = X_full_pca[mask_valid]
        y_train_refine = full_labels[mask_valid]
        
        # 2. Target Set: Local atoms that are noise
        X_target_refine = X_full_pca[:N][mask_local_noise]
        
        # 3. Train KNN (using immediate neighbors to smooth)
        # k=5 is robust enough to look at surrounding shell
        knn_refine = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn_refine.fit(X_train_refine, y_train_refine)
        
        # 4. Predict
        refined_labels = knn_refine.predict(X_target_refine)
        
        # 5. Update local labels
        local_labels[mask_local_noise] = refined_labels
        
        # print(f"[{rank}] Refined {count_noise} noise atoms.", flush=True)
    else:
        if rank == 0: print(f"[{rank}] Warning: Not enough valid labels to refine noise.", flush=True)

# Final Output for gathering
# print(f"[{rank}] Final Labels Dist: {np.unique(local_labels, return_counts=True)}", flush=True)

# ==========================================
# 6. GATHERING RESULTS TO ROOT (CORRECT ORDER)
# ==========================================

# 1. Recover global IDs for local atoms
# We use the 'mask' variable defined in "Section 2" of your script.
# Since full_system is loaded on every rank, we can determine which global IDs we own.
all_indices = np.arange(len(full_system), dtype=np.int32)
local_ids = all_indices[mask] # These are the original indices (0 to Total_Atoms-1)

# Ensure data types are correct for MPI
local_labels = local_labels.astype(np.int32)
local_ids = local_ids.astype(np.int32)

# Verify alignment (sanity check)
# N was defined before adding ghosts, so it represents the count of owned atoms
if len(local_ids) != len(local_labels):
    if rank == 0: print("ERROR: Mismatch between local IDs and calculated labels.")
    sys.exit(1)

# 2. Gather the counts of atoms per core to Root
# This tells Root how many items to expect from each rank
local_count = len(local_labels)
counts = comm.gather(local_count, root=0)

# 3. Prepare Receive Buffers on Root
recvbuf_labels = None
recvbuf_ids = None

if rank == 0:
    # Calculate displacements (offset where each rank's data starts in the buffer)
    # logic: [0, count0, count0+count1, ...]
    displacements = np.insert(np.cumsum(counts), 0, 0)[:-1]
    
    total_received_atoms = sum(counts)
    
    # Check if we lost any atoms (optional safety)
    if total_received_atoms != len(full_system):
         print(f"WARNING: Total atoms gathered ({total_received_atoms}) != System size ({len(full_system)})")

    # Allocate arrays to receive raw data
    recvbuf_labels = np.empty(total_received_atoms, dtype=np.int32)
    recvbuf_ids = np.empty(total_received_atoms, dtype=np.int32)
else:
    counts = None
    displacements = None

# 4. Perform Vector Gathers (Gatherv)
# Gather the predicted labels
comm.Gatherv(sendbuf=local_labels, 
             recvbuf=[recvbuf_labels, counts, displacements, MPI.INT], 
             root=0)

# Gather the original IDs associated with those labels
comm.Gatherv(sendbuf=local_ids, 
             recvbuf=[recvbuf_ids, counts, displacements, MPI.INT], 
             root=0)

# 5. Reconstruct the sorted array on Root
if rank == 0:
    # Create an array to hold the final sorted result
    # We initialize with -1 to spot any missing indices later if needed
    final_sorted_labels = np.full(len(full_system), -1, dtype=np.int32)
    
    # MAGIC STEP: Use the gathered IDs as indices
    # This places the label from recvbuf_labels[i] into final_sorted_labels[recvbuf_ids[i]]
    final_sorted_labels[recvbuf_ids] = recvbuf_labels
    
    print("Gathering complete.", flush=True)
    
    # --- Example: Save to file ---
    # np.savetxt("cluster_labels.txt", final_sorted_labels, fmt='%d')
    
    # --- Example: Print stats ---
    # unique, u_counts = np.unique(final_sorted_labels, return_counts=True)
    # print("Cluster distribution:", dict(zip(unique, u_counts)))

    # ==========================================
    # 7. SAVE TO .XYZ WITH LABEL COLUMN
    # ==========================================
    
    OUTPUT_FILE = "clustered_output.xyz"
    
    # 1. Attach the sorted labels to the ASE Atoms object
    # This creates a new column in the internal data structure
    full_system.set_array('label', final_sorted_labels)
    
    # 2. Write to file using 'extxyz' format
    # This format preserves the extra 'label' column in the output file header
    # Format: Species X Y Z label
    try:
        io.write(OUTPUT_FILE, full_system, format='extxyz')
        print(f"Successfully saved clustered structure to {OUTPUT_FILE}", flush=True)
    except Exception as e:
        print(f"Error saving .xyz file: {e}", flush=True)