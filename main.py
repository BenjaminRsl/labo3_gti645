import mpi4py.MPI
from mpi4py import MPI
import numpy as np
from scipy.linalg import lu
import argparse
from dataclasses import dataclass
import time
import math

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--mat_size", default=256, type=int,
                    help="Size of the matrix to compute LU on")
args = vars(parser.parse_args())

CHECK_EPSILON = 1e-4


def init_random_mat(rows, cols, max_value=100) -> np.ndarray:
    return np.random.rand(rows, cols) * max_value


def check_is_upper(mat: np.ndarray) -> bool:
    rows = mat.shape[0]
    for i in range(rows):
        for j in range(i):
            if abs(mat[i][j]) > CHECK_EPSILON:
                return False
    return True


def check_is_lower(mat: np.ndarray) -> bool:
    rows = mat.shape[0]
    cols = mat.shape[1]
    for i in range(rows):
        for j in range(i+1, cols):
            if abs(mat[i][j]) > CHECK_EPSILON:
                return False
    return True


def check_equal(mat_a: np.ndarray, mat_b: np.ndarray) -> bool:
    rows = mat_a.shape[0]
    cols = mat_a.shape[1]
    for i in range(rows):
        for j in range(cols):
            if (abs(mat_a[i][j] - mat_b[i][j])) > CHECK_EPSILON:
                return False
    return True


def check_correct(mat_a: np.ndarray, mat_l: np.ndarray, mat_u: np.ndarray) -> bool:
    if not check_is_upper(mat_u):
        print("Incorrect: U is not upper")
        return False

    mat_result = np.matmul(mat_l, mat_u)
    if not check_equal(mat_a, mat_result):
        print("Incorrect: A != LU")
        return False

    print("Correct")
    return True


def exemple():
    # Initialisation de la matrice
    A_data = [[0.7063956, 0.0958044, 0.4083480, 0.0296047, 0.3537505, 0.3891269, 0.3340674, 0.7760742],
    [0.2530419, 0.3183366, 0.6654437, 0.2700742, 0.3496487, 0.1803152,   0.8940230, 0.9117640],
    [0.4949135, 0.0521427, 0.3901374, 0.6078649, 0.5272401, 0.8876878, 0.0169656, 0.5881695],
    [0.2591390, 0.7560347, 0.7787556, 0.6083338, 0.2959305, 0.1282858, 0.4537364, 0.9144353],
    [0.1366075, 0.0067270, 0.5809726, 0.3003515, 0.6920679, 0.7308718, 0.3723367, 0.0115566],
    [0.7243697, 0.9559875, 0.4004689, 0.5467245, 0.9444092, 0.1769764,  0.1137369, 0.7531579],
    [0.3082837, 0.6498422, 0.9510023, 0.6785915, 0.6560145, 0.1313168,   0.2105410, 0.9547708],
    [0.7357185, 0.1029297, 0.6172203, 0.2037524, 0.0872543, 0.3861805,   0.9129948, 0.1102923]]

    A = np.array(A_data, dtype=np.double)

    #######################
    #        Panel1       #
    #######################

    # On prend le premier panel
    P = A[:,0:2]

    # Découpage de ce panel en sous-matrices
    A11 = P[0:2,:]
    A21 = P[2:4,:]
    A31 = P[4:6,:]
    A41 = P[6:8,:]

    # Calcul des LU locaux
    L1, U11 = lu(A11, permute_l=True)
    L2, U21 = lu(A21, permute_l=True)
    L3, U31 = lu(A31, permute_l=True)
    L4, U41 = lu(A41, permute_l=True)

    # Concaténation 2 à 2 des U obtenus
    # en parallèle, les processus imparis envoient leur U à leur copain pair
    # par exemple, 1 envoie à 0, 3 envoie à 2...
    V1 = np.concatenate((U11, U21), axis=0)
    V3 = np.concatenate((U31, U41), axis=0)

    # LU des matrices obtenus
    # En parallèle, seuls les processus pairs le font
    L1_, U1_ = lu(V1, permute_l=True)
    L3_, U3_ = lu(V3, permute_l=True)

    L1__ = np.dot(L1_, L1)
    L3__ = np.dot(L3_, L3)

    # Concaténation 2 à 2 des U obtenus
    # En parallèle, ça se fait ent re processus de rangs r+-2**s
    # si s est le numéro de l'étape et r le rand du processus
    Vend = np.concatenate((U1_[0:2, :], U3_[0:2, :]), axis=0)

    # Ici on est à la fin donc on obtient le U final
    L_, U_11 = lu(Vend, permute_l=True)

    # Calcul du L
    L11 = np.dot(A11, np.linalg.inv(U_11))
    L21 = np.dot(A21, np.linalg.inv(U_11))
    L31 = np.dot(A31, np.linalg.inv(U_11))
    L41 = np.dot(A41, np.linalg.inv(U_11))

    L_1 = np.concatenate((L11, L21, L31, L41), axis=0)

    #######################
    #        Update1      #
    #######################

    # Partie de droite
    # Les processus qui ont fait la panel facto envoient le inv(L1)
    T1 = np.linalg.inv(L11)

    A12 = A[0:2, 2:4]
    A13 = A[0:2, 4:6]
    A14 = A[0:2, 6:8]

    U_12 = np.dot(T1, A12)
    U_13 = np.dot(T1, A13)
    U_14 = np.dot(T1, A14)

    # On update ce qu'il y a en-dessous, en colonnes
    # On utilise le U de la colonne et le L de la ligne
    A22 = A[2:4, 2:4]
    A32 = A[4:6, 2:4]
    A42 = A[6:8, 2:4]
    A22_ = A22 - np.dot(L21, U_12)
    A32_ = A32 - np.dot(L31, U_12)
    A42_ = A42 - np.dot(L41, U_12)

    A23 = A[2:4, 4:6]
    A33 = A[4:6, 4:6]
    A43 = A[6:8, 4:6]
    A23_ = A23 - np.dot(L21, U_13)
    A33_ = A33 - np.dot(L31, U_13)
    A43_ = A43 - np.dot(L41, U_13)

    A24 = A[2:4, 6:8]
    A34 = A[4:6, 6:8]
    A44 = A[6:8, 6:8]
    A24_ = A24 - np.dot(L21, U_14)
    A34_ = A34 - np.dot(L31, U_14)
    A44_ = A44 - np.dot(L41, U_14)

    #######################
    #        Panel2       #
    #######################

    # On recommence sur la trailing matrix (ici 3x3)
    # En prenant comme panel [A22_; A32_; A42_]
    L2, U2 = lu(A22_, permute_l=True)
    L3, U3 = lu(A32_, permute_l=True)
    L4, U4 = lu(A42_, permute_l=True)

    V1 = np.concatenate((U2, U3), axis=0)
    L2_, U2_ = lu(V1, permute_l=True)
    V1 = np.concatenate((U2_, U4), axis=0)

    L_, U_22 = lu(V1, permute_l=True)

    # Calcul du L
    L22 = np.dot(A22_, np.linalg.inv(U_22))
    L32 = np.dot(A32_, np.linalg.inv(U_22))
    L42 = np.dot(A42_, np.linalg.inv(U_22))

    L_2 = np.concatenate((L22, L32, L42), axis=0)

    #######################
    #        Update2      #
    #######################

    # Partie de droite
    # On diffuse le inv(L) dans la ligne
    T2 = np.linalg.inv(L22)

    U_23 = np.dot(T2, A23_)
    U_24 = np.dot(T2, A24_)

    # On update ce qu'il y a en-dessous, en colonnes
    A33__ = A33_ - np.dot(L32, U_23)
    A43__ = A43_ - np.dot(L42, U_23)

    A34__ = A34_ - np.dot(L32, U_24)
    A44__ = A44_ - np.dot(L42, U_24)

    #######################
    #        Panel3       #
    #######################

    # On recommence sur la trailing matrix (ici 2x2)
    # En prenant comme panel [A33__; A43__]
    L3, U3 = lu(A33__, permute_l=True)
    L4, U4 = lu(A43__, permute_l=True)
    V1 = np.concatenate((U3, U4), axis=0)
    L_, U_33 = lu(V1, permute_l=True)

    # Calcul du L
    L33 = np.dot(A33__, np.linalg.inv(U_33))
    L43 = np.dot(A43__, np.linalg.inv(U_33))

    L_3 = np.concatenate((L33, L43), axis=0)

    #######################
    #        Update3      #
    #######################
    # Partie de droite
    # On divise le inv(L) dans la ligne (il n'en reste qu'un)
    T3 = np.linalg.inv(L33)
    U_34 = np.dot(T3, A34__)

    # On update ce qu'il y a en-dessous, un seul bloc
    A44___ = A44__ - np.dot(L43, U_34)

    # Il ne reste qu'un bloc à faire, on facto directement
    L_4, U_44 = lu(A44___, permute_l=True)

    #######################
    #    Dernier Bloc     #
    #######################
    # Formation du résultat final
    Z = np.zeros((2, 2))

    #######################
    #      Résultat       #
    #######################
    L = np.concatenate((L_1, np.concatenate((Z, L_2), axis=0), np.concatenate((Z, Z, L_3), axis=0), np.concatenate((Z, Z, Z, L_4), axis=0)), axis=1)
    U = np.concatenate((np.concatenate((U_11, Z, Z, Z), axis=0), np.concatenate((U_12, U_22, Z, Z), axis=0), np.concatenate((U_13, U_23, U_33, Z), axis=0), np.concatenate((U_14, U_24, U_34, U_44), axis=0)), axis=1)

    #######################
    #    Verification     #
    #######################

    check_correct(A, L, U)

@dataclass
class Info:
    comm: mpi4py.MPI.Intracomm
    rank: int
    merge_rank: int
    num_proc: int
    mat_size: int
    submat_size: int
    num_iterations: int
    size_log2: int
    A: np.ndarray
    Z: np.ndarray


def inv_rank(info: Info):
    return info.num_proc - info.rank - 1


def find_U_panel_final(info: Info, U_panel: np.ndarray, current_iteration: int) -> (int, np.ndarray):
    """Will only give right result on rank marked by the return value"""

    U_panel_final = np.zeros((info.submat_size, info.submat_size))

    rank_start = current_iteration
    merge_mat = U_panel
    for i in range(info.size_log2):
        jump = 2 ** i
        is_receiving_cell = info.merge_rank % (2 ** (i + 1)) == 0
        is_merging_cell = info.merge_rank % jump == 0

        if is_receiving_cell:
            pair_rank = info.rank - jump

            if pair_rank >= rank_start:
                pair_mat = np.zeros((info.submat_size, info.submat_size))
                info.comm.Recv(pair_mat, source=pair_rank, tag=i)

                concatenated_mat = np.concatenate((pair_mat, merge_mat), axis=0)
                _, merge_mat = lu(concatenated_mat, permute_l=True)

        elif is_merging_cell:
            if info.rank >= rank_start:
                pair_rank = info.rank + jump
                info.comm.Send(merge_mat.ravel(), dest=pair_rank, tag=i)

        is_last_merge = i + 1 == info.size_log2
        if is_last_merge and info.rank == info.num_proc - 1:
            _, U_panel_final = lu(merge_mat, permute_l=True)

    return info.num_proc - 1, U_panel_final


def col_to_row(info: Info, col_mat: np.ndarray) -> np.ndarray:
    row_mat = np.zeros((col_mat.shape[1], col_mat.shape[0]))
    num_submatrixes = col_mat.shape[0] // info.submat_size
    for i in range(num_submatrixes):
        row_mat[0: info.submat_size, info.submat_size * i: info.submat_size * i + info.submat_size] = col_mat[info.submat_size * i: info.submat_size * i + info.submat_size, 0: info.submat_size]
    return row_mat

def resolve_lu(info: Info) -> (np.ndarray, np.ndarray):
    """Will only give right result on rank 0"""
    A = np.copy(info.A)
    L = np.zeros((info.mat_size, info.mat_size))
    U = np.zeros((info.mat_size, info.mat_size))

    for current_iteration in range(info.num_iterations - 1):
        panel = A[:, info.submat_size * current_iteration:info.submat_size * current_iteration + info.submat_size]

        A_panel = np.zeros((info.submat_size, info.submat_size))
        info.comm.Scatter(panel.ravel(), A_panel, root=0)
        _, U_panel = lu(A_panel, permute_l=True)

        U_panel_final_root, U_panel_final = find_U_panel_final(info, U_panel, current_iteration)
        info.comm.Bcast(U_panel_final, root=U_panel_final_root)

        # Compute the column for the L matrix corresponding to this iteration
        L_column_element = np.zeros((info.submat_size, info.submat_size))
        if (info.rank >= current_iteration):
            L_column_element = np.dot(A_panel, np.linalg.inv(U_panel_final))

        L_column = np.zeros((info.mat_size, info.submat_size))
        info.comm.Allgather(L_column_element, L_column)
        if info.rank == 0:
            L[0:info.mat_size, current_iteration * info.submat_size:current_iteration * info.submat_size + info.submat_size] = L_column

        
        # Compute the row for the U matrix corresponding to this iteration
        L_column_element_current = L_column[info.submat_size * current_iteration:info.submat_size * current_iteration + L_column_element.shape[0], 0 : L_column_element.shape[1]]
        T = np.linalg.inv(L_column_element_current)
       
        '''
        T = np.zeros((info.submat_size, info.submat_size))
        if (info.rank == current_iteration):
            T = np.linalg.inv(L_column_element)

        info.comm.Bcast(T, root=current_iteration)
        '''

        U_row_element = np.zeros((info.submat_size, info.submat_size))
        if info.rank == current_iteration:
            U_row_element = U_panel_final
        elif info.rank > current_iteration:
            A_row_element = A[info.submat_size * current_iteration:info.submat_size * current_iteration + info.submat_size, info.submat_size * info.rank:info.submat_size * info.rank + info.submat_size]
            U_row_element = np.dot(T, A_row_element)

        U_row_col = np.zeros((info.mat_size, info.submat_size))
        info.comm.Allgather(U_row_element, U_row_col)
        U_row = col_to_row(info, U_row_col)

        if info.rank == 0:
            U[current_iteration * info.submat_size:current_iteration * info.submat_size + info.submat_size, 0:info.mat_size] = U_row

        # Update the A matrix for this iteration
        update_row_A = A[:, info.submat_size * info.rank:info.submat_size * info.rank + info.submat_size]
        update_U = U_row_col[info.rank * info.submat_size: info.rank * info.submat_size + info.submat_size, 0: info.submat_size]

        if (info.rank > current_iteration):
            for i in range(current_iteration + 1, info.num_iterations):
                update_A = update_row_A[i * info.submat_size: i * info.submat_size + info.submat_size, 0: info.submat_size]
                update_L =L_column[i * info.submat_size: i * info.submat_size + info.submat_size, 0: info.submat_size]

                update_row_A[i * info.submat_size: i * info.submat_size + info.submat_size, 0: info.submat_size] = update_A - np.dot(update_L, update_U)

        update_A_final = np.zeros((info.mat_size * info.num_proc, info.submat_size))
        info.comm.Allgather(update_row_A.ravel(), update_A_final)
        for i in range(info.num_proc):
            A[:, info.submat_size * i:info.submat_size * i + info.submat_size] = update_A_final[info.mat_size * i: info.mat_size * i + info.mat_size, :]

    if (info.rank == 0):
        last_A = A[info.mat_size - info.submat_size:info.mat_size, info.mat_size - info.submat_size:info.mat_size]
        last_L, last_U = lu(last_A, permute_l=True)
        L[info.mat_size - info.submat_size:info.mat_size, info.mat_size - info.submat_size:info.mat_size] = last_L
        U[info.mat_size - info.submat_size:info.mat_size, info.mat_size - info.submat_size:info.mat_size] = last_U

    return L, U


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_proc = comm.Get_size()

    mat_size = args["mat_size"]

    if mat_size % (num_proc * 2) != 0:
        print("The size of the matrix (%d) cannot be divided by the number of process (%d). Returning" % (mat_size, num_proc))
        return

    submat_size = mat_size // num_proc
    num_iterations = num_proc
    size_log2 = math.ceil(math.log2(mat_size))
    merge_rank = num_iterations - rank - 1

    Z = np.zeros((submat_size, submat_size))
    
    if rank == 0:
        A = init_random_mat(mat_size, mat_size)
    else:
        A = np.zeros((mat_size, mat_size))

    comm.Bcast(A, root=0)

    info = Info(comm, rank, merge_rank, num_proc, mat_size, submat_size, num_iterations, size_log2, A, Z)

    time_start = time.time()
    L, U = resolve_lu(info)

    if rank == 0:
        time_end = time.time()
        elapsed_time = time_end - time_start
        if check_correct(A, L, U):
            print("Execution time: %f" % elapsed_time)



#mpiexec -n 4 python main --mat_size 256
if __name__ == '__main__':
    main()
