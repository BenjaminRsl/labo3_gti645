from doctest import Example
from mpi4py import MPI
import numpy as np
from scipy.linalg import blas
from scipy.linalg import lapack
import math
from scipy.linalg import lu



"""
    Initialiser la matrice entre 0 et 100
"""
def matrice_init(n):
    matrice = np.random.rand(n, n) * 100
    return matrice


"""
    Code de vérification de la décomposition LU    
"""
def verification(L, U, A):
    result = np.dot(L, U) - A

    if np.all(result < 1e-4):
        print("Correct ;)")
        return True
    else:
        print("Incorrect :(")
        return False




"""
    Décomposition LU qui foncionne pour n = 3
"""
def exemple():
    L = [[1,0,0],[2,1,0],[3,2,1]]
    U = [[4,3,2],[0,1,-2],[0,0,2]]
    A = [[4, 3, 2], [8, 7, 2], [12, 11, 4]]

    L = np.array(L)
    U = np.array(U)
    A = np.array(A)

    verification(L, U, A)



def parallele(A, i,comm):

    rank = comm.Get_rank()

    P = A[:,i:i+1]

    A_i= P[rank:rank+1,:]
    
    iteration = math.ceil(math.log2(A.shape[0]))
    
    for i in range(iteration):
        U_i = pannel(rank, A_i,i)


    if(rank == 0):
        comm.Bcast(U_i,0)
    else:
        U_i0 = np.zeros()
        comm.Bcast(U_i0,0)


    L_i = blas.dtrsm(1.0,A_i,U_i0)

    print(L_i)
    print(U_i)

    L_0 = np.zero()
    comm.Allreduce(L_0, L_i, op=MPI.CONCATENATE)

    maj(rank,L_0,A,i)




def pannel(rank, A_i, i):

    L_i, U_i = lapack.dgetrf(A_i)

    #Envoie à son voison si on est impaire
    if(rank % 2**(i+1) !=0):
        #envoie
        print(f"envoie à {rank - 2**i}")
        comm.Send(A_i, rank - 2**i, 0)
 
    else:
        #reception
        print(f"reception de {rank + 2**i}")
        U_voisin = np.zeros()
        comm.Recv(U_voisin, rank + 2**i, 0)
    
    V_i = [U_i,U_voisin]
    
    return V_i



def maj(rank,L_i,A,i):

    nb_ligne = A.shape[0]

    if(rank==0):
        T_0 = np.linalg.inv(L_i)
        comm.Bcast(L_i,0)
    else :
        T_i = np.zeros()
        comm.Bcast(T_i,0)
        A_i = A[i:i+1,rank:rank+1]
        U_i = T_i * A_i

        for j in range(2, nb_ligne, 2):
            A[j:j+1,rank:rank+1] = A[j:j+1,rank:rank+1] - L_i * U_i




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

taille = 4
saut = 2 

nb_travail= (taille // saut) // size # nombre de travail par processus
reste = (taille // saut) % size # reste de la division euclidienne

#A = matrice_init(taille)
A = [[1, 2, 3, 4], [8, 6, 7, 5], [5, 5 , 6, 6], [9, 4, 1, 7]]
L  = [[1,0,0,0],[8,1,0,0],[5,0.5,1,0],[9,1.4,4.4,1]]
U = [[1,2,3,4],[0,-10,-17,-27],[0,0,-0.5,-0.5],[0,0,0,11]]

L = np.array(L,dtype=np.float64)
U = np.array(U,dtype=np.float64)
A = np.array(A,dtype=np.float64)




lu, ipiv, info = lapack.dgetrf(A, overwrite_a=True)




L = np.tril(lu, k=-1) + np.eye(A.shape[0], A.shape[1], dtype=lu.dtype)
U = np.triu(lu)

print(lu,U,L)

verification(L, U, A)

#print(lu, ipiv, info)

'''
for i in range(0, taille, saut):
    parallele(A, i,comm)
'''