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



def parallele(saut, A, i,taille,modulo):

    rank = comm.Get_rank()   

    for j in range(0, A.shape[0], saut):

        print(f"[{rank}]", j, modulo, j % modulo,rank * saut , flush=True)

        if(j % modulo == rank * saut ):
            print( f"[{rank}]",j,modulo, flush=True)

            P = A[:,i:(i+1)+1]

            A_i= P[j:(j+1)+1,:]
            

            iteration = math.ceil(math.log2(A.shape[0] // saut ))
            
            U_i = np.zeros(A.shape[0])

            print(f"[{rank}] début de la boucle de {iteration} itérations", flush=True)
            for k in range(iteration):
                U_i = pannel(rank, A_i,k)



            U_i = comm.bcast(U_i,root=0)

            #U_i = np.array(U_i,dtype=np.float64)
            print(f"[{rank}] U_i : ", U_i, flush=True)
            



            L_i = blas.dtrsm(1.0,A_i,U_i)

            print("\n\n")
            print(f"[{rank}] L_i: ", L_i)
            print("\n\n")
            print(f"[{rank}] U_i : ", U_i)

            

            L_0 = np.zeros(A.shape[0])

            comm.Allgather(L_0, L_i)

            print("L_0 : ", L_0)


            exit()

            L_0 = L_0.reshape(L_i.shape)
            print("L_0 : ", L_0)
            comm.Barrier()

            maj(j,L_0,A,i)

            return U_i,L_i
 
def create_vector_type(n):
    # Créer un type MPI pour un vecteur de taille n
    return MPI.FLOAT.Create_contig(n).Commit()


def pannel(rank, A_i, i):

    L_i, U_i = lu(A_i,permute_l=True)

    print(f"[{rank}] L_i : ", L_i, flush=True)
    print(f"[{rank}] U_i : ", U_i, flush=True)


    print(f"[{rank}] i : ", i, flush=True)
    #Envoie à son voison si on est impaire
    if(rank % 2**(i+1) !=0):
        #envoie
        print(f"[{rank}] envoie à {rank - 2**i}", flush=True)
        comm.Send(U_i, dest=rank - 2**i, tag=0)
        return None
 
    else:
        #reception
        print(f"[{rank}] reception de {rank + 2**i}", flush=True)
        U_voisin = np.zeros(A.shape[0])
        comm.Recv(U_voisin,rank + 2**i, 0)
        U_voisin = U_voisin.reshape(U_i.shape)
        print(f"[{rank}] U_voisin : ", U_voisin, flush=True)
        V_i = [U_i,U_voisin]
        return V_i
    




def maj(rank,L_i,A,i):

    nb_ligne = A.shape[0]

    if(rank==0):
        T_0 = np.linalg.inv(L_i)
        comm.Bcast(T_0,0)
    else :
        T_i = np.zeros(A.shape[0])
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


nb_travail= math.ceil((taille // saut) / size) # nombre de travail par processus
modulo = size * saut # période




print(f"[{rank}]",taille, saut, size, flush=True)
print(f"[{rank}] travail :  ",nb_travail, flush=True)
reste = (taille // saut) % size # reste de la division euclidienne

#A = matrice_init(taille)
A = [[1, 2, 3, 4], [8, 6, 7, 5], [5, 5 , 6, 6], [9, 4, 1, 7]]
L  = [[1,0,0,0],[8,1,0,0],[5,0.5,1,0],[9,1.4,4.4,1]]
U = [[1,2,3,4],[0,-10,-17,-27],[0,0,-0.5,-0.5],[0,0,0,11]]

L = np.array(L,dtype=np.float64)
U = np.array(U,dtype=np.float64)
A = np.array(A,dtype=np.float64)


print(A[:2,:2])
L1, U1 = lu(A[:2,:2],permute_l=True)

print(A[2:4,:2])
L2, U2 = lu(A[2:4,:2],permute_l=True)

V1 = np.concatenate((U1,U2),axis=0)



L3 , U3 = lu(V1,permute_l=True)


U3_inv = np.linalg.inv(U3)


L4 = np.dot(A[:2,:2],U3_inv)
L5 = np.dot(A[2:4,:2],U3_inv)

L_1 = np.concatenate((L4,L5),axis=0)

print("L_1 :", L_1)

##

T1 = np.linalg.inv(L4)

print(T1)

U12 = T1 * A[:2,2:4]
A[2:4, 2:4] = A[2:4,2:4] - L5 * U2


L_2, U_2 = lu(A[2:4,2:4],permute_l=True)

Z = np.zeros( 2 )

L = [L_1, [Z, L_2]]
U = [[U3, Z], [U12,U_2]]

print("L : \n", L)

print("U : \n", U)



'''
comm.Barrier()
for i in range(0, taille, saut):
    parallele(saut, A, i,nb_travail,modulo)

print(f"[{rank}] Fin", flush=True)



'''