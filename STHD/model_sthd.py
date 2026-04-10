import numpy as np
import squidpy as sq
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def fill_F(X, Y, Z, N, Mu, Var, F):
    for a in prange(X):
        for t in range(Z):
            for g in range(Y):
                diff = N[a, g] - Mu[t, g]
                F[a, t] = F[a, t] - 0.5 * (diff * diff) / Var[g]

def prepare_constants(sthd_data, anisotropic=True):
    X = sthd_data.adata.obs.shape[0]  
    Y = sthd_data.adata.shape[1]  
    Z = sthd_data.lambda_cell_type_by_gene_matrix.shape[0]  

    N = sthd_data.adata.to_df().values.astype("float32")
    Mu = sthd_data.lambda_cell_type_by_gene_matrix.astype("float32")  
    Var = np.var(N, axis=0).astype("float32") + 1e-6

    sq.gr.spatial_neighbors(
        sthd_data.adata, spatial_key="spatial", coord_type="generic", delaunay=True
    )
    A_csr = sthd_data.adata.obsp["spatial_connectivities"].copy()
    
    if anisotropic:
        rows, cols = A_csr.nonzero()
        diffs = N[rows] - N[cols]
        sq_dists = np.sum((diffs ** 2) / Var, axis=1)
        median_dist = np.median(sq_dists)
        if median_dist == 0: median_dist = 1.0 
        similarities = np.exp(-sq_dists / median_dist)
        A_csr.data = similarities.astype("float32")
    else:
        A_csr.data = np.ones_like(A_csr.data, dtype="float32")

    Acsr_row = A_csr.indptr
    Acsr_col = A_csr.indices
    Acsr_data = A_csr.data  

    F = np.zeros([X, Z], dtype="float32")
    fill_F(X, Y, Z, N, Mu, Var, F)
    
    return X, Y, Z, F, Acsr_row, Acsr_col, Acsr_data

def prepare_training_weights(X, Z):
    W = np.zeros([X, Z], dtype="float32")
    eW = np.zeros([X, Z], dtype="float32")
    P = np.ones([X, Z], dtype="float32") / Z
    Phi = np.zeros([X], dtype="float32")
    grad = np.zeros([X, Z], dtype="float32")
    m = np.zeros([X, Z], dtype="float32")
    v = np.zeros([X, Z], dtype="float32")
    return W, eW, P, Phi, grad, m, v

def train(n_iter, step_size, beta, constants, weights, beta1=0.9, beta2=0.999, epsilon=1e-8):
    X, Y, Z, F, Acsr_row, Acsr_col, Acsr_data = constants
    W, eW, P, Phi, grad, m, v = weights

    for i in range(n_iter):
        update_softmax(W, eW, Phi, P, X, Z)
        calculate_gradients(P, F, Acsr_row, Acsr_col, Acsr_data, grad, X, Z, beta)
        update_adam(W, grad, m, v, beta1, beta2, i+1, step_size, epsilon, X, Z)
        
        ll_prot, ce_space = calculate_losses(P, F, Acsr_row, Acsr_col, Acsr_data, X, Z)
        total_loss = -ll_prot + (beta * ce_space)
        print(f"Iter {i} | Total: {total_loss:.4f} | LL: {ll_prot:.4f} | CE: {ce_space:.4f}")

    return P

@njit(parallel=True, fastmath=True)
def update_softmax(W, eW, Phi, P, X, Z):
    for a in prange(X):
        Phi[a] = 0.0
        for t in range(Z):
            eW[a, t] = np.exp(W[a, t])
            Phi[a] += eW[a, t]
        for t in range(Z):
            P[a, t] = eW[a, t] / Phi[a]

@njit
def csr_obtain(row, column, data, i):
    return column[row[i]:row[i+1]], data[row[i]:row[i+1]]

@njit(parallel=True, fastmath=True)
def calculate_gradients(P, F, Acsr_row, Acsr_col, Acsr_data, grad, X, Z, beta):
    for a in prange(X):
        expected_F = 0.0
        for t in range(Z): expected_F += P[a, t] * F[a, t]
            
        neighbors, weights = csr_obtain(Acsr_row, Acsr_col, Acsr_data, a)
        
        for t in range(Z):
            ll_grad = F[a, t] - expected_F
            
            comp1 = 0.0
            comp2 = 0.0
            for i, a_star in enumerate(neighbors):
                w = weights[i]
                cur1 = 0.0
                for t2 in range(Z): cur1 += P[a, t2] * np.log(P[a_star, t2] + 1e-8)
                comp1 += w * (np.log(P[a_star, t] + 1e-8) - cur1)
                comp2 += w * (P[a_star, t] - P[a, t])
            space_grad = P[a, t] * comp1 + comp2
            
            grad[a, t] = -(ll_grad - beta * space_grad) / X

@njit(parallel=True, fastmath=True)
def update_adam(W, grad, m, v, beta1, beta2, i, alpha, epsilon, X, Z):
    for a in prange(X):
        for t in range(Z):
            m[a, t] = m[a, t] * beta1 + (1 - beta1) * grad[a, t]
            v[a, t] = v[a, t] * beta2 + (1 - beta2) * (grad[a, t]**2)
            m_hat = m[a, t] / (1 - beta1**i)
            v_hat = v[a, t] / (1 - beta2**i)
            W[a, t] = W[a, t] - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

@njit(parallel=True, fastmath=True)
def calculate_losses(P, F, Acsr_row, Acsr_col, Acsr_data, X, Z):
    ll_prot = 0.0
    ce_space = 0.0
    for a in prange(X):
        for t in range(Z): ll_prot += P[a, t] * F[a, t]
        neighbors, weights = csr_obtain(Acsr_row, Acsr_col, Acsr_data, a)
        for t in range(Z):
            cur = 0.0
            for i, a_star in enumerate(neighbors):
                cur += weights[i] * np.log(P[a_star, t] + 1e-8)
            ce_space -= P[a, t] * cur
    return ll_prot / X, ce_space / X