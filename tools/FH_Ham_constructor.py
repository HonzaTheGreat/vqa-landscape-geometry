import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time
import itertools

# --- Constants for Pauli Matrices ---
I2 = csr_matrix(np.eye(2, dtype=complex))
X2 = csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
Y2 = csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=complex))
Z2 = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))

def kron_list(op_list):
    """Computes Kronecker product of a list of matrices efficiently."""
    res = op_list[0]
    for op in op_list[1:]:
        res = kron(res, op)
    return res

def construct_fermi_hubbard(sites, t, U):
    """Constructs 1D Fermi-Hubbard Hamiltonian (OBC)."""
    N = 2 * sites
    dim = 2**N
    H = csr_matrix((dim, dim), dtype=complex)
    
    # 1. Hopping Terms (Nearest Neighbor)
    for spin in [0, sites]: 
        for i in range(sites - 1):
            ops_x, ops_y = [I2] * N, [I2] * N
            ops_x[spin+i] = ops_x[spin+i+1] = X2
            ops_y[spin+i] = ops_y[spin+i+1] = Y2
            H -= 0.5 * t * (kron_list(ops_x) + kron_list(ops_y))

    # 2. Interaction Terms (On-site Coulomb)
    for i in range(sites):
        u, v = i, i + sites
        ops_zu, ops_zv, ops_zuzv = [I2] * N, [I2] * N, [I2] * N
        ops_zu[u] = Z2
        ops_zv[v] = Z2
        ops_zuzv[u] = Z2; ops_zuzv[v] = Z2
        
        term = identity(dim) - kron_list(ops_zu) - kron_list(ops_zv) + kron_list(ops_zuzv)
        H += (U / 4.0) * term

    return H

def calculate_m_efficient(H):
    """Calculates effective degrees of freedom m using trace properties."""
    dim = H.shape[0]
    if dim == 0: return 0
    
    # Ground State Energy
    if dim < 16:
        lambda_1 = np.linalg.eigvalsh(H.toarray())[0]
    else:
        # Use shift-invert mode for robust lowest eigenvalue solving
        evals, _ = eigsh(H, k=1, which='SA')
        lambda_1 = evals[0]
        
    tr_H = H.diagonal().sum()
    nuc_norm = tr_H - (lambda_1 * dim)
    numerator = np.abs(nuc_norm)**2
    
    norm_H_sq = np.sum(np.abs(H.data)**2) 
    denominator = norm_H_sq - (np.abs(tr_H)**2 / dim)
    
    return (numerator / denominator).real if denominator != 0 else 0

def get_symmetry_projector(sites, use_particle=True, use_spin=False, use_spatial=False):
    """
    Builds a projection matrix restricting the Hilbert space based on chosen symmetries.
    Note: Projects into the symmetric (+1) parity sector of the chosen Z2 symmetries.
    """
    N = 2 * sites
    dim_full = 2**N
    
    if not use_particle:
        return identity(dim_full, format='csr'), dim_full

    n_up, n_down = sites // 2, sites // 2
    
    # Generate valid half-filling bitstrings
    up_states = [seq for seq in itertools.product([0, 1], repeat=sites) if sum(seq) == n_up]
    dw_states = [seq for seq in itertools.product([0, 1], repeat=sites) if sum(seq) == n_down]
    
    rows, cols, data = [], [], []
    visited = set()
    col_idx = 0
    
    for up in up_states:
        for dw in dw_states:
            state = tuple(up) + tuple(dw)
            if state in visited: continue
            
            # Generate the symmetry orbit (states that map to each other)
            orbit = {state}
            
            # Spatial Reflection (Mirror left-to-right)
            if use_spatial:
                orbit.update({p[:sites][::-1] + p[sites:][::-1] for p in orbit})
                
            # Spin-Flip Z2 (Swap Up and Down registers)
            if use_spin:
                orbit.update({p[sites:] + p[:sites] for p in orbit})
                
            # Create a symmetric superposition state for this orbit
            weight = 1.0 / np.sqrt(len(orbit))
            for p in orbit:
                visited.add(p)
                # Convert binary tuple back to sparse matrix index
                idx = sum(val * (2**i) for i, val in enumerate(p))
                rows.append(idx)
                cols.append(col_idx)
                data.append(weight)
                
            col_idx += 1
            
    P = csr_matrix((data, (rows, cols)), shape=(dim_full, col_idx))
    return P, col_idx

# --- Main Simulation Loop ---
if __name__ == "__main__":
    t_val = 1.0
    U_val = 2.0
    
    # Warning: Limit to 5 sites (10 qubits). 6 sites (12 qubits) for the "Full Space" 
    # will take a significant amount of time due to computing exact trace norms on 4096x4096.
    site_counts = [2, 3, 4, 5, 6, 7] 
    
    # Define which symmetries to stack and test
    scenarios = {
        "Full Hilbert Space":      {"use_particle": False, "use_spin": False, "use_spatial": False},
        "+ Particle Number":  {"use_particle": True,  "use_spin": False, "use_spatial": False},
        "+ Spin Parity":      {"use_particle": True,  "use_spin": True,  "use_spatial": False},
        "+ Spatial Parity":   {"use_particle": True,  "use_spin": True,  "use_spatial": True},
    }
    
    results = {name: [] for name in scenarios.keys()}

    print(f"Starting Simulation | t={t_val}, U={U_val}")
    print("="*65)

    start_total = time.time()

    for sites in site_counts:
        print(f"\n[Processing {sites} Sites ({2*sites} Qubits)]")
        t0 = time.time()
        
        # Build the massive, unconstrained Hamiltonian once
        H_full = construct_fermi_hubbard(sites, t_val, U_val)
        
        for name, config in scenarios.items():
            # Build projector and compress Hamiltonian
            P, dim_sub = get_symmetry_projector(sites, **config)
            
            # Apply the projection (identity matrix is returned if no symmetries are used)
            H_sub = P.T @ H_full @ P if config["use_particle"] else H_full
            
            # Compute degrees of freedom
            m_val = calculate_m_efficient(H_sub)
            results[name].append(m_val)
            
            print(f"  {name:<25} | Dim: {dim_sub:<5} | m: {m_val:.2e}")
            
        print(f"  (Took {time.time() - t0:.2f}s)")

    print("="*65)
    print(f"Total time: {time.time() - start_total:.2f}s")

    # --- Plotting ---
    plt.figure(figsize=(10, 7))
    markers = ['o', 's', '^', 'D']

    for (name, m_vals), marker in zip(results.items(), markers):
        plt.plot(site_counts, m_vals, marker=marker, linestyle='-', linewidth=2, label=name)

    plt.yscale('log')
    plt.xlabel('Number of Sites (L)', fontsize=12, fontweight='bold')
    plt.ylabel('Degrees of Freedom m', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Symmetry Sector", fontsize=10, title_fontsize=11)
    plt.xticks(site_counts) 
    
    plt.tight_layout()
    plt.show()