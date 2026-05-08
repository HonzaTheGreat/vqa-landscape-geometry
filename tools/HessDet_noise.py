import numpy as np
import matplotlib.pyplot as plt

def theoretical_hessian_mean(E, p, m, r=1):
    """Compute the theoretical mean of the Hessian matrix."""
    return 2 * r * (1 - E) * np.eye(p)

def generate_hessian_matrices(E, p, r, m, N, seed=None):
    rng = np.random.default_rng(seed)

    # Wishart part: W = X^T X, X ~ N(0,1)^{(2m) x p}
    X = rng.standard_normal(size=(N, 2*m, p))
    W = np.matmul(X.transpose(0, 2, 1), X)  # (N,p,p)

    # GOE part: symmetric with offdiag var=1, diag var=2
    Y = rng.standard_normal(size=(N, p, p))
    GOE = (Y + np.swapaxes(Y, 1, 2)) / np.sqrt(2.0)

    # Assemble H
    H = (r/m) * W + (r/m) * np.sqrt(2*m*E) * GOE
    H = H - (2*r*E) * np.eye(p)[None, :, :]  # correct diagonal shift
    return H

def plot_mean_heatmaps(matrices, titles, E0, p, m, r, N_list):
    # Common color scale for fair visual comparison
    vmin = min(M.min() for M in matrices)
    vmax = max(M.max() for M in matrices)

    # Plot heatmaps
    n = len(titles); ncols = 3; nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, N, M in zip(axes, N_list, matrices):
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"N = {N}")
        ax.set_xlabel("j"); ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(len(titles), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Mean Hessian heatmaps at E={E0} (p={p}, m={m}, r={r})", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    p, m, r = 16, 8, 1.0
    energies = np.linspace(0.0, 0.5, 201)
    E0 = 0.1                    # pick an energy to inspect
    N_list = [100, 500, 1000, 5000, 10000, 50000]
    seed = 42                   # for reproducibility

    # Compute mean Hessians for different N
    mean_mats = []
    for N in N_list:
        H = generate_hessian_matrices(E0, p, r, m, N, seed=seed)
        mean_mats.append(H.mean(axis=0))

    # Plot heatmaps
    titles = [f"N={N}" for N in N_list]
    plot_mean_heatmaps(mean_mats, titles, E0, p, m, r, N_list)

    # Difference from theoretical mean
    H_theory = theoretical_hessian_mean(E0, p, m, r)
    diff_mats = [M - H_theory for M in mean_mats]
    diff_titles = [f"N={N}, diff from theory" for N in N_list]
    plot_mean_heatmaps(diff_mats, diff_titles, E0, p, m, r, N_list)
