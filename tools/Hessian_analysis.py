import math
from typing import Dict, List, Sequence, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

def generate_hessian_matrices(E: float, p: int, r: int, m: int, N: int, *, seed: Optional[int] = None, return_parts: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(size=(N, 2 * m, p))

    W = np.matmul(X.transpose(0, 2, 1), X)
    Y = rng.standard_normal(size=(N, p, p))
    GOE = (Y + np.swapaxes(Y, 1, 2)) / np.sqrt(2.0)

    H = (r / m) * W + (r / m) * np.sqrt(2 * m * E) * GOE
    H = H - (2 * r * E) * np.eye(p)[None, :, :]
    if return_parts: return H, W, GOE
    return H, None, None

def plot_matrix_heatmaps(mats: Sequence[np.ndarray], titles: Sequence[str]) -> None:
    k = len(mats); plt.figure(figsize=(6 * k, 5))
    for i, (M, t) in enumerate(zip(mats, titles), 1):
        plt.subplot(1, k, i); plt.title(t); plt.imshow(M, cmap="viridis", aspect="auto"); plt.colorbar(label="Value"); plt.xlabel("Index"); plt.ylabel("Index")
    plt.tight_layout(); plt.show()

def plot_mean_matrices(energies: Sequence[float], m: int, p: int, r: int = 1, N: int = 1000, seed: Optional[int] = None):
    for E in energies:
        H, W, G = generate_hessian_matrices(E=E, p=p, r=r, m=m, N=N, seed=seed)
        mean_H = H.mean(axis=0); mean_W = W.mean(axis=0) if W is not None else None; mean_G = G.mean(axis=0) if G is not None else None
        mats, titles = [mean_H], [f"Mean Hessian at E={E:.3g}"]
        if mean_W is not None: mats.append(mean_W); titles.append(f"Mean W at E={E:.3g}")
        if mean_G is not None: mats.append(mean_G); titles.append(f"Mean GOE at E={E:.3g}")
        plot_matrix_heatmaps(mats, titles)

def positive_definiteness_analysis(energies: Sequence[float], p: int, m: int, r: int = 1, *, to_plot: bool = False, N: int = 1000, seed: Optional[int] = None, compute_det: bool = True) -> Tuple[List[float], List[float]]:
    prob_pd_list: List[float] = []; mean_det_list: List[float] = []
    for E in energies:
        H, _, _ = generate_hessian_matrices(E=E, p=p, r=r, m=m, N=N, seed=seed, return_parts=False)
        eigvals = np.linalg.eigvalsh(H); min_eig = eigvals.min(axis=1); prob_pd_list.append(float((min_eig > 0).mean()))
        mean_det_list.append(float(np.linalg.det(H).mean()) if compute_det else float("nan"))
    if to_plot:
        plt.figure(figsize=(10, 5)); plt.plot(energies, prob_pd_list, marker="o"); plt.xlabel("Energy E"); plt.ylabel("P(H is positive definite)"); plt.title(f"PD probability vs E (p={p}, m={m}, r={r}, N={N})"); plt.grid(True); plt.show()
    return prob_pd_list, mean_det_list

def sweep_pd_over_params(energies: Sequence[float], p_values: Sequence[int], m_values: Sequence[int], *, r: int = 1, N: int = 1000, seed: Optional[int] = None, plot: bool = True) -> Dict[Tuple[int, int], List[float]]:
    results: Dict[Tuple[int, int], List[float]] = {}
    for m in m_values:
        for p in p_values:
            prob_pd, _ = positive_definiteness_analysis(energies, p=p, m=m, r=r, N=N, seed=seed, to_plot=False, compute_det=False)
            results[(p, m)] = prob_pd
    if plot:
        plt.figure(figsize=(12, 8))
        for (p, m), prob_pd in results.items():
            plt.plot(energies, prob_pd, label=f"p={p}, m={m}")
        plt.xlabel("Energy E"); plt.ylabel("P(H is positive definite)"); plt.title(f"PD probability vs E for (p, m), r={r}, N={N}"); plt.legend(); plt.grid(True); plt.show()
    return results

def sweep_mean_det_over_params(energies: Sequence[float], p_values: Sequence[int], m_values: Sequence[int], *, r: int = 1, N: int = 1000, seed: Optional[int] = None, plot: bool = True) -> Dict[Tuple[int, int], List[float]]:
    results: Dict[Tuple[int, int], List[float]] = {}
    for m in m_values:
        for p in p_values:
            _, mean_det = positive_definiteness_analysis(energies, p=p, m=m, r=r, N=N, seed=seed, to_plot=False, compute_det=True)
            results[(p, m)] = mean_det
    if plot:
        num_plots = len(results); cols = 3; rows = (num_plots + cols - 1) // cols
        plt.figure(figsize=(6 * cols, 4 * rows))
        for i, ((p, m), mean_det) in enumerate(results.items(), 1):
            plt.subplot(rows, cols, i); plt.plot(energies, mean_det); plt.xlabel("Energy E"); plt.ylabel("Mean det(H)"); plt.title(f"Mean det(H) vs E (p={p}, m={m}, r={r}, N={N})"); plt.grid(True)
        plt.show()
    return results

def conditioned_mean_det(energies, p: int, m: int, r: int = 1, *, N: int = 1000, seed: int | None = None, to_plot: bool = True):
    means = []
    for E in energies:
        H, _, _ = generate_hessian_matrices(E=E, p=p, r=r, m=m, N=N, seed=seed, return_parts=False)
        eigvals = np.linalg.eigvalsh(H); mask = (eigvals.min(axis=1) >= 0)
        det_vals = np.zeros(H.shape[0], dtype=float)
        if np.any(mask): det_vals[mask] = np.linalg.det(H[mask])
        means.append(float(det_vals.mean()))
    if to_plot:
        plt.figure(figsize=(10, 5)); plt.plot(energies, means, marker="o"); plt.xlabel("Energy E"); plt.ylabel("Kac–Rice mean det(H)"); plt.title(f"Kac–Rice mean det(H) vs E (p={p}, m={m}, r={r}, N={N})"); plt.grid(True); plt.show()
    return means

def sweep_conditioned_mean_det(energies, p_values, m_values, *, r: int = 1, N: int = 1000, seed: int | None = None, plot: bool = True):
    results = {}
    for m in m_values:
        for p in p_values:
            means = conditioned_mean_det(energies, p=p, m=m, r=r, N=N, seed=seed, to_plot=False)
            results[(p, m)] = means
    if plot:
        num_plots = len(results); cols = 3; rows = (num_plots + cols - 1) // cols
        plt.figure(figsize=(6 * cols, 4 * rows)); plt.subplots_adjust(hspace=0.4, wspace=0.4)
        for i, ((p, m), means) in enumerate(results.items(), 1):
            plt.subplot(rows, cols, i); plt.plot(energies, means); plt.xlabel("Energy E"); plt.ylabel("Kac–Rice mean det(H)"); plt.title(f"p={p}, m={m}"); plt.grid(True)
        plt.show()
    return results

def main():
    MODE = "sweep_det"  # Options: "mean_matrices", "prob", "sweep_pd", "sweep_det", "cond_det", "sweep_cond_det"
    p = 4; m = 4; r = 1; N = 4000; seed = None
    emin, emax, nE = 1e-9, 0.5, 100
    energies = np.linspace(emin, emax, num=nE)
    m_values = [4]
    p_values = [3,4,5,6,7,8]


    if MODE == "mean_matrices":
        plot_mean_matrices(energies, m=m, p=p, r=r, N=N, seed=seed)
    elif MODE == "prob":
        positive_definiteness_analysis(energies, p=p, m=m, r=r, to_plot=True, N=N, seed=seed, compute_det=False)

    elif MODE == "sweep_pd":
        sweep_pd_over_params(energies, p_values=p_values, m_values=m_values, r=r, N=N, seed=seed, plot=True)
    elif MODE == "sweep_det":
        sweep_mean_det_over_params(energies, p_values=p_values, m_values=m_values, r=r, N=N, seed=seed, plot=True)

    elif MODE == "cond_det":
        conditioned_mean_det(energies, p=p, m=m, r=r, N=N, seed=seed, to_plot=True)
    elif MODE == "sweep_cond_det":
        sweep_conditioned_mean_det(energies, p_values=p_values, m_values=m_values, r=r, N=N, seed=seed, plot=True)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

if __name__ == "__main__":
    main()
