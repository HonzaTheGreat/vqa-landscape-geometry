import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_hessian_matrices(E, p, r, m, N, rng):
    X = rng.normal(size=(N, 2 * m, p))
    W = np.matmul(X.transpose(0, 2, 1), X)

    Y = rng.normal(size=(N, p, p))
    GOE = (Y + Y.transpose(0, 2, 1)) / np.sqrt(2.0)

    H = (r / m) * W + (r / m) * np.sqrt(2 * m * E) * GOE
    H -= 2 * r * E * np.eye(H.shape[-1])[None, :, :]
    return H


def kac_rice_formula(E, p, m, r, num_samples, rng):
    Hessians = generate_hessian_matrices(E, p, r, m, num_samples, rng)
    eigvals = np.linalg.eigvalsh(Hessians)
    min_eigvals = eigvals.min(axis=1)

    prob_pd = np.mean(min_eigvals > 0)

    det_vals = np.linalg.det(Hessians)
    det_vals[min_eigvals <= 0] = 0
    mean_det = np.mean(det_vals) if len(det_vals) > 0 else 0

    prefactor = ((m / (4 * np.pi * r * E)) ** (p / 2)) * (m**m / math.gamma(m)) * (E ** (m - 1)) * np.exp(-m * E) * (
        (2 * np.pi) ** p
    )
    full_result = prefactor * mean_det
    return full_result, prob_pd, prefactor, mean_det


def compute_kac_rice_for_energies(energies, p, m, r, num_samples, rng):
    results, probs, prefactors, raw_dets = [], [], [], []
    for E in energies:
        res, prob, pref, raw = kac_rice_formula(E, p, m, r, num_samples, rng)
        results.append(res)
        probs.append(prob)
        prefactors.append(pref)
        raw_dets.append(raw)
    return results, probs, prefactors, raw_dets


def make_energies(emin, emax, num, spacing="lin"):
    if spacing.lower().startswith("geo"):
        return np.geomspace(max(emin, 1e-12), emax, num=num)
    return np.linspace(emin, emax, num=num)


def save_results_to_csv(energies, all_results, all_probs, all_prefactors, all_raw, p_values, set_name, m, emin, emax, spacing, num_E, num_samples):
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(
        {
            "Energy": energies,
            **{f"KacRice_p{p}": all_results[p] for p in p_values},
            **{f"ProbPD_p{p}": all_probs[p] for p in p_values},
            **{f"Prefactor_p{p}": all_prefactors[p] for p in p_values},
            **{f"RawDet_p{p}": all_raw[p] for p in p_values},
        }
    )

    fname = f"KacRice_{set_name}_m_{m}_p_{p_values[0]}_{p_values[-1]}_E_{emin:g}_{emax:g}_{spacing}_N{num_E}_samples_{num_samples}.csv"
    data_path = os.path.join("data", fname)
    df.to_csv(data_path, index=False)
    print(f"  -> Saved: {data_path}")


STORE_RESULTS = False
PLOT_RESULTS = True
R = 1
SEED = None
SAMPLE_SIZES = [1000]

PARAMETER_SETS = [
    {
        "name": "set2",
        "m": 32,
        "p_values": [50, 54, 58, 62, 63, 64],
        "emin": 1e-8,
        "emax": 0.01,
        "num_energies": 200,
        "spacing": "lin",
    },
]


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    for ps in PARAMETER_SETS:
        set_name = ps.get("name", f"m{ps['m']}")
        m = ps["m"]
        p_values = ps["p_values"]
        emin = ps.get("emin", 1e-9)
        emax = ps.get("emax", 0.2)
        num_E = ps.get("num_energies", 200)
        spacing = ps.get("spacing", "lin")
        energies = make_energies(emin, emax, num=num_E, spacing=spacing)

        overlay_results = {p: {} for p in p_values}

        for num_samples in SAMPLE_SIZES:
            print(f"\n=== Running {set_name} | m={m} | samples={num_samples} | E∈[{emin}, {emax}] ({spacing}, {num_E} pts) ===")
            all_results, all_probs, all_prefactors, all_raw = {}, {}, {}, {}

            for p in p_values:
                print(f"  -> Computing for p = {p}, m = {m}")
                results, probs, prefactors, raw_dets = compute_kac_rice_for_energies(energies, p, m, R, num_samples, rng)
                all_results[p] = results
                all_probs[p] = probs
                all_prefactors[p] = prefactors
                all_raw[p] = raw_dets
                overlay_results[p][num_samples] = results

            if STORE_RESULTS:
                save_results_to_csv(
                    energies,
                    all_results,
                    all_probs,
                    all_prefactors,
                    all_raw,
                    p_values,
                    set_name,
                    m,
                    emin,
                    emax,
                    spacing,
                    num_E,
                    num_samples,
                )

        if PLOT_RESULTS:
            num_plots = len(p_values)
            cols = int(np.ceil(np.sqrt(num_plots)))
            rows = int(np.ceil(num_plots / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes = axes.flatten() if num_plots > 1 else [axes]

            for i, p in enumerate(p_values):
                ax = axes[i]
                for s in sorted(overlay_results[p].keys()):
                    ax.plot(energies, overlay_results[p][s], label=f"N={s}")
                ax.set_title(f"p = {p}")
                ax.set_xlabel("Energy (E)")
                ax.set_ylabel("E[Crt0(E)]")
                ax.grid(True)
                ax.legend()

            for ax in axes[num_plots:]:
                ax.axis("off")

            fig.tight_layout(rect=[0, 0.03, 1, 0.92])
            fig.subplots_adjust(hspace=0.4)

    if PLOT_RESULTS:
        plt.show()
