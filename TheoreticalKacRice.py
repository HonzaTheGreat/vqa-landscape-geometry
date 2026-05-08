import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

from tools.Plotting import plot_all_results_together, plot_multiple_results


def prefactor_func(E, m, p, r=1):
    return ((m / (4*np.pi*r*E))**(p/2)) * (m**m / math.gamma(m)) * (E**(m-1)) * np.exp(-m*E) * ((2*np.pi)**p)


def kac_rice_formula(E, p, m, r, num_samples):
    mean_Hessian =  2*r *(1-E) * np.eye(p)  

    # # Plot the mean Hessian heatmap
    # plt.figure(figsize=(6, 5))
    # plt.title(f"Mean Hessian at E={E:.3g}, p={p}, m={m}, r={r}")
    # plt.imshow(mean_Hessian, cmap="viridis", aspect="auto")
    # plt.colorbar(label="Value")
    # plt.xlabel("Index")
    # plt.ylabel("Index")
    # plt.tight_layout()
    # plt.show()

    mean_det = np.linalg.det(mean_Hessian)

    # if mean_det < 0:
    #     mean_det = 0

    prefactor = prefactor_func(E, m, p, r)
    full_result = prefactor * mean_det
    return full_result, prefactor, mean_det

def compute_kac_rice_for_energies(energies, p, m, r, num_samples):
    results, probs, prefactors, raw_dets = [], [], [], []
    for E in energies:
        res, pref, mean = kac_rice_formula(E, p, m, r, num_samples)
        results.append(res)
        prefactors.append(pref)
        raw_dets.append(mean)
    return results, prefactors, raw_dets



if __name__ == "__main__":
    store_results = False
    plot_results = True

    r = 1
    min_energy = 1e-9
    max_energy = 0.5
    num_energies = 100
    num_samples = 500

    # Generate linear spaced energies
    energies = np.linspace(min_energy, max_energy, num=num_energies)
    # energies = np.geomspace(min_energy, max_energy, num=num_energies)


    m_min = 4
    m_max = 4
    p_count = 4

    for m in range(m_min, m_max + 1):
        p_center = 2 * m
        p_values = [p_center + offset for offset in range(-p_count, p_count + 1)]
        p_values = [3,4,5,6,7,8]

        all_results, all_probs, all_prefactors, all_raw = {}, {}, {}, {}

        for p in p_values:
            print(f"Computing for p = {p}, m = {m}")
            results, prefactors, raw_dets = compute_kac_rice_for_energies(energies, p, m, r, num_samples)
            all_results[p] = results
            all_prefactors[p] = prefactors
            all_raw[p] = raw_dets


        # Store results to data folder
        if store_results:
            df = pd.DataFrame({**{'Energy': energies},
                              **{f'KacRice_p{p}': all_results[p] for p in p_values},
                              **{f'ProbPD_p{p}': all_probs[p] for p in p_values},
                              **{f'Prefactor_p{p}': all_prefactors[p] for p in p_values},
                              **{f'RawDet_p{p}': all_raw[p] for p in p_values}})
            df.to_csv(f"Extended_KacRice_p_{p_values[0]}_{p_values[-1]}_m_{m}.csv", index=False)


        if plot_results:
            # Plotting all results together
            plot_all_results_together(all_results, energies, p_values, fig_id=1)    
            # plot_all_results_together(all_raw, energies, p_values, fig_id=2)   
            plt.show()