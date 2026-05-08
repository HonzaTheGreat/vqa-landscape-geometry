# visualize_single_csv.py
import argparse
import csv
import sys
import pandas as pd
import matplotlib.pyplot as plt

from tools.Plotting import (
    plot_all_results_together,
    plot_all_results_logy,
    plot_all_results_combined_logy,
    plot_all_results_combined_normalized,
    plot_multiple_results,
    plot_multiple_results_logy,
    plot_multiple_results_combined,
    plot_multiple_results_combined_logy,
)

def main(csv_path, csv_filename):
    try:
        df = pd.read_csv(csv_path + csv_filename)
    except FileNotFoundError:
        print(f"File '{csv_path}' not found.")
        sys.exit(1)

    # Energies
    energies = df["Energy"].values

    # Detect available p-values from KacRice columns
    kr_cols = [c for c in df.columns if c.startswith("KacRice_p")]
    if not kr_cols:
        print("No 'KacRice_p*' columns found in this CSV.")
        sys.exit(1)

    p_values = sorted(int(c.replace("KacRice_p", "")) for c in kr_cols)

    # Build dictionaries for each metric
    all_results     = {p: df[f"KacRice_p{p}"].values     for p in p_values}
    all_probs       = {p: df[f"ProbPD_p{p}"].values      for p in p_values}
    all_prefactors  = {p: df[f"Prefactor_p{p}"].values   for p in p_values}
    all_raw         = {p: df[f"RawDet_p{p}"].values      for p in p_values}

    # ---------- Plot ----------
    # Kac-Rice plots
    plot_all_results_together(all_results, energies, p_values)

    # Combined normalized overlay for nicer comparison across p
    #plot_all_results_combined_normalized(all_results, energies, title='Kac-Rice Estimates (normalized)')

    # Probability plots
    # plot_multiple_results(energies, all_probs, 'P[PD Hessian]', 'Probability of Positive Definite Hessians', fig_id=4)
    
    # Determinant plots
    # plot_multiple_results_combined(energies, all_raw, 'Mean |det(H)|', 'Mean Determinant Combined', fig_id=5)
    # plot_all_results_together(all_raw, energies, p_values, fig_id=6)
    #plot_multiple_results_combined_logy(energies, all_raw, '|det(H)|', 'Mean Determinant Combined (log)', fig_id=7)
    

    plt.show()   


if __name__ == "__main__":
    csv_filename = "data/m_32/KacRice_m_32_p_50_64_E_1e-08_0.01_lin_N200_samples_100000.csv"
    csv_path = ""

    main(csv_path, csv_filename)
