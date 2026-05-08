
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_all_results_together(all_results, energies, p_values, fig_id=None):
    if fig_id is not None:
        plt.figure(fig_id)
    num_plots = len(p_values)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, p in enumerate(p_values):
        ax = axes[i]
        ax.plot(energies, all_results[p], '-b')
        ax.set_title(f'p = {p}')
        ax.set_xlabel('Energy (E)')
        ax.grid(True)

    for ax in axes[num_plots:]:
        ax.axis('off')

    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.subplots_adjust(hspace=0.4)


def plot_all_results_logy(all_results, energies, p_values, fig_id=None):
    if fig_id is not None:
        plt.figure(fig_id)
    num_plots = len(p_values)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, p in enumerate(p_values):
        ax = axes[i]
        ax.plot(energies, all_results[p], '-b')
        ax.set_yscale('log')
        ax.set_title(f'p = {p}')
        ax.set_xlabel('Energy (E)')
        ax.set_ylabel('log E[Crt₀(E)]')
        ax.grid(True, which='both', linestyle='--')

    for ax in axes[num_plots:]:
        ax.axis('off')

    fig.suptitle('Log-Scale Kac-Rice Estimates', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.subplots_adjust(hspace=0.4)


def plot_all_results_combined_logy(all_results, energies, title='Kac-Rice Estimates (log scale)', fig_id=None):
    if fig_id is not None:
        plt.figure(fig_id)
    plt.figure(figsize=(8, 6))
    for p, values in all_results.items():
        plt.plot(energies, values, label=f'p = {p}')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Energy (E)')
    plt.ylabel('log E[Crt₀(E)]')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()


def plot_multiple_results(energies, data_dict, ylabel, title, fig_id=None):
    if fig_id is not None:
        plt.figure(fig_id)
    num_plots = len(data_dict)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, (p, values) in enumerate(data_dict.items()):
        ax = axes[i]
        ax.plot(energies, values, '-b')
        ax.set_title(f'p = {p}')
        ax.set_xlabel('Energy (E)')
        ax.set_ylabel(ylabel)
        ax.grid(True)

    for ax in axes[num_plots:]:
        ax.axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.subplots_adjust(hspace=0.4)


def plot_multiple_results_logy(energies, data_dict, ylabel, title, fig_id=None):
    if fig_id is not None:
        plt.figure(fig_id)
    num_plots = len(data_dict)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, (p, values) in enumerate(data_dict.items()):
        ax = axes[i]
        ax.plot(energies, values, '-b')
        ax.set_yscale('log')
        ax.set_title(f'p = {p}')
        ax.set_xlabel('Energy (E)')
        ax.set_ylabel(f'log {ylabel}')
        ax.grid(True, which='both', linestyle='--')

    for ax in axes[num_plots:]:
        ax.axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.subplots_adjust(hspace=0.4)


def plot_multiple_results_combined(energies, data_dict, ylabel, title, fig_id=None):
    if fig_id is not None:
        plt.figure(fig_id)
    plt.figure(figsize=(8, 6))
    for p, values in data_dict.items():
        plt.plot(energies, values, label=f'p = {p}')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('Energy (E)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_multiple_results_combined_logy(energies, data_dict, ylabel, title, fig_id=None):
    if fig_id is not None:
        plt.figure(fig_id)
    plt.figure(figsize=(8, 6))
    for p, values in data_dict.items():
        plt.plot(energies, values, label=f'p = {p}')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Energy (E)')
    plt.ylabel(f'log {ylabel}')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.tight_layout()


def plot_all_results_combined_normalized(all_results, energies, title='Kac-Rice Estimates (normalized)', fig_id=None):
    """
    Plot all p-series in a single figure with each series normalized by its own max absolute value.

    Inputs:
    - all_results: dict[int, np.ndarray] mapping p -> values over energies
    - energies: 1D array-like of energy values

    Behavior:
    - Each series y is transformed to y / max(|y|) if max(|y|) > 0, else left unchanged.
    - Produces a single combined plot with legend across p values.
    """
    if fig_id is not None:
        plt.figure(fig_id)
    plt.figure(figsize=(8, 6))

    for p, values in all_results.items():
        arr = np.asarray(values)
        max_abs = np.nanmax(np.abs(arr)) if arr.size > 0 else 0.0
        if max_abs > 0:
            norm = arr / max_abs
        else:
            norm = arr  # avoid divide-by-zero; keep as-is if all zeros
        plt.plot(energies, norm, label=f'p = {p}')

    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel('Energy (E)')
    plt.ylabel('Normalized E[Crt₀(E)]')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
