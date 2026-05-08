import numpy as np
import matplotlib.pyplot as plt
import math

def get_exponent_alpha(m, p):
    """Calculates alpha = m - p/2 - 1"""
    return m - (p / 2) - 1

def shape_function_F(E, m, p):
    """
    Calculates F(E) = E^(m - p/2 - 1) * exp(-mE)
    This determines the shape of the distribution.
    """
    exponent = get_exponent_alpha(m, p)
    # We use np.power safely; for negative exponents, E=0 is singularity
    term1 = np.power(E, exponent) 
    term2 = np.exp(-m * E)
    return term1 * term2

def full_prefactor_P(E, m, p):
    """
    Calculates the full Pre-factor P(E) including constants.
    P(E) = C * F(E)
    """
    # 1. Calculate Constant C
    # C = (m / 4*pi)^(p/2) * (m^m / Gamma(m)) * (2*pi)^p
    term_a = (m / (4 * np.pi))**(p / 2)
    term_b = (m**m) / math.gamma(m)
    term_c = (2 * np.pi)**p
    constant_C = term_a * term_b * term_c
    
    # 2. Get Shape F(E)
    F_E = shape_function_F(E, m, p)
    
    return constant_C * F_E

def plot_regime_grid(m, p_list, func_type='shape', E_max=0.5):
    """
    Plots a 2x3 grid of subfigures.
    func_type: 'shape' for F(E), 'full' for P(E)
    """
    # Avoid E=0 exactly to prevent divide by zero errors in code, 
    # though mathematically it diverges.
    E = np.linspace(1e-5, E_max, 500)
    
    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    axes = axes.flatten()
    
    # Dictionary for labels based on type
    if func_type == 'shape':
        main_title = f"Shape Function $\\mathcal{{F}}(E) = E^{{m - p/2 - 1}} e^{{-mE}}$ (m={m})"
        y_label = "$\\mathcal{F}(E)$"
    else:
        main_title = f"Full Prefactor $P(E)$ (m={m})"
        y_label = "$P(E)$"

    for idx, p in enumerate(p_list):
        if idx >= len(axes): break
        
        ax = axes[idx]
        alpha = get_exponent_alpha(m, p)
        
        # Calculate Y values
        if func_type == 'shape':
            y = shape_function_F(E, m, p)
        else:
            y = full_prefactor_P(E, m, p)

        # Plot
        ax.plot(E, y, linewidth=2, color='royalblue')
        
        # Determine Regime for Title
        if alpha > 0:
            regime = "Underparameterized"
        elif alpha == 0:
            regime = "Transition Point"
        else:
            regime = "Overparameterized"

        ax.set_title(f"$p={p}$  |  $\\alpha={alpha}$ \n({regime})", fontsize=10, fontweight='bold')
        ax.set_xlabel("Energy $E$")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        
        # Handle Y-axis limits for divergent cases to keep plots readable
        if alpha < 0:
            # Zoom in on the curve part, cut off the infinite spike
            # We pick a reasonable y-max based on the value at E=0.05
            if func_type == 'shape':
                y_at_slice = shape_function_F(np.array([0.05]), m, p)[0]
                ax.set_ylim(0, y_at_slice * 3) 
            else:
                y_at_slice = full_prefactor_P(np.array([0.05]), m, p)[0]
                ax.set_ylim(0, y_at_slice * 3)

    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # Parameters from your notes
    m = 4
    
    # We use 6 values to fill the 2x3 grid.
    # Notes used 3, 4, 5, 6, 7. We add 8 to show strong divergence.
    p_values = [3, 4, 5, 6, 7, 8] 
    
    print(f"Plotting for m={m} with p values: {p_values}")
    
    # 1. Plot Shape F(E)
    plot_regime_grid(m, p_values, func_type='shape', E_max=0.5)
    
    # 2. Plot Full Prefactor P(E)
    #plot_regime_grid(m, p_values, r=r, func_type='full', E_max=0.5)