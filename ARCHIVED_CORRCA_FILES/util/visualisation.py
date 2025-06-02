import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

def old_plot_multicondition_rr_profiles(condition_data, lags, 
                                  colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
                                  condition_names=['Condition 1', 'Condition 2', 'Condition 3'],
                                  title='Recurrence Rate Profiles by Condition',
                                  xlabel='Lag (seconds)',
                                  ylabel='Recurrence Rate (RR)'):
    """
    Plot RR profiles for three conditions with 95% CIs in one plot.
    
    Parameters:
    condition_data (list): List of three lists, each containing RR profiles for one condition
    lags (array): Time lags in seconds (x-axis values)
    colors (list): Colors for each condition
    condition_names (list): Names for each condition
    title, xlabel, ylabel (str): Plot labels
    """
    plt.figure(figsize=(12, 6))
    
    for i, (rr_profiles, color, name) in enumerate(zip(condition_data, colors, condition_names)):
        # Convert to numpy array (n_dyads × n_lags)
        rr_matrix = np.array(rr_profiles)
        
        # Calculate statistics
        mean_rr = np.mean(rr_matrix, axis=0)
        sem = np.std(rr_matrix, axis=0, ddof=1) / np.sqrt(len(rr_profiles))
        ci_width = t.ppf(0.975, len(rr_profiles)-1) * sem
        
        # Plot confidence interval (lighter shade)
        plt.fill_between(lags, 
                        mean_rr - ci_width,
                        mean_rr + ci_width,
                        color=color, alpha=0.15,
                        label='_nolegend_')
        
        # Plot mean line
        plt.plot(lags, mean_rr, color=color, 
                 linewidth=2.5, 
                 alpha=0.9,
                 marker='o' if i==0 else 's' if i==1 else '^',
                 markersize=5,
                 markevery=5,
                 label=name)
        
        # Mark peak lag
        peak_lag = lags[np.argmax(mean_rr)]
        plt.scatter(peak_lag, np.max(mean_rr), 
                    color=color, marker='*', s=120,
                    edgecolor='k', zorder=10)
    
    # Add reference lines and formatting
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.title(title, pad=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.legend(framealpha=1, loc='upper right')
    
    # Set symmetrical x-axis if centered on 0
    if np.min(lags) < 0 and np.max(lags) > 0:
        xlim = max(abs(np.min(lags)), abs(np.max(lags)))
        plt.xlim(-xlim, xlim)
    
    plt.tight_layout()
    plt.show()



def normalize_conditions(condition_data, target_mean=0.05):
    """Normalize all conditions to have the same mean RR while preserving profile shapes"""
    # Calculate current means
    condition_means = [np.mean(np.concatenate(cond)) for cond in condition_data]
    global_mean = np.mean(condition_means)
    
    # Compute scaling factors
    scaling_factors = [target_mean / cm for cm in condition_means]
    
    # Apply scaling
    normalized_data = []
    for cond, factor in zip(condition_data, scaling_factors):
        normalized_data.append([rr_profile * factor for rr_profile in cond])
    
    return normalized_data

def center_conditions(condition_data):
    """Center each condition around 0 (show change from mean)"""
    centered_data = []
    for cond in condition_data:
        cond_array = np.concatenate(cond)
        condition_mean = np.mean(cond_array)
        centered_data.append([profile - condition_mean for profile in cond])
    return centered_data

def plot_multicondition_rr_profiles(condition_data, lags, 
                                  condition_names=['Intro (no zoom)', 'Discussion (no zoom)', 'Reschu (no zoom)',
                                                 'Intro (zoom)', 'Discussion (zoom)', 'Reschu (zoom)'],
                                  title='Recurrence Rate Profiles by Condition',
                                  xlabel='Lag (seconds)',
                                  ylabel='Recurrence Rate (RR)'):
    """
    Plot RR profiles for zoom/no-zoom conditions with consistent styling.
    Conditions 1-3: no zoom
    Conditions 4-6: zoom
    """
    # Color scheme: same hue for matching conditions, different saturation for zoom
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c',  # no zoom (vibrant)
        '#8ba8ca', '#ffbc79', '#96d196'   # zoom (paler versions)
    ]
    
    # Line styles: solid for no zoom, dashed for zoom
    line_styles = ['-', '-', '-', '--', '--', '--']
    
    # Markers: consistent within task types
    markers = ['o', 's', '^', 'o', 's', '^']
    
    plt.figure(figsize=(14, 7))
    
    for i, (rr_profiles, color, style, marker, name) in enumerate(
            zip(condition_data, colors, line_styles, markers, condition_names)):
        
        rr_matrix = np.array(rr_profiles)
        mean_rr = np.mean(rr_matrix, axis=0)
        std_dev = np.std(rr_matrix, axis=0, ddof=1)
        
        # Plotting order: zoom first (background) then no zoom (foreground)
        zorder = 5 if i < 3 else 10
        
        # STD band
        plt.fill_between(lags, 
                        mean_rr - std_dev,
                        mean_rr + std_dev,
                        color=color, alpha=0.15,
                        zorder=zorder,
                        label='_nolegend_')
        
        # Mean line
        plt.plot(lags, mean_rr, 
                 color=color,
                 linestyle=style,
                 linewidth=2.5,
                 marker=marker,
                 markersize=6,
                 markevery=7,
                 zorder=zorder,
                 label=name)
    
    # Reference lines and formatting
    plt.axvline(0, color='k', linestyle=':', alpha=0.3)
    plt.title(title, pad=20, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(alpha=0.15)
    
    # Create custom legend grouping
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linestyle='-', lw=3, label='No Zoom'),
        Line2D([0], [0], color='k', linestyle='--', lw=3, label='Zoom'),
        Line2D([0], [0], color='#1f77b4', marker='o', lw=0, label='Intro', markersize=8),
        Line2D([0], [0], color='#ff7f0e', marker='s', lw=0, label='Discussion', markersize=8),
        Line2D([0], [0], color='#2ca02c', marker='^', lw=0, label='Reschu', markersize=8)
    ]
    
    plt.legend(handles=legend_elements, framealpha=1, 
               loc='upper right', fontsize=10, ncol=2)
    
    # Symmetrical x-axis
    xlim = max(abs(np.min(lags)), abs(np.max(lags)))
    plt.xlim(-xlim, xlim)
    
    plt.tight_layout()
    plt.show()