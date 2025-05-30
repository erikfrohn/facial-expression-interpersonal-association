import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import seaborn as sns
import pandas as pd

FACTORS = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
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



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from scipy.stats import spearmanr

def plot_mixed_effects_model(df):
    # Filter only RESCHU runs

    reschu_df = df[df['phase'].str.contains('reschu_run')].copy()

    # Create numerical run identifier
    reschu_df['run_number'] = reschu_df['phase'].str.extract('(\d+)').astype(int)

    # Get unique pairs and assign colors
    pairs = reschu_df['pair'].unique()
    palette = sns.color_palette("husl", len(pairs))
    pair_colors = dict(zip(pairs, palette))

    # Create plots for each factor
    for factor in ['f1', 'f2', 'f4', 'f5', 'f6']:
        # Calculate global Spearman correlation
        rho, p = spearmanr(reschu_df[factor], reschu_df['score'])
        
        # Format p-value annotation
        if p < 0.001:
            p_text = f'p < 0.001'
        else:
            p_text = f'p = {p:.3f}'
        
        # Calculate global linear trend
        global_slope, global_intercept, _, _, _ = linregress(reschu_df[factor], reschu_df['score'])
        
        plt.figure(figsize=(10,6))
        
        # Plot individual pairs with common slope
        for pair in pairs:
            pair_data = reschu_df[reschu_df['pair'] == pair]
            pair_intercept = np.mean(pair_data['score'] - global_slope * pair_data[factor])
            x_vals = np.array([pair_data[factor].min(), pair_data[factor].max()])
            y_vals = pair_intercept + global_slope * x_vals
            plt.plot(x_vals, y_vals, color=pair_colors[pair], alpha=0.5, lw=1)
            # Plot points
            sns.scatterplot(
                data=pair_data,
                x=factor,
                y='score',
                hue='run_number',
                palette='viridis',
                s=100,
                edgecolor='w',
                linewidth=0.5,
                legend=False,
                ax=plt.gca()
            )

        # Add bold global trend line
        x_global = np.linspace(reschu_df[factor].min(), reschu_df[factor].max(), 100)
        y_global = global_intercept + global_slope * x_global
    
        # Add global trend line
        plt.plot(x_global, y_global, color='black', lw=3, linestyle='--', 
                label=f'Global Slope (β={global_slope:.2f})')
        
        # Add correlation annotation
        plt.text(0.05, 0.95, 
                f"Spearman's ρ = {rho:.2f}\n{p_text}", 
                transform=plt.gca().transAxes,
                va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8))

        # Add intercept difference legend
        handles = [plt.Line2D([0], [0], color=c, lw=2) for c in pair_colors.values()]
        # plt.legend(handles, pairs, title='Pairs (Lines Show Baseline Differences)', 
        #         bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title(f'Score vs {factor}\nCommon Slope with Pair Baselines | {p_text}')
        plt.xlabel(f'{factor} Value')
        plt.ylabel('Team Score')
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

def delay_profile(df):
    for f in FACTORS:
    # Load and prepare data
        df_fac = df[df['factor'] == f]
    # Create condition DataFrames
        df_intro = df_fac[df_fac['phase'] == 'instructional_video_0']
        df_discussion = pd.concat([
            df_fac[df_fac['phase'] == 'discussion_phase_0'], 
            df_fac[df_fac['phase'] == 'discussion_phase_1']
        ])
        df_reschu = pd.concat([
            df_fac[df_fac['phase'] == f'reschu_run_{i}'] 
            for i in range(8)
            ])
        conditions = []
        for zoom in [True, False]:
            df_intro_f = df_intro[df_intro['zoom'] == zoom]
            df_discussion_f = df_discussion[df_discussion['zoom'] == zoom]
            df_reschu_f = df_reschu[df_reschu['zoom'] == zoom]
        # Drop unnecessary columns - make sure these columns exist
            cols_to_drop = ['pair', 'zoom', 'phase', 'beeps', 'score', 'factor']
            df_intro_f = df_intro_f.drop(columns=[col for col in cols_to_drop if col in df_intro_f.columns])
            df_discussion_f = df_discussion_f.drop(columns=[col for col in cols_to_drop if col in df_discussion_f.columns])
            df_reschu_f = df_reschu_f.drop(columns=[col for col in cols_to_drop if col in df_reschu_f.columns])

        # Create lag values (-6 to +6 seconds at 10Hz)
            n_lags = 121  # 6*10*2 + 1 (including 0)
            lags = np.linspace(-6, 6, n_lags)

        # Verify data shape matches expected lags
            print(f"Data columns: {df_intro_f.shape[1]}, Expected: {n_lags}")
            assert df_intro_f.shape[1] == n_lags, "Number of columns doesn't match expected lag points!"

        # Convert to list of RR profiles (each row becomes one profile)
            cond1 = [row for row in df_intro_f.values]
            cond2 = [row for row in df_discussion_f.values]
            cond3 = [row for row in df_reschu_f.values]

        # Verify normalization
            for i, cond in enumerate([cond1,cond2,cond3]):
                print(f"Condition {i+1} mean: {np.mean(np.concatenate(cond)):.3f}")
            conditions.append(cond1)
            conditions.append(cond2)
            conditions.append(cond3)

        print(len(conditions))
    # Plot with normalized data
        plot_multicondition_rr_profiles(
            condition_data=conditions,
            lags=lags,
            title=f'RR profile for {f}'
    )
        


def plot_crp_with_signals(p1, p2, recurrence_matrix, title='Cross-Recurrence Plot with Time Series', figsize=(8, 8)):
    """
    Plots the CRP with the original time series (p1 and p2) shown along the axes.

    Parameters:
        p1 (1D array): Time series of person 1.
        p2 (1D array): Time series of person 2.
        recurrence_matrix (2D bool array): CRP from crqa_lag_analysis.
        title (str): Plot title.
        figsize (tuple): Size of the entire plot.
    """
    import matplotlib.gridspec as gridspec

    # Ensure 1D input for plotting
    p1 = p1.squeeze()
    p2 = p2.squeeze()

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1],
                           wspace=0.05, hspace=0.05)

    # Top-left: empty (can be used for legends or stats if desired)
    ax_empty = fig.add_subplot(gs[0, 0])
    ax_empty.axis('off')

    # Top-right: CRP
    ax_crp = fig.add_subplot(gs[0, 1])
    ax_crp.imshow(recurrence_matrix, origin='lower', cmap='Greys', interpolation='nearest', aspect='equal')
    ax_crp.plot(range(min(len(p1), len(p2))), range(min(len(p1), len(p2))), 'r--', alpha=0.4)
    ax_crp.set_ylabel("Time (Person 1)")
    ax_crp.set_xticks([])

    # Bottom-right: Person 2 signal
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_crp)
    ax_bottom.plot(p2, color='blue')
    ax_bottom.set_xlabel("Time (Person 2)")
    ax_bottom.set_yticks([])

    # Left (rotated): Person 1 signal
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_crp)
    ax_left.plot(p1, range(len(p1)), color='green')  # Horizontal axis is value
    ax_left.invert_xaxis()
    ax_left.set_xticks([])
    ax_left.set_ylabel("")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
