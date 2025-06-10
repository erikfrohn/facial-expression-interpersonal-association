import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import seaborn as sns
import pandas as pd
import pandas as pd
import numpy as np
from collections import defaultdict
import os

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.stats import spearmanr
import statsmodels.formula.api as smf

import matplotlib.gridspec as gridspec

FACTORS = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']


FACTOR_LABELS = {
    'f1': "Enjoyment Smile",
    'f2': "Eyebrows Up",
    'f3': "Mouth Open",
    'f4': "Mouth Tightening",
    'f5': "Eye Tightening",
    'f6': "Mouth Frown"
}




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
        

def plot_crp_with_signals(p1, p2, recurrence_matrix, title='', fig=None, outer_grid=None):

    p1 = p1.squeeze()
    p2 = p2.squeeze()

    if fig is None or outer_grid is None:
        fig = plt.figure(figsize=(8, 8))
        outer_grid = gridspec.GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1],
                                       wspace=0.05, hspace=0.05)
    else:
        # expected to be passed from subplot context
        outer_grid = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer_grid,
            width_ratios=[0.5, 4],  # shrink left signal
            height_ratios=[4, 0.5], # shrink bottom signal
            wspace=0.05, hspace=0.05
        )

    ax_empty = fig.add_subplot(outer_grid[0, 0])
    ax_empty.axis('off')

    ax_crp = fig.add_subplot(outer_grid[0, 1])
    ax_crp.imshow(recurrence_matrix, origin='lower', cmap='Greys', interpolation='nearest', aspect='equal')
    ax_crp.plot(range(min(len(p1), len(p2))), range(min(len(p1), len(p2))), 'r--', alpha=0.4)
    ax_crp.set_ylabel("")
    ax_crp.set_xticks([])
    ax_crp.set_title(title, fontsize=12)

    ax_bottom = fig.add_subplot(outer_grid[1, 1], sharex=ax_crp)
    ax_bottom.plot(p2, color='blue')
    ax_bottom.set_xlabel("Time (Person 2)")
    ax_bottom.set_yticks([])
    ax_bottom.set_position([
        ax_bottom.get_position().x0,
        ax_bottom.get_position().y0 + 0.1,  # shift upward
        ax_bottom.get_position().width,
        ax_bottom.get_position().height  # compress height
    ])
    ax_left = fig.add_subplot(outer_grid[0, 0], sharey=ax_crp)
    ax_left.plot(p1, range(len(p1)), color='green')
    ax_left.invert_xaxis()
    ax_left.set_xticks([])
    ax_left.set_ylabel("Time (Person 1)")
    ax_left.set_position([
        ax_left.get_position().x0,
        ax_left.get_position().y0 + 0.09,  # shift upward
        ax_left.get_position().width,
        ax_left.get_position().height * 0.73  # compress height
    ])

    fig.suptitle("Example Cross-Recurrence Plots", fontsize=14)




def calculate_non_event_match_ratio(location, feature_folder, pairs, phases, factors, threshold=0.08, debug=False):
    total_counts = defaultdict(int)
    nem_counts = defaultdict(int)

    for pair in pairs:
        p1, p2 = pair.split("_")
        files = os.path.join(location, pair, feature_folder)

        for phase in phases:
            nav_file = os.path.join(files, f'pp{p1}_{phase}_factors.csv')
            pil_file = os.path.join(files, f'pp{p2}_{phase}_factors.csv')

            if not (os.path.exists(nav_file) and os.path.exists(pil_file)):
                continue

            nav = pd.read_csv(nav_file)
            pil = pd.read_csv(pil_file)

            for f in factors:
                if f not in nav.columns or f not in pil.columns:
                    continue

                nav_vals = nav[f].dropna().values
                pil_vals = pil[f].dropna().values
                min_len = min(len(nav_vals), len(pil_vals))

                nav_vals = nav_vals[:min_len]
                pil_vals = pil_vals[:min_len]

                total_counts[f] += min_len * 2
                nem_counts[f] += np.sum(np.abs(nav_vals) <= threshold)
                nem_counts[f] += np.sum(np.abs(pil_vals) <= threshold)

    # Compute ratios as percentages
    ratios = {}
    for f in factors:
        print(f, nem_counts[f], total_counts[f])
        if total_counts[f] > 0:
            ratios[f] = 100 * nem_counts[f] / total_counts[f]
        else:
            ratios[f] = None

    if debug:
        # Prepare DataFrame for plotting
        ratio_df = pd.DataFrame({
            'factor': [f"{f} - {FACTOR_LABELS[f]}" for f in factors if ratios[f] is not None],
            'percentage': [ratios[f] for f in factors if ratios[f] is not None]
        })


        # === 3. Visualization ===
        plt.figure(figsize=(14, 6))
        sns.barplot(data=ratio_df, x='factor', y='percentage', ci=None)
        plt.ylabel('Percentage of Non-Event Matches')
        plt.xlabel('Facial Factor')
        plt.title('Proportion of Neutral Facial Expressions per Factor')
        for i, row in ratio_df.iterrows():
            plt.text(i, row['percentage'] + 1, f"{row['percentage']:.1f}%", ha='center', fontsize=11)
        plt.ylim(0, max(ratio_df['percentage']) + 10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("NEM count per factor.png", dpi=300, bbox_inches='tight')

    return ratios




# Define a function that performs the unified analysis
def unified_mixed_model_analysis(df, response_variable, save_fig=False, output_path="img/"):

    # 1. Filter for RESCHU run phases
    reschu_df = df[df['phase'].str.contains('reschu_run')].copy()
    reschu_df = reschu_df.dropna(subset=[response_variable])
    reschu_df['run_number'] = reschu_df['phase'].str.extract(r'(\d+)$')[0]

    # 2. Validate necessary columns
    if 'pair' not in reschu_df.columns:
        raise ValueError("Dataframe must contain a 'pair' column for grouping.")

    pairs = reschu_df['pair'].unique()
    palette = sns.color_palette("husl", len(pairs))
    pair_colors = dict(zip(pairs, palette))

    # 3. Run mixed models and collect results
    results = []
    for factor in FACTORS:
        formula = f"{response_variable} ~ {factor} * zoom"
        try:
            model = smf.mixedlm(formula, data=reschu_df, groups=reschu_df['pair'])
            fit = model.fit()
            results.append({
                'factor': factor,
                'global_slope': fit.params[factor],
                'global_p': fit.pvalues[factor],
                'zoom_slope': fit.params.get(f"{factor}:zoom[T.True]", np.nan),
                'zoom_p': fit.pvalues.get(f"{factor}:zoom[T.True]", np.nan)
            })
        except Exception as e:
            print(f"Error for {factor}: {e}")
            results.append({'factor': factor, 'global_slope': np.nan, 'global_p': np.nan, 'zoom_slope': np.nan, 'zoom_p': np.nan})

    results_df = pd.DataFrame(results)

    # 4. Plot results
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    axs = axs.flatten()

    colors = {'True': '#1f77b4', 'False': '#ff7f0e'}
    markers = {'True': 'o', 'False': 's'}

    for i, factor in enumerate(FACTORS):
        ax = axs[i]
        for zoom in [True, False]:
            subset = reschu_df[reschu_df['zoom'] == zoom]
            ax.scatter(
                subset[factor],
                subset[response_variable],
                c=colors[str(zoom)],
                marker=markers[str(zoom)],
                alpha=0.6,
                edgecolor='w',
                linewidth=0.5,
                label=f"Zoom={zoom}"
            )
            sns.regplot(
                x=subset[factor],
                y=subset[response_variable],
                ax=ax,
                scatter=False,
                color=colors[str(zoom)],
                ci=95,
                line_kws={'lw': 2, 'ls': '-' if zoom else '--'}
            )

        # Add model annotations
        row = results_df[results_df['factor'] == factor].iloc[0]
        if row['global_p'] >= 0.05:
            annotation = (f"Global β: {row['global_slope']:.2f} (p={row['global_p']:.3f})\n"
                        f"Zoom Δβ: {row['zoom_slope']:.2f} (p={row['zoom_p']:.3f})")
        else:
            annotation = (f"Global β: {row['global_slope']:.2f} (p={row['global_p']:.3f}*)\n"
                        f"Zoom Δβ: {row['zoom_slope']:.2f} (p={row['zoom_p']:.3f})")

        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                va='top', ha='left', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

        ax.set_title(f"{factor}: {FACTOR_LABELS.get(factor, factor)}")
        ax.set_xlabel("Recurrence Rate (RR)")
        ax.set_ylabel("Score" if i in [0, 3] else "")

    # Shared legend
    handles = [
        plt.Line2D([], [], color=colors['True'], marker='o', ls='-', label='Zoom=True'),
        plt.Line2D([], [], color=colors['False'], marker='s', ls='--', label='Zoom=False')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.suptitle("Mixed Effects Modeling of the Relation between Score and RR, moderated by Zoom", fontsize=16, y=1.02)

    if save_fig:
        fig.savefig(f"{output_path}{response_variable}_zoom_FACE.png", dpi=300, bbox_inches='tight')

    plt.show()
    return results_df
