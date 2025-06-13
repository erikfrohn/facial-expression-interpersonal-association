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
    ax_crp.set_title(title, fontsize=16)

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

def calculate_non_event_match_ratio(location, feature_folder, pairs, phases, factors, threshold=0.08, debug=False, savefig=False):
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
        if savefig: plt.savefig("NEM count per factor.png", dpi=300, bbox_inches='tight')

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
            weight = 'regular'
        else:
            annotation = (f"Global β: {row['global_slope']:.2f} (p={row['global_p']:.3f}*)\n"
                        f"Zoom Δβ: {row['zoom_slope']:.2f} (p={row['zoom_p']:.3f})")
            weight = 'bold'

        ax.text(0.05, 0.95, annotation, transform=ax.transAxes,
                va='top', ha='left', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8), weight= weight)

        ax.set_title(f"{factor}: {FACTOR_LABELS.get(factor, factor)}", fontsize=16)
        ax.set_xlabel("Recurrence Rate (RR)")
        ax.set_ylabel("Score" if i in [0, 3] else "")

    # Shared legend
    handles = [
        plt.Line2D([], [], color=colors['True'], marker='o', ls='-', label='Zoom=True'),
        plt.Line2D([], [], color=colors['False'], marker='s', ls='--', label='Zoom=False')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.suptitle("Mixed Effects Modeling of the Relation between Score and RR, moderated by Zoom", fontsize=24, y=1.02)

    if save_fig:
        fig.savefig(f"{output_path}{response_variable}_zoom_FACE.png", dpi='figure', bbox_inches='tight')

    plt.show()
    return results_df
