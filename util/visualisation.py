import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
import os


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

from statsmodels.stats.multitest import multipletests
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm, ols

def unified_mixed_model_analysis_fdr(df, response_variable, save_fig=False, output_path="img/"):
    # 1. Filter for RESCHU run phases
    reschu_df = df[df['phase'].str.contains('reschu_run')].copy()
    reschu_df = reschu_df.dropna(subset=[response_variable])
    reschu_df['run_number'] = reschu_df['phase'].str.extract(r'(\d+)$')[0]

    # 2. Validate necessary columns
    if 'pair' not in reschu_df.columns:
        raise ValueError("Dataframe must contain a 'pair' column for grouping.")

 
    if response_variable in ['Cooperation', 'Cohesion', 'Empathy']:
        print(f"[INFO] '{response_variable}' detected as dyad-level (flat). Using OLS.")
        reschu_df = reschu_df.groupby(['pair', 'zoom']).agg(
            {**{f: 'mean' for f in FACTORS}, response_variable: 'first'}
        ).reset_index()

    # 4. Run models and collect results
    results = []
    for factor in FACTORS:
        formula = f"{response_variable} ~ {factor} * zoom"
        try:
            if response_variable in ['Cooperation', 'Cohesion', 'Empathy']:
                model = ols(formula, data=reschu_df).fit()
            else:
                model = mixedlm(formula, data=reschu_df, groups=reschu_df['pair']).fit()
            results.append({
                'factor': factor,
                'global_slope': model.params.get(factor, np.nan),
                'global_p': model.pvalues.get(factor, np.nan),
                'zoom_slope': model.params.get(f"{factor}:zoom[T.True]", np.nan),
                'zoom_p': model.pvalues.get(f"{factor}:zoom[T.True]", np.nan)
            })
        except Exception as e:
            print(f"Error for {factor}: {e}")
            results.append({
                'factor': factor,
                'global_slope': np.nan,
                'global_p': np.nan,
                'zoom_slope': np.nan,
                'zoom_p': np.nan
            })

    results_df = pd.DataFrame(results)

    # 5. Apply FDR correction separately to global and zoom p-values
    for p_col in ['global_p', 'zoom_p']:
        pvals = results_df[p_col].dropna().values
        rejected, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        results_df[f'{p_col}_FDR'] = np.nan
        results_df[f'{p_col}_sig_FDR'] = False
        results_df.loc[results_df[p_col].notna(), f'{p_col}_FDR'] = pvals_fdr
        results_df.loc[results_df[p_col].notna(), f'{p_col}_sig_FDR'] = rejected
        print("Raw p-values:", pvals)
        print("FDR-adjusted p-values:", pvals_fdr)
    # 6. Plot
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
            if len(subset) > 1:
                sns.regplot(
                    x=subset[factor],
                    y=subset[response_variable],
                    ax=ax,
                    scatter=False,
                    color=colors[str(zoom)],
                    ci=95,
                    line_kws={'lw': 2, 'ls': '-' if zoom else '--'}
                )

        # Annotation: MEM/OLS β + p + FDR p
        row = results_df[results_df['factor'] == factor].iloc[0]
        global_slope = row['global_slope']
        global_p = row['global_p']
        global_p_fdr = row['global_p_FDR']
        zoom_slope = row['zoom_slope']
        zoom_p = row['zoom_p']
        zoom_p_fdr = row['zoom_p_FDR']

        weight = 'bold' if row['global_p_sig_FDR'] else 'regular'

        text = (
            f"Global β={global_slope:.2f}\n"
            f"p={global_p:.3f}, FDR={global_p_fdr:.3f}\n"
            f"Zoom Δβ={zoom_slope:.2f}\n"
            f"p={zoom_p:.3f}, FDR={zoom_p_fdr:.3f}"
        )

        ax.text(0.05, 0.95, text,
                transform=ax.transAxes,
                va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8),
                weight=weight)

        ax.set_title(f"{factor}: {FACTOR_LABELS.get(factor, factor)}", fontsize=16)
        ax.set_xlabel("Recurrence Rate (RR)")
        ax.set_ylabel(response_variable)

    # Shared legend
    handles = [
        plt.Line2D([], [], color=colors['True'], marker='o', ls='-', label='Zoom=True'),
        plt.Line2D([], [], color=colors['False'], marker='s', ls='--', label='Zoom=False')
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    plt.suptitle(f"Relation between {response_variable} and RR × Zoom (FDR-corrected)", fontsize=24, y=1.02)

    if save_fig:
        fig.savefig(f"{output_path}{response_variable}_zoom_FACE_FDR.png", dpi=300, bbox_inches='tight')

    plt.show()

    return results_df


from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.formula.api import mixedlm, ols

from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm, ols

def unified_rr_tp_analysis_fdr(df, response_variable, save_fig=False, output_path="img/"):
    # Filter RESCHU runs
    reschu_df = df[df['phase'].str.contains('reschu_run')].copy()
    reschu_df['run_number'] = reschu_df['phase'].str.extract(r'(\d+)$')[0]
    reschu_df = reschu_df.dropna(subset=[response_variable])

    if 'pair' not in reschu_df.columns:
        raise ValueError("Dataframe must contain a 'pair' column for grouping.")

    if response_variable in ['Cohesion', 'Empathy', 'Cooperation']:
        print(f"[INFO] '{response_variable}' detected as dyad-level (flat). Using OLS.")
        reschu_df = reschu_df.groupby('pair').agg(
            {**{f: 'mean' for f in FACTORS}, response_variable: 'first'}
        ).reset_index()

    # Scale predictors and response
    for col in FACTORS + [response_variable]:
        reschu_df[f'{col}_scaled'] = (reschu_df[col] - reschu_df[col].mean()) / reschu_df[col].std()

    # Run models
    results = []
    for factor in FACTORS:
        factor_scaled = f'{factor}_scaled'
        formula = f"{response_variable}_scaled ~ {factor_scaled}"
        try:
            if response_variable in ['Cohesion', 'Empathy', 'Cooperation']:
                model = ols(formula, data=reschu_df).fit()
            else:
                model = mixedlm(formula, data=reschu_df, groups=reschu_df['pair']).fit()
            coef = model.params[factor_scaled]
            pval = model.pvalues[factor_scaled]
            results.append({'factor': factor, 'coef': coef, 'pval': pval})
            print(f"{factor}: coef = {coef:.4f}, p = {pval:.4g}")
        except Exception as e:
            print(f"Error fitting model for {factor}: {e}")
            results.append({'factor': factor, 'coef': np.nan, 'pval': np.nan})

    results_df = pd.DataFrame(results)

    # Apply FDR correction
    pvals = results_df['pval'].dropna().values
    rejected, pvals_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')

    results_df['pval_FDR'] = np.nan
    results_df['significant_FDR'] = False
    results_df.loc[results_df['pval'].notna(), 'pval_FDR'] = pvals_fdr
    results_df.loc[results_df['pval'].notna(), 'significant_FDR'] = rejected

    print("Raw p-values:", pvals)
    print("FDR-adjusted p-values:", pvals_fdr)


    # Plot
    plt.figure(figsize=(20, 12))
    for i, factor in enumerate(FACTORS, 1):
        plt.subplot(2, 3, i)
        sns.scatterplot(
            data=reschu_df,
            x=factor,
            y=response_variable,
            hue='pair' if not response_variable in ['Cohesion', 'Empathy', 'Cooperation'] else None,
            palette='husl' if not response_variable in ['Cohesion', 'Empathy', 'Cooperation'] else None,
            s=80,
            edgecolor='w',
            linewidth=0.5,
            legend=False
        )
        if len(reschu_df) > 1:
            sns.regplot(
                data=reschu_df,
                x=factor,
                y=response_variable,
                scatter=False,
                ci=95,
                line_kws={'color': 'black', 'lw': 2, 'ls': '--'}
            )

        # Annotation based on MEM/OLS + FDR
        row = results_df[results_df['factor'] == factor].iloc[0]
        coef = row['coef']
        pval = row['pval']
        pval_fdr = row['pval_FDR']
        sig = row['significant_FDR']

        pval_text = f"p = {pval:.3f}"
        pval_fdr_text = f"FDR p = {pval_fdr:.3f}"
        weight = 'bold' if sig else 'regular'

        plt.text(
            0.05, 0.95,
            f"β = {coef:.2f}\n{pval_text}\n{pval_fdr_text}",
            transform=plt.gca().transAxes,
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8),
            weight=weight
        )

        plt.title(f"{response_variable.capitalize()} vs {factor}: {FACTOR_LABELS.get(factor, factor)}", fontsize=14)
        plt.xlabel('Mean activation')
        plt.ylabel(response_variable.capitalize())
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.suptitle(f"Relation between {response_variable} and RR (FDR-corrected MEM p-values)", fontsize=20, y=1.02)

    if save_fig:
        plt.savefig(f"{output_path}{response_variable}_RR_model_MEM_FDR.png", dpi=300, bbox_inches='tight')

    plt.show()

    return results_df
