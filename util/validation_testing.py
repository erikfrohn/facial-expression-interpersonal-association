import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import mannwhitneyu  # Non-parametric alternative
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
def statistical_factor_analysis(df, debug=False, nem = False):
    # Initialize results storage
    results = []

    # Loop through each factor
    for factor in df['component_factor'].unique():
        # Subset data for the current factor
        subset = df[df['component_factor'] == factor]
        
        # Option 1: Test real vs. fake (ignore non_event_matches)
        if not nem: 
            real = subset[subset['condition'] == 'real']['RR']
            fake = subset[subset['condition'] == 'fake']['RR']
            stat, p = mannwhitneyu(real, fake, alternative='greater')
            results.append({
                'factor': factor,
                'comparison': 'real_vs_fake',
                'statistic': stat,
                'p_value': p
            })
        
        # # # Option 2: Stratify by non_event_matches (uncomment if needed)
        if nem:
            for ne_match in ['included', 'excluded']:
                ne_subset = subset[subset['non_event_matches'] == ne_match]
                real = ne_subset[ne_subset['condition'] == 'real']['RR']
                fake = ne_subset[ne_subset['condition'] == 'fake']['RR']
                stat, p = mannwhitneyu(real, fake, alternative='greater')
                results.append({
                    'factor': factor,
                    'comparison': f'real_vs_fake_{ne_match}',
                    'statistic': stat,
                    'p_value': p
                })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Apply FDR correction to all p-values
    _, results_df['p_fdr'] = fdrcorrection(results_df['p_value'])

    # Print significant results (FDR-corrected p < 0.05)
    significant = results_df[results_df['p_fdr'] < 0.05]
    if debug:
        print("Significant comparisons (Mann-Whitney U, FDR-corrected):")
        print(results_df.sort_values('p_fdr'))
    return results_df