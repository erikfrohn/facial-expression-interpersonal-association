import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import mannwhitneyu 
from scipy.stats import spearmanr  

COMPONENTS = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

def statistical_factor_analysis_lme(df, debug=False, nem=False):
    results = []

    for factor in df['factor'].unique():
        subset = df[df['factor'] == factor]

        # Optional: subset by non_event_matches
        if nem:
            for ne_match in ['included', 'excluded']:
                ne_subset = subset[subset['non_event_matches'] == ne_match]
                if ne_subset['condition'].nunique() < 2:
                    if debug:
                        print(f"Skipping factor={factor}, ne_match={ne_match} due to lack of condition variation.")
                    continue

                model = smf.mixedlm("RR ~ condition", ne_subset, groups=ne_subset["pair"])
                result = model.fit(reml=False)  # Use ML not REML for fixed effects comparison

                condition_coef = result.params.get('condition[T.real]', float('nan'))
                p_value = result.pvalues.get('condition[T.real]', float('nan'))

                results.append({
                    'factor': factor,
                    'comparison': f'real_vs_fake_{ne_match}',
                    'coef': condition_coef,
                    'p_value': p_value
                })
        else:
            if subset['condition'].nunique() < 2:
                if debug:
                    print(f"Skipping factor={factor} due to lack of condition variation.")
                continue

            model = smf.mixedlm("RR ~ condition", subset, groups=subset["pair"])
            result = model.fit(reml=False)

            condition_coef = result.params.get('condition[T.real]', float('nan'))
            p_value = result.pvalues.get('condition[T.real]', float('nan'))

            results.append({
                'factor': factor,
                'comparison': 'real_vs_fake',
                'coef': condition_coef,
                'p_value': p_value
            })

    results_df = pd.DataFrame(results)

    # FDR correction
    if not results_df.empty:
        _, results_df['p_fdr'] = fdrcorrection(results_df['p_value'])
    else:
        results_df['p_fdr'] = []

    # Optional debug print
    if debug:
        print("\nMixed Effects Model Results:")
        print(results_df.sort_values('p_fdr'))

    return results_df

def statistical_factor_analysis_aggregated(df, debug=False):
    """
    Performs Mann-Whitney U test on pair-aggregated RR values.
    Aggregates over all phases per pair per factor.
    """
    results = []

    # Aggregate RR per pair per factor per condition
    grouped = (
        df.groupby(['pair', 'factor', 'condition'])
        .agg({'RR': 'mean'})
        .reset_index()
    )

    for factor in grouped['factor'].unique():
        subset = grouped[grouped['factor'] == factor]

        # Split real and fake conditions
        real = subset[subset['condition'] == 'real']['RR']
        fake = subset[subset['condition'] == 'fake']['RR']

        # Make sure the lengths match across conditions
        stat, p = mannwhitneyu(real, fake, alternative='greater')
        results.append({
            'factor': factor,
            'comparison': 'real_vs_fake',
            'statistic': stat,
            'p_value': p,
            'n_real': len(real),
            'n_fake': len(fake)
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Apply FDR correction
    _, results_df['p_fdr'] = fdrcorrection(results_df['p_value'])

    if debug:
        print("Significant comparisons (Mann-Whitney U, FDR-corrected):")
        print(results_df.sort_values('p_fdr'))

    return results_df

def statistical_factor_analysis(df, debug=False, nem = False):
    # Initialize results storage
    results = []

    # Loop through each factor
    for factor in df['factor'].unique():
        # Subset data for the current factor
        subset = df[df['factor'] == factor]
        
        # Option 1: Test real vs. fake (ignore non_event_matches)
        if not nem: 
            real = subset[subset['condition'] == 'real']['RR']
            fake = subset[subset['condition'] == 'fake']['RR']
            stat, p = mannwhitneyu(real, fake, alternative='two-sided')
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
                stat, p = mannwhitneyu(real, fake, alternative='two-sided')
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

def run_significance_tests(df, dependent_var, components=COMPONENTS, debug=False):
    """
    Run significance tests for a dependent variable against components in different groupings
    
    Args:
        df: DataFrame containing the data
        dependent_var: Name of the dependent variable column
        components: List of component columns to test against
        debug: Whether to print debug information
        
    Returns:
        Tuple of (averaged_results, per_phase_results)
    """
    # Process averaged data - only drop rows where dependent_var is NaN
    df_avg = (df.groupby('pair', as_index=False)
              .agg({
                  'zoom': 'first',
                  dependent_var: 'mean',
                  **{c: 'mean' for c in components}
              })
              .rename(columns={c: f'avg_{c}' for c in components})
              .dropna(subset=[dependent_var]))  # Only drop if dependent_var is NaN
    
    avg_components = [f'avg_{c}' for c in components]
    averaged_results = _test_groupings(df_avg, avg_components, dependent_var, debug)
    
    # Process per-phase data - only drop rows where dependent_var is NaN
    df_phases = pd.concat([df[df['phase'] == f'reschu_run_{i}'] for i in range(8)])
    df_phases = df_phases.dropna(subset=[dependent_var])  # Only drop if dependent_var is NaN
    per_phase_results = significance(df_phases, debug=debug, column=dependent_var)
    
    return averaged_results, per_phase_results

def _test_groupings(df, components, dependent_var, debug=False):
    """
    Helper function to test different groupings in a dataframe
    
    Args:
        df: DataFrame to test
        components: List of component columns to test
        dependent_var: Name of dependent variable column
        debug: Whether to print debug information
        
    Returns:
        DataFrame with significant results
    """
    significant = pd.DataFrame(columns=['setting', 'component', 'rho', 'p'])
    
    # Test full dataset
    if debug: print(f"No partition:")
    for comp in components:
        rho, p = spearmanr(df[comp], df[dependent_var])
        clean_comp = comp.replace('avg_', '')
        if p < 0.05:
            significant.loc[len(significant)] = ['all_phases', clean_comp, rho, p]
        if debug: print(f"{clean_comp}: ρ = {rho:.3f}, p = {p:.4f}")

    # Test zoom conditions
    for condition in [True, False]:
        subset = df[df['zoom'] == condition]
        if debug: print(f"\nZoom = {condition}:")
        for comp in components:
            rho, p = spearmanr(subset[comp], subset[dependent_var])
            clean_comp = comp.replace('avg_', '')
            if p < 0.05:
                significant.loc[len(significant)] = [f'zoom = {condition}', clean_comp, rho, p]
            if debug: print(f"{clean_comp}: ρ = {rho:.3f}, p = {p:.4f}")

    if debug: 
        print("\n")
        print(significant)
    return significant

def significance(df, debug=False, column='score'):
    """
    Original significance function for per-phase testing
    """
    significant = pd.DataFrame(columns=['setting', 'component', 'rho', 'p'])
    if debug:  print(f"No partition:")
    for comp in COMPONENTS:
        rho, p = spearmanr(df[comp], df[column])
        if p < 0.05:
            significant.loc[len(significant)] = ['all_reschu', comp, rho, p]
        if debug: print(f"{comp}: ρ = {rho:.3f}, p = {p:.4f}")

    for condition in [True, False]:
        subset = df[df['zoom'] == condition]
        if debug: print(f"\nZoom = {condition}:")
        for comp in COMPONENTS:
            rho, p = spearmanr(subset[comp], subset[column])
            if p < 0.05:
                significant.loc[len(significant)] = [f'zoom = {condition}', comp, rho, p]
            if debug: print(f"{comp}: ρ = {rho:.3f}, p = {p:.4f}")

    for i in range(8):
        subset = df[df['phase'] == f'reschu_run_{i}']
        if debug: print(f"\nPhase = reschu_run_{i}:")
        for comp in COMPONENTS:
            rho, p = spearmanr(subset[comp], subset[column])
            if p < 0.05:
                significant.loc[len(significant)] = [f'reschu_run_{i}', comp, rho, p]
            if debug: print(f"{comp}: ρ = {rho:.3f}, p = {p:.4f}")
    if debug: 
        print("\n")
        print(significant)
    return significant
def process_dataframe(df, version_name, debug=False):
    """Helper function to process and validate dataframe."""
    df_validation = statistical_factor_analysis_aggregated(df, debug=debug)
    df_significant = df_validation[df_validation['p_fdr'] < 0.05]
    combinations = df_significant[['factor', 'comparison', 'p_value']].drop_duplicates()
    combinations['version'] = version_name
    if debug:
        print(f"Remaining factors for {version_name}: {combinations['factor'].unique()}\n")
    return df_validation