import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_rel, mannwhitneyu, shapiro
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def load_results(path_to_new_data, path_to_result_folder):
    phases = ["instructional_video_0", "discussion_phase_0", "discussion_phase_1"]
    
    all_combined_df = pd.DataFrame()
   
    # Initialize all results dictionary - works
    #all_results = pd.DataFrame()


    # Initialize all results dictionary
    all_results = {
        'Phase': [],
        'Factor': [],
        'Column': [],
        'Test': [],
        'Statistic': [],
        'P-value': [],
        'Participants (Fake)': [],
        'Participants (Real)': []
    }

    for phase in phases:
        fake_dataframes = []
        real_dataframes = []
        fake_files_count = 0
        real_files_count = 0

        for file_name in os.listdir(path_to_result_folder):
            if phase in file_name and file_name.endswith('.csv') and 'pp' in file_name:
                file_path = os.path.join(path_to_result_folder, file_name)
                df = pd.read_csv(file_path)
                if 'fake' in file_name:
                    #print(file_name)
                    fake_dataframes.append(df)
                    fake_files_count+=1
                    #print(f'fake count: {fake_files_count}')
                else:
                    #print(file_name)
                    real_dataframes.append(df)
                    real_files_count+=1
                    #print(f'real count: {real_files_count}')

        min_count = min(fake_files_count, real_files_count)

        # Select equal number of fake and real dataframes
        fake_dataframe = pd.concat(fake_dataframes[:min_count], ignore_index=True)
        real_dataframe = pd.concat(real_dataframes[:min_count], ignore_index=True)

        # Create box plots for all factors and the different Pearson_Correlation / DTW distance
        factors = fake_dataframe['Factor'].unique()
        columns = ['Pearson_Correlation', 'DTW distance']

        # Combine fake and real dataframes for plotting
        fake_dataframe['Type'] = 'Fake'
        real_dataframe['Type'] = 'Real'
        combined_df = pd.concat([fake_dataframe, real_dataframe], ignore_index=True)
        # Add the phase column to combined_df
        combined_df['Phase'] = f'{phase}'
        # Append the combined_df to all_combined_df
        all_combined_df = pd.concat([all_combined_df, combined_df], ignore_index=True)
    
        for i, factor in enumerate(factors):
            for j, column in enumerate(columns):
                # Perform statistical tests
                fake_values = fake_dataframe[fake_dataframe['Factor'] == factor][column]
                real_values = real_dataframe[real_dataframe['Factor'] == factor][column]

                # Check for normality using Shapiro-Wilk test
                _, p_real = shapiro(real_values)
                _, p_fake = shapiro(fake_values)

                if p_real > 0.05 and p_fake > 0.05:
                    # Perform paired t-test
                    stat, p_value = ttest_rel(real_values, fake_values)
                    test_name = 'Paired t-test'
                else:
                    # Perform Mann-Whitney U test
                    stat, p_value = mannwhitneyu(real_values, fake_values)
                    test_name = 'Mann-Whitney U test'

                 # Append results - all
                all_results['Phase'].append(phase)
                all_results['Factor'].append(factor)
                all_results['Column'].append(column)
                all_results['Test'].append(test_name)
                all_results['Statistic'].append(stat)
                all_results['P-value'].append(p_value)
                all_results['Participants (Fake)'].append(len(fake_values))
                all_results['Participants (Real)'].append(len(real_values))
                # # Append to all results
                # for key in all_results:
                #     all_results[key].extend(all_results[key])

    # Save the combined results to a single CSV file
    all_results_df = pd.DataFrame(all_results)
    all_results_df_path = os.path.join(path_to_result_folder, "all_statistical_results_factors.csv")
    all_results_df.to_csv(all_results_df_path, index=False)    


    #print(all_combined_df)

        # Create facet grids for Pearson's correlation and DTW distance
    for column in ['Pearson_Correlation', 'DTW distance']:
        g = sns.FacetGrid(all_combined_df, row="Factor", col='Phase', hue="Type", margin_titles=True, sharey=True)
        g.map(sns.boxplot, "Type", column)
        g.add_legend()
   
        plt.subplots_adjust(top=0.9)
        plt.subplots_adjust(top=0.9)
        suptitle = f'{column} by Phase and Factor - same limit on y-axis'
        g.fig.suptitle(suptitle, fontsize=16)
        # Save the plot with the same name as the suptitle in the specified folder
        file_path = os.path.join(path_to_result_folder, f"{suptitle}.png")
        g.savefig(file_path)
        #g.fig.suptitle(f'{column} by Phase and Factor - same limit on y-axis')
        plt.show()

    # Create facet grids for Pearson's correlation and DTW distance
    for column in ['Pearson_Correlation', 'DTW distance']:
        g = sns.FacetGrid(all_combined_df, row="Factor", col='Phase', hue="Type", margin_titles=True, sharey=False)
        g.map(sns.boxplot, "Type", column)
        g.add_legend()
      
        plt.subplots_adjust(top=0.9)
        suptitle = f'Factors - {column} by Phase and Factor - not same limit on y-axis'
        g.fig.suptitle(suptitle, fontsize=16)
        # Save the plot with the same name as the suptitle in the specified folder
        file_path = os.path.join(path_to_result_folder, f"{suptitle}.png")
        g.savefig(file_path)
        #g.fig.suptitle(f'{column} by Phase and Factor - not same limit on y-axis')
        plt.show()

