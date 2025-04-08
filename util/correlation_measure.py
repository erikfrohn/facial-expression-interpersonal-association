from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.analysis_type import Cross
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
import numpy as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def binarize_components(components, threshold=0.5):
    """
    Scale each component to [0, 1] and binarize based on a threshold.
    
    Parameters:
    -----------
    components : np.ndarray, shape (n_components, n_timepoints)
        Correlated component time series.
    threshold : float, default=0.5
        Threshold for binarization (values >= threshold become 1).
        
    Returns:
    --------
    binary_components : np.ndarray, shape (n_components, n_timepoints)
        Binarized components (0 or 1).
    """
    # Scale each component to [0, 1] (within-component)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(components.T).T  # Transpose for sklearn
    
    # Binarize based on component-specific median
    medians = np.median(scaled, axis=1, keepdims=True)  # Shape (n_components, 1)
    binary_components = (scaled >= medians).astype(int)
    
    return binary_components









#### CRQA DEV HELL :(
def crqa(name, nav_data, pil_data):
    time_series_nav = TimeSeries(nav_data,
                            embedding_dimension=2,
                            time_delay=1)
    time_series_pil = TimeSeries(pil_data,
                            embedding_dimension=2,
                            time_delay=2)
    time_series = (time_series_nav,
                time_series_pil)
    settings = Settings(time_series,
                        analysis_type=Cross,
                        neighbourhood=FixedRadius(0.73),
                        similarity_measure=EuclideanMetric,
                        theiler_corrector=0)
    computation = RQAComputation.create(settings,
                                        verbose=True)
    result1 = computation.run()
    result1.min_diagonal_line_length = 2
    result1.min_vertical_line_length = 2
    result1.min_white_vertical_line_length = 2
    print(result1)
    computation = RPComputation.create(settings)
    result2 = computation.run()
    ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse,
                                        f'{name}_cross_recurrence_plot.png')
    return result1, result2


from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.time_series import TimeSeries

    # Prepare your data (assuming you've loaded it as numpy arrays)
    # p1_data and p2_data should be N x 6 arrays for 6 components

    # Normalize the data
def crqa2(name, p1_data, p2_data):
    p1_data = (p1_data - np.mean(p1_data, axis=0)) / np.std(p1_data, axis=0)
    p2_data = (p2_data - np.mean(p2_data, axis=0)) / np.std(p2_data, axis=0)

    # Create time series objects
    time_series1 = TimeSeries(p1_data,
                            embedding_dimension=1,  # No embedding needed if you're using CCA components
                            time_delay=1)
    time_series2 = TimeSeries(p2_data,
                            embedding_dimension=1,
                            time_delay=1)

    # Configure settings
    settings = Settings(time_series1,
                    time_series2,
                    neighbourhood=FixedRadius(0.5),  # Adjust radius based on your data
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1)

    # Perform computation
    computation = RQAComputation.create(settings)
    result = computation.run()

    # Print results
    print("Recurrence rate: %.4f" % result.recurrence_rate)
    print("Determinism: %.4f" % result.determinism)
    print("Laminarity: %.4f" % result.laminarity)
    print("Average diagonal line length: %.4f" % result.average_diagonal_line)
    print("Longest diagonal line length: %d" % result.longest_diagonal_line)

    computation = RPComputation.create(settings)
    result2 = computation.run()
    ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse,
                                        f'{name}_cross_recurrence_plot2.png')
    

# windowed DO NOT USE
def crqa3(p1_data, p2_data):
    print("PLEASE MAKE SURE YOU WANT TO USE THIS")
    return 0
    window_size = 100  # frames
    step_size = 20     # frames

    results = []
    for i in range(0, len(p1_data)-window_size, step_size):
        window1 = p1_data[i:i+window_size]
        window2 = p2_data[i:i+window_size]
        
        # Normalize within window
        window1 = (window1 - np.mean(window1, axis=0)) / np.std(window1, axis=0)
        window2 = (window2 - np.mean(window2, axis=0)) / np.std(window2, axis=0)
        
        # Compute cRQA
        time_series1 = TimeSeries(window1, embedding_dimension=1, time_delay=1)
        time_series2 = TimeSeries(window2, embedding_dimension=1, time_delay=1)
        settings = Settings(time_series1, time_series2, 
                        neighbourhood=FixedRadius(0.5),
                        similarity_measure=EuclideanMetric)
        result = RQAComputation.create(settings).run()
        
        results.append({
            'window_start': i,
            'RR': result.recurrence_rate,
            'DET': result.determinism,
            })
        
        computation = RPComputation.create(settings)
        result2 = computation.run()
        ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse,
                                            f'{i}_help_cross_recurrence_plot.png')
        

import matplotlib.pyplot as plt


def mimicry(p1_data, p2_data):
    p1_data = (p1_data - np.mean(p1_data, axis=0)) / np.std(p1_data, axis=0)
    p2_data = (p2_data - np.mean(p2_data, axis=0)) / np.std(p2_data, axis=0)

    # Create time series objects
    time_series1 = TimeSeries(p1_data,
                            embedding_dimension=1,  # No embedding needed if you're using CCA components
                            time_delay=1)
    time_series2 = TimeSeries(p2_data,
                            embedding_dimension=1,
                            time_delay=1)

    # Using your normalized time series data from before
    settings = Settings(time_series1,
                    time_series2,
                    neighbourhood=FixedRadius(0.5),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1)

    # Compute the full recurrence plot
    computation = RPComputation.create(settings)
    result = computation.run()

    # After computing your recurrence matrix (from pyRQA or custom implementation)
    mimicry_results = analyze_mimicry_lags(result.recurrence_matrix_reverse, 
                                        max_lag=45,  # 1.5 seconds at 30fps
                                        bin_size=3)

    # Interpret the results
    if mimicry_results['significant_peaks']:
        print("Significant mimicry lags found:")
        for lag, z, rr in mimicry_results['significant_peaks']:
            direction = "P2 follows P1" if lag > 0 else "P1 follows P2"
            print(f"• Lag {abs(lag):.0f} frames ({direction}), z={z:.2f}")
            
        dominant = mimicry_results['dominant_peak']
        print(f"\nDominant mimicry pattern: {abs(dominant[0]):.0f} frames " 
            f"({'P2 mimics P1' if dominant[0] > 0 else 'P1 mimics P2'})")
    else:
        print("No statistically significant mimicry lags detected")



import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

def analyze_mimicry_lags(recurrence_matrix, max_lag=60, bin_size=3):
    """
    Properly normalized diagonal profile analysis for mimicry
    
    Parameters:
    - recurrence_matrix: 2D numpy array from cRQA
    - max_lag: maximum lag to analyze (in frames)
    - bin_size: how many lags to average together (reduces noise)
    
    Returns:
    - Dictionary with lag analysis results
    - Plot of normalized lag profile
    """
    
    # Initialize storage
    raw_lag_profiles = []
    valid_lengths = []
    
    # Calculate recurrence rate at each lag
    for lag in range(-max_lag, max_lag+1):
        diag = np.diag(recurrence_matrix, k=lag)
        if len(diag) > 0:
            raw_lag_profiles.append((lag, np.mean(diag)))
            valid_lengths.append(len(diag))
    
    # Convert to arrays
    lags, rr = zip(*raw_lag_profiles)
    lags, rr, lengths = np.array(lags), np.array(rr), np.array(valid_lengths)
    
    # Normalize by expected chance recurrence
    overall_rr = np.sum(recurrence_matrix) / recurrence_matrix.size
    expected_counts = lengths * overall_rr
    observed_counts = rr * lengths
    normalized_rr = (observed_counts - expected_counts) / (lengths - expected_counts + 1e-10)
    
    # Bin the results (reduces noise)
    binned_lags = []
    binned_rr = []
    for i in range(0, len(lags), bin_size):
        bin_lags = lags[i:i+bin_size]
        bin_rr = normalized_rr[i:i+bin_size]
        if len(bin_lags) > 0:
            binned_lags.append(np.mean(bin_lags))
            binned_rr.append(np.mean(bin_rr))
    
    # Find significant peaks
    z_scores = zscore(binned_rr)
    peak_threshold = 1.96  # p < 0.05
    significant_peaks = [(lag, score, rr) 
                        for lag, score, rr in zip(binned_lags, z_scores, binned_rr)
                        if score > peak_threshold]
    
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(binned_lags, binned_rr, 'b-', label='Normalized RR')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    
    # Mark significant peaks
    for lag, score, rr in significant_peaks:
        plt.plot(lag, rr, 'ro')
        plt.text(lag, rr, f' {lag:.0f}f\n(z={score:.2f})', 
                va='bottom', ha='center')
    
    plt.xlabel('Time Lag (frames)')
    plt.ylabel('Normalized Recurrence Rate')
    plt.title('Mimicry Lag Profile\n(Normalized, binned, with significant peaks marked)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    return {
        'binned_lags': binned_lags,
        'binned_rr': binned_rr,
        'significant_peaks': significant_peaks,
        'dominant_peak': max(significant_peaks, key=lambda x: x[1]) if significant_peaks else None
    }