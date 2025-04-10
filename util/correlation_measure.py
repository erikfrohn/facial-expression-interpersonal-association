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

def crqa(name, p1_data, p2_data, embedding_dimension=3,time_delay=1,radius=0.5, plot=True, normalize=True):
    if normalize:
        p1_data = (p1_data - np.mean(p1_data, axis=0)) / np.std(p1_data, axis=0)
        p2_data = (p2_data - np.mean(p2_data, axis=0)) / np.std(p2_data, axis=0)

    # Create time series objects
    time_series1 = TimeSeries(p1_data,
                            embedding_dimension=embedding_dimension,  # No embedding needed if you're using CCA components
                            time_delay=time_delay)
    time_series2 = TimeSeries(p2_data,
                            embedding_dimension=embedding_dimension,
                            time_delay=time_delay)

    # Configure settings
    settings = Settings(time_series1,
                    time_series2,
                    neighbourhood=FixedRadius(radius),  # Adjust radius based on your data
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1)

    # Perform computation
    computation = RQAComputation.create(settings)
    result = computation.run()

    if plot:
        # Print results
        print("Recurrence rate: %.4f" % result.recurrence_rate)
        print("Determinism: %.4f" % result.determinism)
        print("Laminarity: %.4f" % result.laminarity)
        print("Average diagonal line length: %.4f" % result.average_diagonal_line)
        print("Longest diagonal line length: %d" % result.longest_diagonal_line)

        computation = RPComputation.create(settings)
        result2 = computation.run()
        ImageGenerator.save_recurrence_plot(result2.recurrence_matrix_reverse,
                                            f'{name}.png')
    return result.recurrence_rate, result.determinism, result.laminarity, result.average_diagonal_line, result.longest_diagonal_line
