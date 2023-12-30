import numpy as np
import pandas as pd
import src.features.shape_sorting as shape_sorting
import src.features.size_sorting as size_sorting

from src.constants import PARTITION_PARAMETERS, SHAPE_THRESHOLDS


def add_all_features(df: pd.DataFrame, shape_thresholds=SHAPE_THRESHOLDS,
                     parameters=PARTITION_PARAMETERS):

    bin_names = parameters['bin_names']
    bin_thresholds = parameters['bin_thresholds']
    freq_groups = parameters['freq_groups']

    # Copy the dataframe to preserve the original
    df = df.copy()
    # List of all db loss features
    col_freq = [col for col in df.columns if "dB" in col]
    
    # Split the cols into low, medium, high frequencies
    low, medium, high = col_freq[0:freq_groups[0]], col_freq[3:freq_groups[1]], col_freq[7:freq_groups[2]]
    
    # Merge bin_names with bin_thresholds into dictionary
    size_bins = dict(zip(bin_names, bin_thresholds))
    
    # Apply the transformations 
    
    # Add audiogram age group category
    df = shape_sorting.make_df_with_age_groups(df)
    
    # Add audiogram shape
    df = shape_sorting.shape_rule_2(df, low, medium, high, 
                                    shape_thresholds[0], shape_thresholds[1], np.median, np.max)

    # Add locus sample size bin
    df = size_sorting.filter_class_size(df, size_bins)
    
    return df
