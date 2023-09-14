from pandas import DataFrame
from icecream import ic

def filter_class_size(df: DataFrame, bins: dict):
    """
    Counts the number of instances of each class in the pd.DataFrame and uses
    the bins to label each instance of the DataFrame by it's been.
    Args:
        bins (dict): key (str) name of the bin, value (int) lower bound of the bin
        df (DataFrame): contains audiometric data
    """
    
    def _size_filter_helper(class_size: int, names: list, thresholds: list):
        """
        Helper function to the class size filter.
        
        Args:
            df: pd.DataFrame
            names (list): Name of instance categories. Expected size = 3
            thresholds (list): Thresholds for intstance size categorization. Expected size = 3
        """
        if class_size < thresholds[1]: # Less than the second threshold -> first bin
            bin_ = names[0]
        elif class_size > thresholds[2]: # Greater than the third threshold -> third bin
            bin_ = names[2]
        else:
            bin_ = names[1] # Everything else -> second bin
        return bin_
    
    locus_counts_df = df.groupby(['locus']).size().reset_index(name='counts')
    locus_counts_df.sort_values(by='counts', inplace=True)
    
    keys = [*bins.keys()]
    values = [*bins.values()]

    locus_counts_df['group'] = locus_counts_df['counts'].apply(_size_filter_helper, names=keys, thresholds=values)
    locus_counts_df.drop(columns='counts', inplace=True)
    locus_counts_df.set_index('locus', inplace=True)
    mapping = locus_counts_df.to_dict()['group']
    
    df['instance_group'] = df['locus'].map(mapping)
    
    return df
    
