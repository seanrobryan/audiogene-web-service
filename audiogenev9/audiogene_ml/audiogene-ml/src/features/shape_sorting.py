import numpy as np
from pandas import DataFrame


# TODO: Determine a more descriptive name for this func based on downstream usecases
def fun_case(df: DataFrame, threshold, threshold2):
    """
    Categorizes the audioprofile characteristics as cooke bite, downsloping, or upsloping.

    Parameters
    ----------
    df: pd.DataFrame
    threshold : int
    threshold2 : int
    """
    if (abs(df['low'] - df['high']) < threshold) & \
        (df['medium'] - df['low'] > threshold2) & \
        (df['medium'] - df['high'] > threshold2):
        shape = 'cookie-bite'
    elif df['low'] - df['high'] < 0:
        shape = 'downsloping'
    else:
        shape = 'upsloping'
    return shape


# TODO: Make this age agnostic
def make_df_with_age_groups(df: DataFrame):
    """
    Adds the column 'age_group' to the DataFrame to organize instances
    by age categorically. Rows with an age under 20 are labeled as '0-20'
    and all other rows are labeled '20+'.
    """
    df['age_group'] = np.where(df['age'] < 20, '0-20', '20+')
    return df


# Formerly age_group_f_20_ex
def calc_mean_loss_per_locus_by_age(df: DataFrame):
    """
    Returns a DataFrame with instances group by loci and age with the mean
    dB loss value for the groups.
    """
    df = make_df_with_age_groups(df)
    # TODO: Refactor to determine within the function what cols are not frequenies
    non_freq_cols = ['age', 'id', 'id_num']
    for col in non_freq_cols:
        if col not in df.columns:
            non_freq_cols.remove(col)
    df = df.drop(labels=non_freq_cols, axis=1)
    df = df.groupby(['locus', 'age_group']).mean()
    df.columns = df.columns.str.replace('dB', '')
    return df


# TODO: Determine a more descriptive name for this func based on downstream usecases
def shape_rule_2(
    df: DataFrame,
    low: list, medium: list, high: list, threshold: int = 25, 
    threshold2: int = 10, func = np.median, func2 = np.max):
    """
    
    """
    # Get the full data frame with the categorical age column added
    df = make_df_with_age_groups(df)

    ### Apply the appropriate functions to each subset of the df and insert the shapes into the df
        # On the subset of cols `low` for every row, apply the func and store the result for each row
        # in the new column 'low'
    df['low'] = df[low].apply(func=func, axis=1)
    # Repeat for medium and high
    df['medium'] = df[medium].apply(func=func2, axis=1)
    df['high'] = df[high].apply(func=func, axis=1)

    df['shape'] = df.apply(fun_case, args=(threshold, threshold2), axis=1)
    return df
