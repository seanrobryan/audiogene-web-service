import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def split_counts(df: pd.DataFrame, enc: LabelEncoder = None, 
                 set_col: str = 'set', target: str = 'locus', 
                 set_names: list = ['Train', 'Test']) -> pd.DataFrame:
    """
    Creates a report how instances are broken into test and train sets
    """
    # Split the sets, counts the values, recombine into a dataframe
    count_df = pd.concat([df.loc[df[set_col] == set_, target].value_counts() for set_ in set_names], axis=1)
    count_df.columns = set_names
    
    count_df['Total'] = sum([count_df.loc[:, set_] for set_ in set_names])
    
    for set_ in set_names:
        count_df[f"{set_} Frac"] = count_df[set_]/count_df['Total']
    
    if enc is not None:
        count_df.index = enc.inverse_transform(count_df.index)
    return count_df


def plot_class_distribution(df, save_fig_to: str = None):
    small_threshold = 20
    large_threshold = 300
    
    if isinstance(df, str):
        df = pd.read_csv(df).drop(columns=['Unnamed: 0','2c0','2c1','3c0','3c1','3c2'])


    # Create a new Column
    df_count = df.groupby(['locus']).size().reset_index(name='counts')
    
    # TODO: Replace all this with the functions I've already written
    # df_count.sort_values(by='counts', ascending=False)
    df_count_merge = df.merge(df_count, on=['locus'])

    # Condition Instances (Can be Changed Later)
    conditions = [
        (df_count_merge['counts'] <= small_threshold),
        (df_count_merge['counts'] > small_threshold) & (df_count_merge['counts'] <= large_threshold),
        (df_count_merge['counts'] > large_threshold),
    ]

    # Create a list of the values we want to assign for each condition
    values = ['small', 'medium', 'large']

    # Create a new column and use np.select to assign values to it using our lists as arguments
    df_count_merge['shape'] = np.select(conditions, values)
    df_count_merge_filter = df_count_merge[['locus', 'shape']].value_counts().reset_index(name='counts').sort_values(by='counts', ascending=True).reset_index(drop=True)

#     df_count_merge_filter.to_csv(file_parameter_instances, index=False)

    fig,ax = plt.subplots(figsize=(30, 20))
    df_count.sort_values('counts').plot.bar(x =  'locus', y = 'counts', ax = ax, legend=False)
    ax.set_ylabel('Number of Audiograms per Gene')
    ax.set_xlabel('Gene')
    ax.tick_params(labelsize='large')

    axis_labels = df_count_merge_filter.groupby(by=['shape']).idxmax().sort_values(by='counts').iloc[0:-1].reset_index(drop=True)
    old_value = 0
    c = ['yellow', 'green']
    for a in range(axis_labels.shape[0]):
        ax.axvline(axis_labels['counts'][a] + 0.5, color="red", linestyle="--", lw=2, label="lancement")
        plt.axvspan(old_value, axis_labels['counts'][a] + 0.5,  alpha=0.5, color=c[a])
        old_value = axis_labels['counts'][a] + 0.5
    fig.set_size_inches(15, 12)
    plt.axvspan(old_value, df_count_merge_filter.shape[0], alpha=0.5, color='purple')
    
    if save_fig_to is not None:
        fig.savefig(save_fig_to, dpi=200)
    
    return fig