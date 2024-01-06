import pandas as pd
from sklearn.utils import resample,shuffle

def upsampling(df_count_merge, n, state, nn=0): 
    unique_locus = df_count_merge.locus.unique()
    df_count_merge_resample = pd.DataFrame()
    for u in unique_locus:
        df_count_merge_filter = df_count_merge[df_count_merge['locus'] == u]

        if n < 0:
            # Downsampling
            if (df_count_merge_filter.shape[0] > (n * -1)):
                df_count_merge_filter = resample(df_count_merge_filter, random_state=state,n_samples=(n * -1),replace=False, stratify=df_count_merge_filter)
        # Upsampling
        if (df_count_merge_filter.shape[0] >= nn) and (df_count_merge_filter.shape[0] < n):
            df_count_merge_filter = resample(df_count_merge_filter, random_state=state,n_samples=n,replace=True, stratify=df_count_merge_filter)
            
        df_count_merge_resample = pd.concat([df_count_merge_resample, df_count_merge_filter])
                
    return df_count_merge_resample


def resampling(df: pd.DataFrame, sampling_thresholds: dict, upsample = True, 
               downsample = True, verbose: bool = False) -> pd.DataFrame:
    resampled = []
    if verbose:
        old = 0
        new = 0
    df.sort_values(by='instance_group', inplace=True)
    for locus in df['locus'].unique():
        locus_df = df[df['locus'] == locus]
        locus_bin = locus_df['instance_group'].values[0]
        
        if verbose:
            print(f"{locus_bin.upper()} Locus")
        
        sampling_criteria = sampling_thresholds[locus_bin]
        
        if sampling_criteria['type'] == 'up' and locus_df.shape[0] < sampling_criteria['cutoff'] and upsample:
            resampled_df = resample(locus_df, **sampling_criteria['criteria'], stratify=locus_df)
        elif sampling_criteria['type'] == 'down' and locus_df.shape[0] > sampling_criteria['cutoff'] and downsample:
            resampled_df = resample(locus_df, **sampling_criteria['criteria'], stratify=locus_df)
        else:
            resampled_df = locus_df
        
        resampled.append(resampled_df)

        
        if verbose:
            old += len(locus_df)
            new += len(resampled_df)
            print(f"{locus_df['locus'].values[0]}\nOld size: {len(locus_df)}\nNew size: {len(resampled_df)}")
            
    if verbose: print(f"Overall: \nOld size: {old}\nNew size: {new}")
        
    return pd.concat(resampled)