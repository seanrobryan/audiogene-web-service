import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIG_TITLE_SIZE = 20
SUB_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14


def plot_gene_audiograms(df, gene, subset=False, subset_size=10, fig_size=(10, 6), opacity=0.15, color='k',
                         linestyle='-', marker='o', xlims=[0,8000], ylims=[0,130]):
    '''
    @param df pd.DataFrame containing audiometric data
    @param gene String with the gene name
    @param subset Boolean flag to include only a limited number of audiograms on the plot
    @subset_size Int the number of audiograms to include if the subset is True
    @param fig_size
    @param opacity float controlling the alpha value of the matplotlib plot 
    @param color
    @param linestyle
    @param marker
    @param xlims
    @param ylims
    

    @returns matplotlib figure and axis containing the graph of all audiograms selected for plotting
    '''
    gene_df = get_gene_df(df, gene).drop(columns=['age', 'gene'])
    columns = gene_df.columns.to_numpy().astype(np.int)
    
    title = f"{gene} Audiogram"
    
    if subset:
        gene_df = gene_df.sample(n=subset_size)
        if subset_size > 1:
            title = title + 's'

    fig = plt.figure(num=None, figsize=fig_size)
    ax1 = plt.subplot(111)

    for audiogram in gene_df.values:
        plt.plot(columns, audiogram, f"{color}{marker}{linestyle}",alpha=opacity)
    
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.gca().invert_yaxis()
    ax1.xaxis.tick_top() 
    ax1.set_xlabel('Frequency (Hz)')    
    ax1.xaxis.set_label_position('top') 
    ax1.set_ylabel('Hearing Loss (dB)')
    
    plt.title(title)

    plt.yticks(np.arange(*ylims, 10))
    plt.grid()

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.79, box.height*.9])

    return fig, ax1


def plot_mean_gene_audiograms(df, gene, subset=False, subset_size=10, fig_size=(10,6), opacity=0.15):
    '''
    @param df pd.DataFrame containing audiometric data
    @param gene String with the gene name
    @param subset Boolean flag to include only a limited number of audiograms on the plot
    @subset_size Int the number of audiograms to include if the subset is True
    @param fig_size 
    @param opacity float controlling the alpha value of the matplotlib plot

    @returns fig, ax1 matplotlib figure and axis containing the graph of them mean audiogram
    '''
    gene_df = get_gene_df(df, gene).drop(columns=['age', 'gene'])
    columns = gene_df.columns.to_numpy().astype(np.int)
    
    if subset:
        gene_df = gene_df.sample(n=subset_size)

    fig = plt.figure(num=None, figsize=fig_size)
    ax1 = plt.subplot(111)

    gene_mean = get_mean_audiogram(gene_df)
    plt.plot(columns, gene_mean, 'k-', alpha=opacity)


    plt.ylim([0,130])
    plt.xlim([0,8000])
    plt.gca().invert_yaxis()
    ax1.xaxis.tick_top() 
    ax1.set_xlabel('Frequency (Hz)')    
    ax1.xaxis.set_label_position('top') 
    ax1.set_ylabel('Hearing Loss (dB)')
    
    plt.title(f"{gene} Audiograms")

    plt.yticks( np.linspace(0,130,14) )
    plt.grid()

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.79, box.height*.9])

    return fig, ax1


def get_mean_audiogram(df):
    '''
    @param df pd.DataFrame containing audiometric data

    @returns A pd.Series containing the mean of each hearing loss column
    '''
    cols = [c for c in df.columns if c not in ['age', 'gene', 'id']]
    return df.loc[:, cols].mean(axis=0)



def plot_mean_audiograms_by_age(df, gene = None, age_bins=[20, 40, 60, 80, 100],
                                fig_size=(10,6), opacity=0.5, colors=['#a6ceff', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99']):
    '''
    @param df pd.DataFrame containing audiometric data
    @param gene String with the gene name
    @param age_bins Upper limit of each age bin
    @param fig_size 
    @param opacity float controlling the alpha value of the matplotlib plot

    @returns fig, ax1 The matplotlib figure and axis objects containing the plot
    '''
    if gene is not None:
        gene_df = get_gene_df(df, gene).drop(columns='gene')
    else:
        gene = df['gene'][0]
        gene_df = df.drop(columns='gene')
    columns = gene_df.drop(columns='age').columns.to_numpy().astype(np.int)
    
    # Getting mean of audiogram age bin
    binned_age_means_df = get_age_bins_df(gene_df, age_bins)

    fig = plt.figure(num=None, figsize=fig_size)
    ax1 = plt.subplot(111)

    for index, audiogram, color in zip(binned_age_means_df.index, binned_age_means_df.values, colors):
        plt.plot(columns, audiogram, alpha=opacity, label=f"Ages: {index}", color=color)    

    plt.ylim([0,130])
    plt.xlim([0,8000])
    plt.gca().invert_yaxis()
    ax1.xaxis.tick_top() 
    ax1.set_xlabel('Frequency (Hz)', fontsize=AXIS_LABEL_SIZE)    
    ax1.xaxis.set_label_position('top') 
    ax1.set_ylabel('Hearing Loss (dB)', fontsize=AXIS_LABEL_SIZE)
    
    plt.title(f"{gene} Audiograms")

    plt.yticks( np.linspace(0,130,14) )
    plt.grid()

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.79, box.height*.9])

    return fig, ax1


def get_age_bins_df(df, bins):
    '''
    @param df pandas DataFrame with auidograms
    @param bins list of ints containing the upper bound of the age brackets
    '''
    bin_range = int(abs(bins[0] - bins[-1]) / (len(bins)-1))
    bin_means = []

    for age in bins:
        is_below_max = df['age'] <= age
        is_above_min = df['age'] > (age-bin_range)
        cur_age_df = df.loc[is_below_max & is_above_min]
        
        bin_means.append(get_mean_audiogram(cur_age_df))
        
    labels = [f"{x - bin_range} - {x}" for x in bins]
    return pd.concat(bin_means, axis=1, keys=labels).T


def get_gene_df(df, gene):
    '''
    @param df: Pandas DataFrame 
    @param gene: String representation for gene of interest
    
    @returns: All instances of gene in df
    '''
    return df[df['gene'] == gene]


def average_age_range(df, age, range_, avg, gene=None):
    '''
    @param df: Pandas DataFrame 
    @param age: Center of the age range of interest
    @param range_: Ages +/- to be included in the aggregated
    @param func: The type of avg
    @param gene: String representation for gene of interest
    '''
    
    if gene is not None:
        gene_df = get_gene_df(df, gene)
    else:
        gene_df = df
    age_diff = gene_df['age'] - age
    is_in_range = abs(age_diff) <= range_
    
    if avg == 'mean':
        return gene_df.loc[is_in_range].mean()
    elif avg == 'median':
        return gene_df.loc[is_in_range].median()
    else:
        print('Available average not selected. Returning original dataframe.')
        return df


def average_age_bin(df, age, bin_size, avg, gene=None):
    '''
    @param df: Pandas DataFrame 
    @param age: Top of the age range of interest
    @param bin_size: Size of bin, i.e., years below age to be included
    @param func: The type of avg
    @param gene: String representation for gene of interest
    '''
    if gene is not None:
        gene_df = get_gene_df(df, gene)
    else:
        gene_df = df

    is_above_min = gene_df['age'] > bin_size[0]
    is_under_max = gene_df['age'] <= bin_size[1]
    
    if avg == 'mean':
        return gene_df.loc[is_under_max & is_above_min].mean()
    elif avg == 'median':
        return gene_df.loc[is_under_max & is_above_min].median()
    else:
        print('Available average not selected. Returning original dataframe.')
        return df
