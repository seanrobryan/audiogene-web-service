import pandas as pd
from scipy.io import arff
from typing import List, Union

from src.scripts.convert_loci_to_gene import convert_loci_to_gene
from src.constants import AUDIOGRAM_LABELS, POLYNOMIAL_COEFFICIENTS_LABELS, GENOTYPE_LABELS

labels = AUDIOGRAM_LABELS + POLYNOMIAL_COEFFICIENTS_LABELS


def arff_to_mi_df(filepath: str, add_gene: bool = True) -> pd.DataFrame:
    arff_dat = arff.loadarff(filepath)
    df = pd.DataFrame(arff_dat[0])
    df.locus = df.locus.astype('string').str.replace("b'", '').str.replace("'", '')
    df.id = df.id.astype('string').str.replace("b'", '').str.replace("'", '')
    
    if add_gene:
        df['gene'] = df.locus.apply(convert_loci_to_gene)
    
    return df


def mi_df_to_si_df(mi_df: pd.DataFrame, mi_col: str = 'bag') -> pd.DataFrame:
    interm_df = mi_df.explode(mi_col).reset_index().drop(columns='index')
    s = interm_df.loc[:, [mi_col]]
    arrs = []
    for bag in s.values:
        arr = []
        for v in bag[0]:
            arr.append(v)
        arrs.append(arr)
    
    audiogram_df = pd.DataFrame(arrs, columns=labels)
    si_df = pd.concat([interm_df.drop(columns=['bag']), audiogram_df], axis=1)
    return si_df


def si_df_to_mi_df(si_df: pd.DataFrame, bag_on: str = 'id', mi_col_name: str = 'bag',
                   tgt_cols: Union[List[str], str] = GENOTYPE_LABELS) -> pd.DataFrame:
    bags = []
    bag_ids = si_df[bag_on].unique()
    for id in bag_ids:
        group = si_df.loc[si_df[bag_on] == id, :]
        tgts, feats = group.loc[:, tgt_cols], group.loc[:, labels].to_numpy()
        unq_tgts =[]
        for c in tgt_cols:  
            vals = tgts.loc[:, c]
            if len(vals.unique()) != 1:
                raise ValueError(f"{len(vals.unique())} unique values found for {bag_on} = {id} at column {c}, expected 1.")
            else:
                unq_tgts.append(vals.unique()[0])
        bags.append([id] + unq_tgts + [feats])
    
    return pd.DataFrame(bags, columns=[bag_on] + tgt_cols + [mi_col_name])


def arff_mi_to_si_df(filepath: str) -> pd.DataFrame:
    return mi_df_to_si_df(arff_to_mi_df(filepath))

