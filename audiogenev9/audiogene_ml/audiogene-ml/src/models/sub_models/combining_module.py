import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

def one_hot_encode_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    # One hot encode the predictions
    ohe_predictions = pd.get_dummies(predictions_df['prediction'])

    # Insert columns for unpredicated loci for future proofing data integrity
    ohe_predictions = _add_numerical_range_cols(ohe_predictions, len(predictions_df['locus'].unique()))
    
    # Rearrange the one hot encoded columns for easier accessing
    ohe_predictions = ohe_predictions[sorted(ohe_predictions.columns)]
    ohe_predictions = predictions_df.join(ohe_predictions)

    # Add 'locus' to column name for easier extraction by searching for 'locus' in col
    ohe_predictions.columns = [x if type(x) == str else f"locus_{x}" for x in ohe_predictions.columns]
    return ohe_predictions


def combine_predictions(df: pd.DataFrame, combining_type: str = 'majority_vote', top_n: int = 3) -> pd.DataFrame:
    # One hot encode submodel predictions
    if combining_type == 'majority_vote':
        ohe_grouped = df.loc[:, ['id_num', *[c for c in df.columns if 'locus_' in c]]].groupby('id_num')
        ohe_summed = ohe_grouped.sum()
        id_to_locus = df.drop_duplicates(subset='id_num').loc[:, ['id_num', 'locus']]
        ranked = ohe_summed.values.argsort(axis=1)[:, :top_n]
        results = pd.DataFrame(ranked, index=ohe_summed.index, columns=[f"Rank {h}" for h in range(1, top_n+1)])
        return results.merge(id_to_locus, on='id_num')


def _add_numerical_range_cols(df: pd.DataFrame, n_cols: int) -> pd.DataFrame:
    for i in range(n_cols):
        try:
            df.loc[:, i]
        except KeyError:
            df[i] = 0
    return df


def add_weighted_numerical_range_cols(df: pd.DataFrame, n_cols: int, weights_col: str) -> pd.DataFrame:
    cols = [x for x in range(n_cols)]
    # np.vstack unnests the lists of values while stacking the rows of the series
    data = np.vstack(df[weights_col].values)
    d = pd.DataFrame(data, columns = cols)
    return pd.DataFrame.join(df, d).drop(columns=weights_col)


def expand_predictions(df: pd.DataFrame, n_genes: int,) -> pd.DataFrame:
    def _special_zip(_df: pd.DataFrame, class_: str, val: str, n_classes: int) -> pd.Series:
        arr = np.zeros(n_classes,)
        for _, c, v in _df[[class_, val]].itertuples():
            arr[c] = v
        return arr

    df = df.copy()
    p = df[['prediction', 'classes']].explode(['classes', 'prediction'])
    p = p.reset_index().groupby('index')
    df['prediction'] = p.apply(_special_zip, 'classes', 'prediction', n_genes)
    df = df.drop(columns='classes')
    df = add_weighted_numerical_range_cols(df, n_genes, 'prediction')
    return df


def get_top_n_predictions(df: pd.DataFrame, n_kept:int, n_genes: int) -> pd.DataFrame:
    tabular_preds = df.iloc[:, -n_genes:]
    
    top_n = []
    for _, x in tabular_preds.iterrows():
        x_top_n = x.sort_values(ascending=False).iloc[:n_kept]
        top_n.append(np.append(x_top_n.values, x_top_n.index.values))
            
    prob_labels = [f"Rank {i+1} Prob" for i in range(n_kept)]
    class_labels = [f"Rank {i+1} Class" for i in range(n_kept)]
    
    labels = prob_labels + class_labels
    d = pd.DataFrame(top_n, columns=labels)
    top_n_df = df.join(d).drop(columns=[x for x in range(n_genes)])
    return top_n_df


def select_sub_predictions(df: pd.DataFrame, 
                           model_type_contributions: dict = {'size': 2, 'age': 3, 'shape': 3},
                           keep_probs: bool = False) -> pd.DataFrame:
    if not keep_probs:
        df = df.drop(columns=df.columns[df.columns.str.contains('Prob')])
    
    all_cols = set(); f = True
    kept_rankings = []
    for type_, n in model_type_contributions.items():
        type_df = df[df['model'].str.replace('shape_age', 'shape').str.contains(type_)]
        type_df.columns = [c.replace('Class', type_) if 'Rank' in c else c for c in type_df.columns]
        
        if type_ == 'size':
            type_df = type_df.iloc[:, :n-3]
            sizes = type_df['instance_group'].unique()

            for s in sizes:
                size_df = type_df[type_df['instance_group'] == s]
                size_df.columns = [c.replace('size', f"size-{s}") if 'Rank' in c else c for c in size_df.columns]
            
                kept_cols = ['id_num'] + [c for c in size_df.columns if 'size' in c]
                kept_rankings.append(size_df.loc[:, kept_cols])
                all_cols.update(kept_cols)

        else:
            # TODO: Refactor this to be more flexable to n
            kept_cols = ['id_num'] + [c for c in type_df.columns if 'Rank' in c]
            kept_rankings.append(type_df.loc[:, kept_cols])
            all_cols.update(kept_cols)
    
    
    size_rankings = [rankings for rankings in kept_rankings if rankings.shape[1] == 3]
    size_cols = set()
    for ranking in size_rankings:
        for col in ranking.columns:
            size_cols.add(col)
    rankings_df = pd.concat(kept_rankings)
    return rankings_df

def compose_rankings_by_id(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Correct duplicate column names
    # NOTE: Performance is not impacted at this time by incorrect column names
    def _compose_id_ranking(id_df: pd.DataFrame):
            no_id_dropped_all_na_cols = id_df[id_df.columns[~id_df.isna().all()]]
            return no_id_dropped_all_na_cols
    
    df = df.drop_duplicates()
    dfs = []
    for id in df['id_num'].unique():
        composed_id_df = _compose_id_ranking(df[df['id_num'] == id])
        composed_id_df.columns = ['id_num', 'Rank 1 size', 'Rank 2 size', 'Rank 1 age','Rank 2 age', 'Rank 3 age', 'Rank 1 shape', 'Rank 2 shape', 'Rank 3 shape']
        stacked = composed_id_df.drop(columns='id_num').stack().reset_index().drop(columns='level_0')#.drop(columns='level_1')
        stacked = stacked.set_index('level_1').T
        stacked.index = [id]
        stacked = stacked.astype(int)
        dfs.append(stacked)
        
    return pd.concat(dfs)


def ohe_proba_predictions(df: pd.DataFrame, gene_encoder: OneHotEncoder) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.ndarray((df.shape[0], len(gene_encoder.get_feature_names_out())))
    for pos, idx in enumerate(df.index):
        x = df.loc[idx].to_numpy().reshape(-1,1)
        x = gene_encoder.transform(x)
        x = x.sum(axis=0).reshape(1,-1)
        arr[pos] = x
    return arr, df.index.to_numpy()