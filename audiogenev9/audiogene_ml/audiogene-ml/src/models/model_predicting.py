import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin

from src.constants import GENES, N_GENES

def make_predictions(df: pd.DataFrame, sub_models: dict, 
                     feature_cols: list, use_proba: bool = True) -> pd.DataFrame:
    predictions = []
    for key, model in sub_models.items():
        partition = _get_partition(df, key)  
        X = partition.loc[:, feature_cols]        
        
        if use_proba:
            test_preds = model.predict_proba(X)
        else:
            test_preds = model.predict(X)
    
        # Storing results
        for id, prediction in zip(partition.index, test_preds):
            kept = partition.loc[id, ['id_num', 'shape', 'age_group', 'instance_group', 'locus']].to_list()
            data = [*kept, key, prediction, model.classes_]
            predictions.append(data)

    predictions_df = pd.DataFrame(predictions, columns=['id_num', 'shape', 'age_group', 
                                        'instance_group', 'locus', 'model', 'prediction', 'classes'])
    return predictions_df

def _get_partition(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if 'size' in key:
            partition = df
    elif 'age' in key:
        partition = df[df['age_group'] == key.split('_')[1]]
        if 'shape_age' in key:
            shape, age = key.split('_')[-2:]
            partition = df[(df['age_group'] == age) & (df['shape'] == shape)]
            
    return partition

def ensemble_predict_n_probs(ensemble: ClassifierMixin, data: pd.DataFrame, n: int) -> pd.DataFrame:
    probs = ensemble.predict_proba(data)
    best_n = get_top_n_predictions(probs, n)
    return best_n

def get_top_n_predictions(preds: pd.DataFrame, n: int):
    return np.argsort(preds, axis=1)[:, -n:]


def get_gene_rankings(arr: np.array) -> pd.DataFrame:
    gene_idxs = {i: g for i, g in zip(range(N_GENES), GENES)}
    ranked = [[gene_idxs[arr[r, c]] for c in range(arr.shape[1])] for r in range(arr.shape[0])]
    return pd.DataFrame(ranked, columns=[str(x) for x in range(1, len(GENES) + 1)])

