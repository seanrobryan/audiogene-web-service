import pandas as pd
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from typing import List, Iterable


# temp.ipynb
# def top_k_confusion_matrix(df: pd.DataFrame, k) -> ConfusionMatrixDisplay:
#     def _gene_or_top_pred(s, k):
#         return s.gene if in_top_k(s, k) else s.iloc[0]
#
#     pred = df.loc[:, [x for x in range(1, 23)] + ['gene']].apply(_gene_or_top_pred, axis=1, args=(k,))
#
#     return ConfusionMatrixDisplay.from_predictions(y_true=df.gene, y_pred=pred, xticks_rotation=60)


# imbalanced-learn-experimenting.ipynb
# comp-temp.ipynb
#############################
# build_ensemble.ipynb
def top_k_confusion_matrix(df: pd.DataFrame, k) -> ConfusionMatrixDisplay:
    def _gene_or_top_pred(s, k):
        return s.gene if in_top_k(s, k) else s.iloc[0]

    pred = df.loc[:, k_labels(k) + ['gene']].apply(_gene_or_top_pred, axis=1, args=(k,))

    return ConfusionMatrixDisplay.from_predictions(y_true=df.gene, y_pred=pred, xticks_rotation=60)


def in_top_k(s: pd.Series, k: int) -> bool:
    return s.gene in s.iloc[:k].values


def k_labels(k: int) -> List[str]:
    return [str(x) for x in range(1, k+1)]


def get_top_k_df(df: pd.DataFrame, k: int, other_cols: Iterable = None) -> pd.DataFrame:
    return df.loc[:, k_labels(k) + list(other_cols)]

