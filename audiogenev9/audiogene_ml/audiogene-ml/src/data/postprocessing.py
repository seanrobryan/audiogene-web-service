import numpy as np
import pandas as pd
from src.models.model_predicting import get_gene_rankings
from src.constants import GENES, N_GENES
import os


def probabilities_to_predictions(probabilities: np.ndarray):
    predictions = np.argsort(probabilities, axis=1)[:, ::-1]
    ranked = get_gene_rankings(predictions)
    return ranked


def combined_probs_and_rankings(probabilities, rankings, ids):
    res_df = pd.DataFrame(probabilities, columns=GENES).set_index(ids)
    preds_df = pd.DataFrame(rankings, columns=[str(x) for x in range(1, 24)]).set_index(ids)
    full_df = pd.concat([res_df, preds_df], axis=1)
    return full_df


def cakephp_formatted_predictions(combined_df: pd.DataFrame):
    # Format results for AudioGene site
    output_list = []
    for idx, row in combined_df.iterrows():
        id_str = f"{idx}\t"
        probs_str = ','.join([str(x) for x in row.iloc[:N_GENES].values])
        ranks_str = ','.join(row.iloc[N_GENES:].values) + '\t'
        output_list.append(id_str + ranks_str + probs_str)
    output_str = '\n'.join(output_list)
    return output_str


def write_results_react(file_path, df: pd.DataFrame):
    ext = os.path.splitext(file_path)[-1]
    if ext == '.csv':
        df.to_csv(file_path)
    elif ext == '.json':
        df.to_json(file_path, orient='index', indent=4)
    else:
        raise ValueError(f"Unexpected file extension {ext}. Must be .json or .csv.")


def write_results_php_site(file_path, probabilities, rankings, ids):
    results_df = combined_probs_and_rankings(probabilities, rankings, ids)
    output_str = cakephp_formatted_predictions(results_df)
    with open(file_path, "w") as f:
        f.write(output_str)


def write_results(file_path, probabilities: np.ndarray, ids, site_type: str, include_probabilities: bool = False):
    ranked = probabilities_to_predictions(probabilities)
    if site_type == 'cakephp':
        write_results_php_site(file_path, probabilities, ranked, ids)
    elif site_type == 'react':
        if include_probabilities:
            results_df = combined_probs_and_rankings(probabilities, ranked, ids)
        else:
            results_df = ranked
            results_df.index = ids
        write_results_react(file_path, results_df)
    else:
        raise ValueError(f"Unexpected site type {site_type}.")
