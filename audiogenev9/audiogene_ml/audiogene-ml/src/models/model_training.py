import pandas as pd
import numpy as np
from icecream import ic
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, make_scorer, roc_auc_score, top_k_accuracy_score
from utils import filter, sample

def train_ensemble(df: pd.DataFrame, feature_cols: list, target: str, sampling_thresholds: dict,
                   ensemble: dict, search_params: dict, k_folds = 10,
                   n_jobs = 5, scoring = ['accuracy'], scoring_params = {},
                   verbosity=1, score_func = None, refit = 'accuracy'):
    
#     if score_func is not None:
#         scoring = {"Custom Scorer": make_scorer(score_func = scoring, **scoring_params, needs_proba = True)}
    
    # scoring = {"Accuracy": make_scorer(accuracy_score), "AUC": "roc_auc", }
    
    model_types = [('size', 'age_group'), ('age', 'instance_group'), ('shape', 'shape')]
    trained_models = {}
    for type_ in model_types:
        print(f"Training {type_[0].upper()} models")

        model_keys = sorted([key for key in ensemble.keys() if type_[0] in key.split('_')[0]])
        
        groups = sorted(df[type_[1]].unique())
        
        if type_[0] == 'shape':
            df = df.copy()
            df = sample.resampling(df, sampling_thresholds)
            df = df.reset_index().drop(columns='index')
            df['id_num'] = df.index
            
            groups = [group for group in groups for _ in range(2)]
        
        keys_to_cat = dict((v, k) for v, k in zip(model_keys, groups))
        
        for mk in model_keys:
            cur_clf = ensemble[mk]
            
            print(f'\n{mk}\n')
            
            gs_cv_clf = GridSearchCV(
                estimator = cur_clf,
                param_grid = search_params[f"{type_[0]}_params"],
                cv = k_folds,
                n_jobs = n_jobs,
                # scoring = scoring(**scoring_params) if scoring_params else scoring,
                scoring = scoring,
                verbose = verbosity,
                refit = refit
            )

            if type_[0] == 'age':
                age = mk.split('_')[1]
                filtered_df = filter.filter_ages(df, age)
            elif type_[0] == 'size':
                size = mk.split('_')[1]
                filtered_df = filter.filter_instance_groups(df, size)
            else:
                filtered_df = filter.filter_shapes(df, keys_to_cat[mk])
                
            
            filtered_x_train = filtered_df.loc[filtered_df['set'] == 'Train', feature_cols]
            filtered_y_train = np.squeeze(filtered_df.loc[filtered_df['set'] == 'Train', target])
            gs_cv_clf.fit(filtered_x_train, filtered_y_train)
            
            best_params = gs_cv_clf.fit(filtered_x_train, filtered_y_train).best_params_
            
            cur_clf = cur_clf.set_params(**best_params)
            
            filtered_x_test = filtered_df.loc[filtered_df['set'] == 'Test', feature_cols]
            filtered_y_test = np.squeeze(filtered_df.loc[filtered_df['set'] == 'Test', target])
            
            cur_clf.fit(filtered_x_train[feature_cols], filtered_y_train)
            
            trained_models[mk] = cur_clf

    return trained_models

