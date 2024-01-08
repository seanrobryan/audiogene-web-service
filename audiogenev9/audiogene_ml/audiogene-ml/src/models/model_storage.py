import datetime
from joblib import dump, load
import os
from sklearn.base import ClassifierMixin

def save_model(ensemble, path: str = os.getcwd(), ensemble_name: str = 'AG-ensemble.joblib', add_time = False) -> str:
    if add_time:
        time = str(datetime.datetime.now())
        ensemble_path = os.path.join(path, '-'.join([time, ensemble_name]))
    else:
        ensemble_path = os.path.join(path, f"-{ensemble_name}")
        
    dump(ensemble, ensemble_path)
    return ensemble_path


def load_model(model_path: str) -> ClassifierMixin:
    return load(model_path)


def save_ensemble_params(ensemble: dict, path: str = os.getcwd(), ensemble_name = 'AG-ensemble-params.joblib', add_time = False):
    if add_time:
        time = str(datetime.datetime.now())
        ensemble_path = os.path.join(path, '-'.join([time, ensemble_name]))
    else:
        ensemble_path = os.path.join(path, f"-{ensemble_name}")
    
    ensemble_params = {}
    for name, clf in ensemble.items():
        ensemble_params[name] = clf.get_params()
    
    dump(ensemble_params, ensemble_path)
    return ensemble_path

def _load_ensemble_params(params_path: str):
    return load(params_path)

def load_ensemble_from_params(base_submodels: dict, params_path: str = None, params: dict = None):
    fit_models = {}
    if all([params_path, params]) == None:
        raise ValueError("params_path or params must be defined")
    elif params_path != None:
        params = _load_ensemble_params(params_path)

    try:
        for m_key, clf in base_submodels.items():
            print(m_key)
            fit_models[m_key] = clf.set_params(params[m_key])
    except Exception as e:
        print(e)
    
    return fit_models