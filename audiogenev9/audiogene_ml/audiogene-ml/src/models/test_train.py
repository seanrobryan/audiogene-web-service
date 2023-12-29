import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model_training import train_ensemble
# Add the following to src/constants.py
from src.features.build_features import add_all_features
GENES = ['ACTG1', 'CCDC50', 'CEACAM16', 'COCH', 'COL11A2', 'DIAPH1', 'EYA4',
         'GJB2', 'GRHL2', 'GSDME', 'KCNQ4', 'MIRN96', 'MYH14', 'MYH9', 'MYO6',
         'MYO7A', 'P2RX2', 'POU4F3', 'REST', 'SLC17A8', 'TECTA', 'TMC1', 'WFS1']
N_GENES = len(GENES)
GENOTYPE_LABELS = ['locus', 'gene']
FREQUENCIES = [125, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
FREQUENCY_LABELS = [f"{f} dB" for f in FREQUENCIES]
AUDIOGRAM_LABELS = ['age'] + FREQUENCY_LABELS
POLYNOMIAL_COEFFICIENTS_LABELS = [f"{o}c{d}" for o in (2, 3) for d in range(o)]

# Load your data
df = pd.read_csv('ag_9_full_dataset_processed_with_var_test.csv')

# Define your feature columns, target, and sampling thresholds
feature_cols = AUDIOGRAM_LABELS
target = 'gene'
sampling_thresholds = {gene: 150 for gene in GENES}

# Define your ensemble of models
ensemble = {
    'model1': RandomForestClassifier(),
    'model2': SVC()
}

# Define your search parameters for GridSearchCV
search_params = {
    'model1_params': {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
    'model2_params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}
df = add_all_features(df)

# Call the function
trained_models = train_ensemble(df, feature_cols, target, sampling_thresholds, ensemble, search_params)