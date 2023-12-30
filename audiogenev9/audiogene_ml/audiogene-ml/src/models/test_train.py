import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from model_training import train_ensemble
# Add the following to src/constants.py
from src.features.build_features import add_all_features
from ensemble import _PartitionedEnsembleHelper

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
df = pd.read_csv('ag_9_full_dataset_processed_test_no_added_data.csv')

# Define your feature columns, target, and sampling thresholds
feature_cols = AUDIOGRAM_LABELS
target = 'gene'
sampling_thresholds = {gene: 150 for gene in GENES}


# Define your ensemble of models
models = [
    ('large', RandomForestClassifier, {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}),
    ('medium', SVC, {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    ('small', RandomForestClassifier, {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}),
]

search_params = {
    'size_params': {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
    'age_params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'shape_params': {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
}

# Call the function
ensemble = _PartitionedEnsembleHelper.construct_ensemble(models)
target_col = df[target]
df = df[feature_cols + [target]]
# df = add_all_features(df)
# Drop rows with NaN values
df.dropna(inplace=True)
print(df.shape)
# Check for NaN values
print(df.isna().sum())

df[target].value_counts()


from ensemble import PartitionedStackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Instantiate the PartitionedStackingClassifier with your ensemble
clf = PartitionedStackingClassifier(estimators=ensemble)

# Define the StratifiedKFold cross-validator
cv = StratifiedKFold(n_splits=3, shuffle=True)
# get only 10% of the data
df = df.sample(frac=0.01)
print(df.shape)

# Perform cross-validation on your data
for train_index, test_index in cv.split(df[feature_cols], df[target]):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = df[feature_cols].iloc[train_index], df[feature_cols].iloc[test_index]
    y_train, y_test = df[target].iloc[train_index], df[target].iloc[test_index]
    print(y_train.value_counts())
    print(y_test.value_counts())
    print("________________________")
    print(X_train)
    print(y_train)
    print("________________________")
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(clf.predict(X_test))

# df_copy = df.copy()
# df_copy = df_copy.drop(columns='gene')
# # Perform cross-validation on your data
# skf = StratifiedKFold(n_splits=3)
# for train_index, test_index in skf.split(df_copy[feature_cols], target_col):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = df_copy.iloc[train_index], df_copy.iloc[test_index]
#     y_train, y_test = target_col.iloc[train_index], target_col.iloc[test_index]
#     # make sure that all the arrays have a shape of (n_samples, 11)
#     print(X_train.shape, y_train.shape)
#     print(X_test.shape, y_test.shape)
#     if X_train.shape[1] != 11 or X_test.shape[0] == 0:
#         print("ERROR")
#     clf.fit(X_train, y_train)
#     print(clf.score(X_test, y_test))
#     print(clf.predict(X_test))

# Print the cross-validation scores