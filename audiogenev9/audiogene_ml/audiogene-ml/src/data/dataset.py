import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import add_all_features
from src.constants import POLYNOMIAL_COEFFICIENTS_LABELS


def make_dataset_from_csv(data_path: str, drop_coefs: bool = True, drop_patient_id: bool = True) -> pd.DataFrame:
    """Loads dataset at data_path and adds features."""
    df = pd.read_csv(data_path)
    df = make_dataset(df, drop_coefs, drop_patient_id)
    return add_all_features(df)


def split_dataset(df: pd.DataFrame, test_size=0.1, random_state=30, target: str = 'locus', 
                  set_names: list = ['Train', 'Test'], set_col = 'set') -> pd.DataFrame:
    """Applies sklearn's train_test_split and recombines the DataFrame."""
    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_state, stratify=y)

    # Combine into one w/ labels for set
    x_train[set_col] = set_names[0]
    x_test[set_col] = set_names[1]
    x_train[target] = y_train
    x_test[target] = y_test
    
    return x_train.append(x_test)


def make_dataset_from_df(df: pd.DataFrame, drop_coefs: bool = True, drop_patient_id: bool = True) -> pd.DataFrame:
    """Loads dataset at data_path and adds features."""
    df = make_dataset(df, drop_coefs, drop_patient_id)
    return add_all_features(df)


def remove_poly_coefficients(df: pd.DataFrame):
    return df.drop(columns=POLYNOMIAL_COEFFICIENTS_LABELS)


def make_dataset(df: pd.DataFrame, drop_coefs: bool = True, drop_patient_id: bool = True) -> pd.DataFrame:
    if drop_coefs:
        df = remove_poly_coefficients(df)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'id_num'})
    if drop_patient_id:
        df = df.drop(columns='id')
    return df
