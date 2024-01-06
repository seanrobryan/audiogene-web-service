from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import Iterable, NamedTuple, List, Union
import xlrd
import openpyxl
import os
from itertools import islice

FREQS = ['125 Hz', '250 Hz', '500 Hz', '1000 Hz', '1500 Hz', '2000 Hz', '3000 Hz', '4000 Hz', '6000 Hz', '8000 Hz']
FREQ_INTS = [int(f.split(' ')[0]) for f in FREQS]
COLS = ['id', 'age', 'ear'] + FREQS

Numeric = Union[int , float]

@dataclass
class _Audiogram():
    id: str
    age: Numeric
    ear: str
    hearing_loss: List[Numeric]

class _Point(NamedTuple):
    x: Numeric
    y: Numeric

def process(filepath: str, poly_orders: Iterable = (2,3), save_to: str = None) -> pd.DataFrame:
    """Process the data from the template format specified at audiogene.eng.uiowa.edu to a DataFrame
    compatible with the AudioGene predictor.

    Args:
        filepath (str): Location of data.
        poly_orders (Iterable, optional): Order of polynomial coefficients to add. Defaults to (2,3).
        save_to (str, optional): The location for the data to be saved to as a csv file. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """      
    # Extract audiograms from workbook
    if os.path.splitext(filepath)[-1] == '.xls':
        wb = xlrd.open_workbook_xls(filepath)
        audiograms = extract_audiograms_xls(wb)
    else:
        wb = openpyxl.load_workbook(filename=filepath)
        audiograms = extract_audiograms_xlsx(wb)

    
    # Put audiograms into DataFrame and replace missing values
    audio_df = pd.DataFrame(audiograms)
    audio_df = pd.concat([audio_df.drop(columns=['hearing_loss']), pd.DataFrame(audio_df['hearing_loss'].to_list(), columns=FREQS)], axis=1)
    audio_df = audio_df.replace(to_replace='', value=np.NaN)
        
    # Apply the best hearing rule and add polynomial coefficients
    audio_df = apply_best_hearing(audio_df)
    audio_df = fill_in_values(audio_df)
    if poly_orders is not None:
        audio_df = add_polynomial_coeffs(audio_df, poly_orders)
    
    # TODO: It would be better to just change the training data column headers.
    # Change Hz headers to dB
    audio_df = hz_cols_to_db(audio_df)
    
    if save_to is not None:
        audio_df.to_csv(save_to, sep=',')
    return audio_df

# TODO: Refactor these functions to reduce code redundancy
def extract_audiograms_xls(wb: xlrd.book.Book) -> List[_Audiogram]:
    """Extract audiograms from Excel workbook following the template specified at:
    https://audiogene.eng.uiowa.edu/analyses.

    Args:
        wb (xlrd.book.Book): .xls workbook

    Returns:
        List[_Audiogram]: List of audiograms found on the 'Audio' sheet
    """    
    ws = wb.sheet_by_name('Audio')
    audiograms = []
    for r in range(2, ws.nrows):
        i = ws.cell_value(r, 0)
        if i != '':
            age = ws.cell_value(r, 1)
            ear = ws.cell_value(r, 2)
            measurements = []
            for c in range(3, 13):
                measurements.append(ws.cell_value(r,c))
            audiogram = _Audiogram(id=i, age=age, ear=ear, hearing_loss=measurements)
            audiograms.append(audiogram)
    return audiograms


def extract_audiograms_xlsx(wb: openpyxl.workbook.Workbook) -> List[_Audiogram]:
    """Extract audiograms from xlsx formatted Excel workbook following the template specified at:
    https://audiogene.eng.uiowa.edu/analyses.

    Args:
        wb (openpyxl.workbook.Workbook): .xlsx workbook

    Returns:
        List[_Audiogram]: List of audiograms found on the 'Audio' sheet
    """
    ws = wb.get_sheet_by_name('Audio')
    audiograms = []
    for r in list(islice(ws.iter_rows(), 2, None)):
        if r[0].value != '':
            audiograms.append(_Audiogram(
                id=r[0].value,
                age=r[1].value,
                ear=r[2].value,
                hearing_loss=[c.value for c in r[3:]]))
    return audiograms


def apply_best_hearing(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the 'best hearing' rule to all the sets of unique id-age pairings found in the audiograms of df.

    Args:
        df (pd.DataFrame): Audiograms with the columns 
            ['id', 'age', 'ear', '125 Hz', '250 Hz', '500 Hz', '1000 Hz', '1500 Hz', '2000 Hz', '3000 Hz', '4000 Hz', '6000 Hz', '8000 Hz']

    Returns:
        pd.DataFrame: Dataframe with the best hearing audiograms
    """
    df = df.copy()
    df = df.groupby(by=['id', 'age']).apply(np.min).drop(columns=['id', 'age']).reset_index()
    df['ear'] = 'Better'
    df = df.fillna(value=np.NaN)
    return df


def fill_in_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply linear interpolation and extrapolation to fill in missing values.

    Args:
        df (pd.DataFrame): Dataframe of audiograms with missing values (as np.NaN).

    Returns:
        pd.DataFrame: Dataframe with interpolation and extrapolation applied.
    """
    def _interpolate_points(x: Numeric, p1: _Point, p2: _Point) -> float:
        """Linearly interpolate the value at x from p1 and p2.

        Args:
            x (Numeric): int or float value to interpolate at.
            p1 (_Point): namedtuple with an x and y Numeric
            p2 (_Point): numedtuple with an x and y Numeric

        Returns:
            float: Value of y at x.
        """        
        y = p1.y + ((x - p1.x) * (p2.y - p1.y)) / (p2.x - p1.x)
        if y > 120:
            y = 120
        elif y < 0:
            y = 0
        return y
        
    # Prep data
    df = df.copy()
    freq_cols = [c for c in df.columns if 'Hz' in c]
    freqs_df = df.loc[:, FREQS[0]:]
    freqs_df.columns = [int(c.replace(' Hz', '')) for c in freq_cols]
    
    first_col = freqs_df.iloc[:, 0]
    last_col = freqs_df.iloc[:, -1]
    
    # Transpose to make interpolation simplier
    freqs_df = freqs_df.T
    
    # Interpolate values
    freqs_df = freqs_df.interpolate(method='values', axis='index').T
    
    # Extrapolate the first and last columns
    freqs_df.iloc[:, 0] = first_col
    freqs_df.iloc[:, -1] = last_col
    
    # Over all the rows
    for idx, row in freqs_df.iterrows():
        # If column 0 is nan, extrapolate
        if np.isnan(row.iloc[0]):
            x = row.index[0]
            pt1 = _Point(row.index[1], row.iloc[1])
            pt2 = _Point(row.index[2], row.iloc[2])
            row.iloc[0] = _interpolate_points(x, pt1, pt2)
        # If column -1 if nan, extrapolate
        if np.isnan(row.iloc[-1]):
            x = row.index[-1]
            pt1 = _Point(row.index[-2], row.iloc[-2])
            pt2 = _Point(row.index[-3], row.iloc[-3])
            row.iloc[-1] = _interpolate_points(x, pt1, pt2)
        if any(np.isnan(row)):
            np.nan_to_num(row, copy=False, nan=0)
    freqs_df.iloc[idx, :] = row # Replace the column
    freqs_df.columns = freq_cols # Replace int headers with string headers
    df.loc[:, FREQS[0]:] = freqs_df.loc[:, :] # Insert the interpolated data inplace of the old data
    return df

def add_polynomial_coeffs(audiograms: pd.DataFrame, order: Iterable[int]) -> pd.DataFrame:
    loss_vals = audiograms.iloc[:, 3:].values
    all_polys = []
    labels = []
    
    # Make labels based on order(s)
    for ord in order:
        labels.extend([f"{ord}c{c}" for c in range(ord+1)])
    
    # Fit audiograms to polynomials
    for r in range(loss_vals.shape[0]):
        cur_polys = []
        for ord in order:
            cur_polys.extend(np.polyfit(x=FREQ_INTS, y=loss_vals[r], deg=ord))
        all_polys.append(cur_polys)
    
    poly_df = pd.DataFrame(all_polys, columns=labels) # Make polynomial DataFrame
    # Concatenate polynomial coefficients to original DataFrame
    res = pd.concat([
        audiograms,
        poly_df
    ], axis=1)
    
    return res

def hz_cols_to_db(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    df.columns = [f"{c.split(' ')[0]} dB" if 'Hz' in c else c for c in cols]
    return df

def adjust_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    ids = df.id
    if len(ids.unique()) == len(ids):
        return df
    else:
        for idx, id in ids.iteritems():
            age = int(df.loc[idx, 'age'])
            df.loc[idx, 'id'] = f"{id}-{age}y"
        return df
