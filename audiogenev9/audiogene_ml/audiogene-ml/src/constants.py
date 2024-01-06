import os

GENES = ['ACTG1', 'CCDC50', 'CEACAM16', 'COCH', 'COL11A2', 'DIAPH1', 'EYA4',
         'GJB2', 'GRHL2', 'GSDME', 'KCNQ4', 'MIRN96', 'MYH14', 'MYH9', 'MYO6',
         'MYO7A', 'P2RX2', 'POU4F3', 'REST', 'SLC17A8', 'TECTA', 'TMC1', 'WFS1']
N_GENES = len(GENES)
GENOTYPE_LABELS = ['locus', 'gene']
FREQUENCIES = [125, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
FREQUENCY_LABELS = [f"{f} dB" for f in FREQUENCIES]
AUDIOGRAM_LABELS = ['age'] + FREQUENCY_LABELS
POLYNOMIAL_COEFFICIENTS_LABELS = [f"{o}c{d}" for o in (2, 3) for d in range(o)]

# Partition sorting defaults
PARTITION_PARAMETERS = dict(
    bin_names=['small', 'medium', 'large'],
    bin_thresholds=[0, 20, 300],
    freq_groups=[3, 7, 10])
SHAPE_THRESHOLDS = (10, 5)

# Constants related to APS fitting
MINIMUM_AUDIOGRAMS = 10
MINIMUM_AGE = 0
MAXIMUM_AGE = 100
MINIMUM_DB_LOSS = 0
MAXIMUM_DB_LOSS = 130

kHZ_FREQUENCIES = [f/1000 for f in FREQUENCIES]
kHZ_FREQUENCY_LABELS = [f"{f} kHz" for f in kHZ_FREQUENCIES]
PSEUDO_LOG_FREQUENCIES = [1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7]

HOME = "audiogene-ml"
DATA_DIR = "/audiogene-ml/data"
