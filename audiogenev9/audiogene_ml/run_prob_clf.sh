#!/bin/bash

INPUT_FILE="$PWD/$1"
OUTPUT_FILE="$PWD/$2"

VENV="$PWD/$3activate"
PYTHON="$PWD/$3python"
AG_CLF="$PWD/$4"
MODEL="$PWD/audiogene-ml/notebooks/saved_models/cur_model.joblib"

touch $OUTPUT_FILE
result=$(source $VENV && $PYTHON $AG_CLF -i $INPUT_FILE -o $OUTPUT_FILE -m $MODEL -w)
echo $result
