import argparse
import warnings

from src.data import preprocessing, postprocessing
from src.models.model_storage import load_model

parser = argparse.ArgumentParser(description="CLI for AudioGene 9.1")
parser.add_argument('-i', '--input',
                    required=True,
                    help='Input file path',
                    type=str
                    )
parser.add_argument('-o', '--output',
                    help='Output file path',
                    type=str
                    )
parser.add_argument('-p', '--polys',
                    help='Degree of polynomial coefficients to be added seperated by commas',
                    nargs='*',
                    type=int
                    )
parser.add_argument('-m', '--model',
                    required=True)
parser.add_argument('--save_probabilities',
                    help='Save per class likelihood of each audiogram.',
                    action='store_true'
                    )
parser.add_argument('-w', '--warnings',
                    help='Parameter to supress warnings during development and debugging',
                    action='store_true'
                    )

parser.add_argument('-c', '--cakephp',
                    help='Write predictions formatted for the CakePHP site.',
                    action='store_true')


def predict(args: argparse.Namespace):
    if args.warnings is True:
        warnings.simplefilter('ignore')

    # TODO: Add audiogram label-age concat preprocessing
    polys = [ord for ord in args.polys] if args.polys is not None else None
    df = preprocessing.process(args.input, poly_orders=polys)

    model = load_model(args.model)

    # Get predictions and rankings
    probabilities = model.predict_proba(df.drop(columns=['id', 'ear']))

    # Format and return results
    if args.cakephp:
        postprocessing.write_results(args.output, probabilities, df.id, 'cakephp', args.save_probabilities)
    else:
        postprocessing.write_results(args.output, probabilities, df.id, 'react', args.save_probabilities)

    return args.output


if __name__ == "__main__":
    args = parser.parse_args()
    predict(args)
