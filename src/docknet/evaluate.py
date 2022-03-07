import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn import metrics

from docknet.net import read_json, read_pickle


def parse_args():
    """
    Parse command-line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate a pre-trained '
                                                 'Docknet model with a test '
                                                 'set')
    parser.add_argument('--testset', '-t', action='store', required=True,
                        help='Dataset with which to evaluate the Docknet')
    parser.add_argument('--model', '-m', action='store', required=True,
                        help='Docknet model file in JSON or pickle format')

    args = parser.parse_args()

    if not os.path.exists(args.testset):
        print(f'Testset file {args.testset} doesn\'t exist')
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f'Model file {args.model_in} doesn\'t exist')
        sys.exit(1)

    args.model_type = os.path.splitext(args.model.lower())[1]
    if args.model_type not in ['.json', '.pkl']:
        print('Model file must be either a JSON or a pickle file')
        sys.exit(1)

    return args


def main():
    args = parse_args()
    testset = pd.read_csv(args.testset, header=None)
    X = testset.iloc[0:-1, :].values
    Y = testset.iloc[-1:, :].values

    if args.model_type == '.json':
        docknet = read_json(args.model)
    else:
        docknet = read_pickle(args.model)
    docknet.cost_function = 'cross_entropy'
    Y_predicted = docknet.predict(X)
    Y_predicted = np.round(Y_predicted)
    results = metrics.classification_report(Y[0], Y_predicted[0])
    print(results)


if __name__ == '__main__':
    main()
