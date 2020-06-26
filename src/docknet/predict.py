import argparse
import os
import sys

import pandas as pd

from docknet.net import read_json, read_pickle


def parse_args():
    """
    Parse command-line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train Docknet")
    parser.add_argument('--dataset', '-d', action='store', required=True,
                        help="Dataset with which to predict")
    parser.add_argument('--ignore_last_row', '-i', action='store_true',
                        help="Ignore last dataset row; useful when the dataset labels are included in the last row")
    parser.add_argument('--model', '-m', action='store', required=True,
                        help="Docknet model file in json or pickle format")
    parser.add_argument('--output', '-o', action='store', default=None,
                        help="Output path (defaults to standard output)")

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print("Dataset file {} doesn't exist".format(args.dataset))
        sys.exit(1)

    if not os.path.exists(args.model):
        print("Model file {} doesn't exist".format(args.model_in))
        sys.exit(1)

    args.model_type = os.path.splitext(args.model.lower())[1]
    if not args.model_type in ['.json', '.pkl']:
        print("Model file must be either a json or a pkl file")
        sys.exit(1)

    return args


def main():
    args = parse_args()
    X = pd.read_csv(args.dataset, header=None)
    if args.ignore_last_row:
        X = X.iloc[0:-1, :]
    X = X.values

    if args.model_type == '.json':
        docknet = read_json(args.model)
    else:
        docknet = read_pickle(args.model)
    docknet.cost_function = 'cross_entropy'

    Y = docknet.predict(X)
    Y_df = pd.DataFrame(Y)
    if args.output:
        with open(args.output, 'wt', encoding='UTF-8') as fp:
            Y_df.to_csv(fp, header=False, index=False)
    else:
        Y_df.to_csv(sys.stdout, header=False, index_label=False)


if __name__ == '__main__':
    main()
