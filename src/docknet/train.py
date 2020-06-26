import argparse
import os
import sys

import pandas as pd

from docknet.net import read_json, read_pickle
from docknet.initializer.random_normal_initializer import RandomNormalInitializer
from docknet.optimizer.adam_optimizer import AdamOptimizer
from docknet.optimizer.gradient_descent_optimizer import GradientDescentOptimizer


def parse_args():
    """
    Parse command-line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train Docknet")
    parser.add_argument('--trainingset', '-t', action='store', required=True,
                        help="Dataset with which to train the Docknet")
    parser.add_argument('--model-in', '-mi', action='store', required=True,
                        help="Initial model file in json or pickle format describing the docknet layers and, optionally"
                             ", the initial parameters")
    parser.add_argument('--model-out', '-mo', action='store', required=True,
                        help="Model file (json or pkl) where to write the resulting docknet model after training")
    parser.add_argument('--initializer', '-i', action='store', default=None,
                        help="Initializer to use (do not specify, for no initialization, or specify random)")
    parser.add_argument('--optimizer', '-o', action='store', default='adam',
                        help="Optimizer to use (defaults to adam)")
    parser.add_argument('--batch-size', '-b', action='store', required=True, type=int,
                        help="Amount of training examples to use per training iteration")
    parser.add_argument('--max-number-of-epochs', '-e', action='store', required=True, type=int,
                        help="Maximum number of epochs to train")
    parser.add_argument('--error-delta', '-d', action='store', default=0.0, type=float,
                        help="Error increment threshold (defaults to 0, meaning there is no threshold)")
    parser.add_argument('--max-epochs-within-delta', '-ed', action='store', default=-1, type=int,
                        help="Maximum number of epochs with error increment within the specified threshold (defaults to"
                             "-1, meaning there is no maximum")
    parser.add_argument('--stop-file', '-s', action='store', default=None,
                        help="File whose existence will end the training process (defaults to none); at any time during"
                             "training, create this file (e.g. with command touch) so no more epochs are run and the"
                             "model is saved")

    args = parser.parse_args()
    if args.stop_file and os.path.exists(args.stop_file):
        print("Stop file detected; delete it before starting the training")
        sys.exit(1)

    if not os.path.exists(args.trainingset):
        print("Trainingset file {} doesn't exist".format(args.trainingset))
        sys.exit(1)

    if not os.path.exists(args.model_in):
        print("Initial model file {} doesn't exist".format(args.model_in))
        sys.exit(1)

    args.model_in_type = os.path.splitext(args.model_in.lower())[1]
    if not args.model_in_type in ['.json', '.pkl']:
        print("Initial model file must be either a json or a pkl file")
        sys.exit(1)

    args.model_out_type = os.path.splitext(args.model_out.lower())[1]
    if not args.model_out_type in ['.json', '.pkl']:
        print("Output model file must be either a json or a pkl file")
        sys.exit(1)

    return args


def main():
    args = parse_args()
    trainingset = pd.read_csv(args.trainingset, header=None)
    X = trainingset.iloc[0:-1, :].values
    Y = trainingset.iloc[-1:, :].values

    if args.model_in_type == '.json':
        docknet = read_json(args.model_in)
    else:
        docknet = read_pickle(args.model_in)
    initialize = False
    if args.initializer:
        initialize = True
        if args.initializer == 'random':
            docknet.initializer = RandomNormalInitializer()
        else:
            print('Unknown initializer {}; available initializers: random'.format(args.initializer))
    if args.optimizer == 'adam':
        docknet.optimizer = AdamOptimizer()
    elif args.optimizer == 'gradient_descent':
        docknet.optimizer = GradientDescentOptimizer()
    else:
        print('Unknown optimizer {}; available optimizers: adam, gradient_descent'.format(args.optimizer))
        sys.exit(1)
    docknet.cost_function = 'cross_entropy'
    docknet.train(X, Y, args.batch_size, max_number_of_epochs=args.max_number_of_epochs, error_delta=args.error_delta,
                  max_epochs_within_delta=args.max_epochs_within_delta, stop_file_pathname=args.stop_file,
                  initialize=initialize)
    if args.model_out_type == '.json':
        docknet.to_json(args.model_out)
    else:
        docknet.to_pickle(args.model_out)


if __name__ == '__main__':
    main()
