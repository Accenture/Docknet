import argparse
import sys

import pandas as pd

from docknet.data_generator.data_generator_factory import (data_generators,
                                                           make_data_generator)


def parse_args():
    """
    Parse command-line arguments
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate a random dataset')
    parser.add_argument('--generator', '-g', action='store', required=True,
                        help=f'Data generator to use '
                             f'({",".join(data_generators.keys())})')
    parser.add_argument('--x0_min', action='store', default=-5.0, type=float,
                        help='Minimum value of x0')
    parser.add_argument('--x0_max', action='store', default=5.0, type=float,
                        help='Maximum value of x0')
    parser.add_argument('--x1_min', action='store', default=-5.0, type=float,
                        help='Minimum value of x1')
    parser.add_argument('--x1_max', action='store', default=5.0, type=float,
                        help='Maximum value of x1')
    parser.add_argument('--size', '-s', action='store', required=True,
                        type=int, help='Sample size')
    parser.add_argument('--output', '-o', action='store', default=None,
                        help='Output path (defaults to standard output)')

    args = parser.parse_args()
    if args.generator not in data_generators.keys():
        print(f'Unknown data generator {args.generator}; available generators '
              f'are: {",".join(data_generators.keys())}')
        sys.exit(1)
    if args.x0_min >= args.x0_max:
        print('Empty x0 range')
        sys.exit(1)
    if args.x1_min >= args.x1_max:
        print('Empty x1 range')
        sys.exit(1)
    return args


def main():
    args = parse_args()
    generator = make_data_generator(args.generator, (args.x0_min, args.x0_max),
                                    (args.x1_min, args.x1_max))
    X, Y = generator.generate_balanced_shuffled_sample(args.size)
    X_df = pd.DataFrame(X)
    Y_df = pd.DataFrame(Y)
    sample_df = pd.concat([X_df, Y_df], axis=0, ignore_index=True)
    if args.output:
        with open(args.output, 'wt', encoding='UTF-8') as fp:
            sample_df.to_csv(fp, header=False, index=False)
    else:
        sample_df.to_csv(sys.stdout, header=False, index_label=False)


if __name__ == '__main__':
    main()
