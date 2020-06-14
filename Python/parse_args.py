"""Parse command line arguments"""

import argparse


def save_plots():
    """Parse args"""
    parser = argparse.ArgumentParser(description=(
        'CMC lab'
    ))
    parser.add_argument(
        '--save', '--save_figures', '-s',
        help='Save figures',
        dest='save',
        action='store_true'
    )
    args, _ = parser.parse_known_args()
    return args.save

