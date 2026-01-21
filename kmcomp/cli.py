"""Command-line interface for K-Means Compression Tool."""

import argparse
from pathlib import Path

from .model import main as model_main


def main():
    """
    This is the main function for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="K-Means Compression CLI Tool")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        type=Path,
        help="Path to the input image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=Path,
        nargs="?",
        const=True,
        default=None,
        help="Path to the output file.",
    )
    parser.add_argument(
        "-n",
        "--n-neighbors",
        dest="n_neighbors",
        type=int,
        default=12,
        help="Maximum number of clusters to use.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force the use of n_neighbors.",
    )
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save the compressed image."
    )
    args = parser.parse_args()

    model_main(
        input_path=args.input_path,
        output_path=args.output_path,
        n_neighbors=args.n_neighbors,
        force=args.force,
        save=args.save,
    )


if __name__ == "__main__":
    main()
