from argparse import ArgumentParser
from pathlib import Path

from train import train


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the input objects directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--batch', type=int, default=2 ** 18, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    options = parser.parse_args()

    objects = list(Path(options.input).glob('*'))
    for idx, object_path in enumerate(objects):
        train(str(object_path), f"{options.output}/{object_path.stem}", options.batch, options.epochs, options.lr,
              f"{idx + 1}/{len(objects)}")


if __name__ == "__main__":
    main()
