from argparse import ArgumentParser
from pathlib import Path

from torch.utils.data import DataLoader

from obj_dataset import ObjDataset
from sdf_network import SDFNetwork
from sdf_trainer import SDFTrainer


def train(input_path: str, output: str, batch: int, epochs: int, lr: float, description: str = ""):
    train_dataset = ObjDataset(input_path, size=100, batch_size=batch)
    train_data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    validation_dataset = ObjDataset(input_path, size=1, batch_size=batch)
    validation_data_loader = DataLoader(validation_dataset, batch_size=1)

    sdf_trainer = SDFTrainer(SDFNetwork().cuda(), train_data_loader, validation_data_loader, lr, epochs, Path(output))
    sdf_trainer.train(description)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input object')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--batch', type=int, default=2 ** 18, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    options = parser.parse_args()
    train(options.input, options.output, options.batch, options.epochs, options.lr)


if __name__ == "__main__":
    main()
