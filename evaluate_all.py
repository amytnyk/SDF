import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Generator, Any

from pandas import DataFrame
from tqdm import tqdm

from evaluate import Evaluator


def iterate_all_trained_models(path: str) -> Generator[str, Any, None]:
    for entry in os.listdir(path):
        entry_path = Path(path) / entry
        if os.path.isdir(entry_path):
            yield entry


def main():
    parser = ArgumentParser()
    parser.add_argument('--models', type=str, required=True, help='Path to the trained models')
    parser.add_argument('--meshes', type=str, required=True, help='Path to the original objects')
    parser.add_argument('--output', type=str, required=True, help='Output csv path')
    parser.add_argument('--points', type=int, default=10000, help='Number of points')

    options = parser.parse_args()
    models = list(iterate_all_trained_models(options.models))

    results = {
        int(model): Evaluator(Path(options.models) / model / "best.pth", Path(options.meshes) / f"{model}.obj",
                              options.points).evaluate()
        for model in tqdm(models)
    }
    result_array = []
    for model, result in results.items():
        result_array.append([model, result.model_size, result.quality_surface, result.quality_volume,
                             result.time_per_batch, result.time_per_point])
    df = DataFrame(sorted(result_array))
    df.columns = ["Model number", "Model size (MB)", "Quality (surface)", "Quality (volume)",
                  "Time per 100k batch (ms)", "Time per point (ns)"]
    df.to_csv(options.output, index=False, float_format='%.4f')

    print(f"Model size: {df['Model size (MB)'].mean():.4f} MB")
    print(f"Mean quality near surface: {df['Quality (surface)'].mean():.4f}")
    print(f"Mean quality on the bounding volume: {df['Quality (volume)'].mean():.4f}")
    print(f"Mean time per 100k batch: {df['Time per 100k batch (ms)'].mean():.4f} ms")
    print(f"Mean time per point: {df['Time per point (ns)'].mean():.4f} ns")


if __name__ == "__main__":
    main()
