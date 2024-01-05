import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from pysdf import SDF
from sklearn.metrics import f1_score
from trimesh import Trimesh

from sdf_network import SDFNetwork


@dataclass
class EvaluationResult:
    model_size: float
    quality_surface: float
    quality_volume: float
    time_per_batch: float
    time_per_point: float


class Evaluator:
    def __init__(self, model_path: Path, mesh_path: Path, points: int):
        saved = torch.load(model_path, map_location='cuda')

        self._model: SDFNetwork = SDFNetwork().cuda()
        self._model.load_state_dict(saved['model'], strict=False)

        self._mesh: Trimesh = self._scale_mesh(trimesh.load(mesh_path))

        self._sdf_fn: SDF = SDF(self._mesh.vertices, self._mesh.faces)

        self._points: int = points

    @staticmethod
    def _scale_mesh(mesh: Trimesh) -> Trimesh:
        v_min = mesh.vertices.min(0)
        v_max = mesh.vertices.max(0)
        v_center = (v_min + v_max) / 2
        v_scale = 2 / np.sqrt(np.sum((v_max - v_min) ** 2)) * 0.95

        mesh.vertices = (mesh.vertices - v_center[None, :]) * v_scale

        return mesh

    def _measure_model_size(self) -> float:
        return sum([param.nelement() * param.element_size() for param in self._model.parameters()] +
                   [buffer.nelement() * buffer.element_size() for buffer in self._model.buffers()])

    def evaluate(self) -> EvaluationResult:
        points_surface = self._mesh.sample(self._points).astype(np.float32)
        sample_surface = points_surface + np.random.normal(0, 1e-2, size=points_surface.shape).astype(np.float32)

        sample_volume = np.random.uniform(-1, 1, size=(self._points, 3)).astype(np.float32)

        cuda_surface = torch.from_numpy(sample_surface).cuda()
        cuda_volume = torch.from_numpy(sample_volume).cuda()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                sdfs_surface = self._model(cuda_surface)
                sdfs_volume = self._model(cuda_volume)

        random_points = [
            torch.from_numpy(np.random.uniform(-1, 1, size=(100000, 3)).astype(np.float32)).cuda()
            for _ in range(10)]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                torch.cuda.synchronize()
                start = time.time()
                for points in random_points:
                    self._model(points)

                torch.cuda.synchronize()
                end = time.time()

        return EvaluationResult(
            model_size=self._measure_model_size() / 1024 ** 2,
            quality_surface=f1_score(self._sdf_fn(sample_surface) > 0, sdfs_surface.detach().cpu().numpy() < 0,
                                     average='weighted'),
            quality_volume=f1_score(self._sdf_fn(sample_volume) > 0, sdfs_volume.detach().cpu().numpy() < 0,
                                    average='weighted'),
            time_per_batch=(end - start) * 1e3 / len(random_points),
            time_per_point=(end - start) * 1e9 / len(random_points) / len(random_points[0]),
        )


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--mesh', type=str, required=True, help='Path to the original object')
    parser.add_argument('--points', type=int, default=10000, help='Number of points')

    options = parser.parse_args()

    result = Evaluator(Path(options.model), Path(options.mesh), options.points).evaluate()

    print(f"Model size: {result.model_size:.3f} MB")
    print(f"Quality near the surface: {result.quality_surface:.4f}")
    print(f"Quality on the bounding volume: {result.quality_volume:.4f}")
    print(f"Time per batch: {result.time_per_batch:.4f} ms")
    print(f"Time per point: {result.time_per_point:.4f} ns")


if __name__ == "__main__":
    main()
