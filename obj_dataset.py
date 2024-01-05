import numpy as np

from torch.utils.data import Dataset

import trimesh
import pysdf


class ObjDataset(Dataset):
    def __init__(self, path: str, size: int = 100, batch_size: int = 2 ** 18):
        super().__init__()

        self._mesh = trimesh.load(path, force='mesh')

        v_min = self._mesh.vertices.min(0)
        v_max = self._mesh.vertices.max(0)
        v_center = (v_min + v_max) / 2
        v_scale = 2 / np.sqrt(np.sum((v_max - v_min) ** 2)) * 0.95

        self._mesh.vertices = (self._mesh.vertices - v_center[None, :]) * v_scale

        self._sdf_fn = pysdf.SDF(self._mesh.vertices, self._mesh.faces)

        self._batch_size = batch_size
        assert self._batch_size % 8 == 0, "batch size must be divisible by 8."

        self._size = size

    def __len__(self):
        return self._size

    def __getitem__(self, _):
        # online sampling
        sdfs = np.zeros((self._batch_size, 1))
        # surface
        points_surface = self._mesh.sample(self._batch_size * 7 // 8)
        # perturb surface
        points_surface[self._batch_size // 2:] += 0.01 * np.random.randn(self._batch_size * 3 // 8, 3)
        # random
        points_uniform = np.random.rand(self._batch_size // 8, 3) * 2 - 1
        points = np.concatenate([points_surface, points_uniform], axis=0).astype(np.float32)

        sdfs[self._batch_size // 2:] = -self._sdf_fn(points[self._batch_size // 2:])[:, None].astype(np.float32)

        return {
            'sdfs': sdfs,
            'points': points,
        }
