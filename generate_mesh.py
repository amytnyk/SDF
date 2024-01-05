from argparse import ArgumentParser

import numpy as np
import torch
from torch import FloatTensor

from mcubes import marching_cubes
from trimesh import Trimesh

from sdf_network import SDFNetwork


def extract_fields(bound_min: FloatTensor, bound_max: FloatTensor, resolution: int, query_func):
    n = 16
    x_space = torch.linspace(bound_min[0], bound_max[0], resolution).split(n)
    y_space = torch.linspace(bound_min[1], bound_max[1], resolution).split(n)
    z_space = torch.linspace(bound_min[2], bound_max[2], resolution).split(n)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(x_space):
            for yi, ys in enumerate(y_space):
                for zi, zs in enumerate(z_space):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys),
                                                  len(zs)).detach().cpu().numpy()
                    u[xi * n: xi * n + len(xs), yi * n: yi * n + len(ys), zi * n: zi * n + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    vertices, triangles = marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def generate_mesh(model: SDFNetwork, resolution: int) -> Trimesh:
    def query_func(pts):
        pts = pts.to('cuda')
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                sdfs = model(pts)
        return sdfs

    bounds_min = torch.FloatTensor([-1, -1, -1])
    bounds_max = torch.FloatTensor([1, 1, 1])

    vertices, triangles = extract_geometry(bounds_min, bounds_max, resolution=resolution, threshold=0,
                                           query_func=query_func)

    return Trimesh(vertices, triangles, process=False)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output', type=str, required=True, help='Path to the output .ply file')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution')

    options = parser.parse_args()

    saved = torch.load(options.model, map_location='cuda')
    model = SDFNetwork().cuda()
    model.load_state_dict(saved['model'], strict=False)

    generate_mesh(model, options.resolution).export(options.output)


if __name__ == "__main__":
    main()
