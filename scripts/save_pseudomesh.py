#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#
import sys
import os
import torch
from os import makedirs
from games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from argparse import ArgumentParser

class GaussiansLoader:

    gaussians : GaussianModel

    def __init__(self, model_path, gaussians : GaussianModel, load_iteration):
        """b
        :param path: Path to colmap loader main folder.
        """
        self.model_path = model_path
        self.gaussians = gaussians

        if load_iteration == -1:
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        else:
            self.loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.gaussians.load_ply(
            os.path.join(self.model_path,
            "point_cloud",
            "iteration_" + str(self.loaded_iter),
            "point_cloud.ply")
        )

        if hasattr(self.gaussians, 'prepare_vertices'):
            self.gaussians.prepare_vertices()
        if hasattr(self.gaussians, 'prepare_scaling_rot'):
            self.gaussians.prepare_scaling_rot()


def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    if verbose:
        print('mesh saved to: ', filepath)


def save_pseudomesh_info(
        sh_degree,
        model_path,
        iteration : int,
        scale = 1,
        save_faces: bool = False,
        save_vertices: bool = False
):
    with torch.no_grad():
        gaussians = PointsGaussianModel(sh_degree)
        model = GaussiansLoader(model_path, gaussians, load_iteration=iteration)

        pseudomesh_info_path = os.path.join(model_path, "pseudomesh_info", "ours_{}".format(model.loaded_iter))
        makedirs(pseudomesh_info_path, exist_ok=True)

        v1, v2, v3 = model.gaussians.v1, model.gaussians.v2, model.gaussians.v3
        triangles = torch.stack([v1, v2, v3], dim=1)
        torch.save(triangles, f'{pseudomesh_info_path}/triangles.pt')

        faces = torch.range(0, triangles.shape[0] * 3 - 1).reshape(triangles.shape[0], 3)
        vertices = triangles.reshape(triangles.shape[0] * 3, 3)

        if save_faces:
            torch.save(faces, f'{pseudomesh_info_path}/faces.pt')

        if save_vertices:
            torch.save(vertices, f'{pseudomesh_info_path}/vertices.pt')
        filename = f'{pseudomesh_info_path}/scale_{scale}.obj'
        write_simple_obj(mesh_v=(vertices * scale).detach().cpu().numpy(), mesh_f=faces, filepath=filename)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--scale", default=2, type=int)
    parser.add_argument("--save_faces", action="store_true")
    parser.add_argument("--save_vertices", action="store_true")

    args = parser.parse_args()


    print("Pseudomesh info " + args.model_path)

    model_path = args.model_path

    save_pseudomesh_info(
        args.sh_degree,
        args.model_path,
        args.iteration,
        args.scale,
        args.save_faces,
        args.save_vertices
    )