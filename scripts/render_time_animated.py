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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from renderer.gaussian_animated_renderer import render
import torchvision
import trimesh
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel


def transform_ficus_sinus(vertices, t, idxs):
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 0] * 2 *torch.pi + t)  # sinus
    vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)  # sinus
    return vertices


def transform_hotdog_fly(vertices, t, idxs):
    vertices_new = vertices.clone()
    f = torch.sin(t) * 0.5
    #vertices_new[:, 2] += f * vertices[:, 0] ** 2 # parabola
    #vertices_new[:, 2] += 0.3 * torch.sin(vertices[:, 0] * torch.pi + t)
    vertices_new[:, 2] += t * (vertices[:, 1] ** 2 + vertices[:, 1] ** 2) ** (1 / 2) * 0.01
    return vertices_new


def transform_ficus_pot(vertices, t, idxs):
    if t > 8 * torch.pi:
        vertices[idxs, 2] += 0.005 * torch.sin(vertices[idxs, 1] * 5 * torch.pi + t)
    else:
        vertices[idxs, 2] -= (0.005+t) * (vertices[idxs, 0]/10) ** 2
    return vertices


def transform_ship_sinus(vertices, t, idxs=None):
    f = torch.sin(t) * 0.5
    vertices[:, 2] += 0.05 * torch.sin(vertices[:, 0] * torch.pi + f) # sinus
    return vertices


def make_smaller(vertices, t, idxs=None):
    vertices_new = vertices.clone()
    f = torch.sin(t) + 1
    vertices_new = f * vertices_new
    return vertices_new


def do_not_transform(vertices, t):
    return vertices


def render_set(mesh_scene, model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "time_animated")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    t = torch.linspace(0, 10 * torch.pi, len(views))

    vertices = gaussians.vertices

    # chose indexes if you want change partly
    idxs = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        new_vertices = transform_hotdog_fly(vertices, t[idx], idxs)
        triangles = new_vertices[torch.tensor(gaussians.faces).long()].float().cuda()
        rendering = render(idxs, triangles, view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianMeshModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'update_alpha'):
            gaussians.update_alpha()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        mesh_scene = trimesh.load(f'{dataset.source_path}/mesh.obj', force='mesh')

        if not skip_train:
             render_set(mesh_scene, dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(mesh_scene, dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_mesh")
    parser.add_argument("--num_splats", nargs="+", type=int, default=[2])
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)