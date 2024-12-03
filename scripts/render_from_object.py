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
from renderer.gaussian_points_animated_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
import trimesh


def render_set(model_path, object_path, name, iteration, views, gaussians, pipeline, background, scale):
    basename = os.path.basename(object_path).split('.')[0]
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), basename)

    makedirs(render_path, exist_ok=True)
    mesh_scene = trimesh.load(object_path, force='mesh')
    triangles = torch.tensor(mesh_scene.triangles).cuda().float() / scale
    triangles[:,:,0] -= 0.2

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(triangles, view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + f".png"))


def render_sets(
        object_path, dataset : ModelParams, iteration : int,
        pipeline : PipelineParams, skip_train : bool, skip_test : bool, scale
):
    with torch.no_grad():
        gaussians = PointsGaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if hasattr(gaussians, 'prepare_vertices'):
            gaussians.prepare_vertices()
        if hasattr(gaussians, 'prepare_scaling_rot'):
            gaussians.prepare_scaling_rot()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(
                 dataset.model_path, object_path, "train",
                 scene.loaded_iter, scene.getTrainCameras(),
                 gaussians, pipeline, background, scale
             )

        if not skip_test:
             render_set(
                 dataset.model_path, object_path, "test",
                 scene.loaded_iter, scene.getTestCameras(),
                 gaussians, pipeline, background, scale
             )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--gs_type', type=str, default="gs_points")
    parser.add_argument("--scale", default=2, type=float)
    parser.add_argument("--object_path", default="", type=str)


    args = get_combined_args(parser)
    model.gs_type = args.gs_type

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        args.object_path, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.scale
    )