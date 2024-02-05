#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import numpy as np
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from flame_gaussian_renderer import flame_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from flame_splatting.scene.gaussian_flame_model import GaussianFlameModel
from skimage.segmentation import flood, flood_fill

def _render_set(
        gaussians, shape_params, expression_params, pose_params, neck_pose,
        transl, iteration, views, pipeline, background, render_path, gts_path,
        filename_vertices_save = "flame_render_vertices"
):
    vertices, _ = gaussians.point_cloud.flame_model(
        shape_params=shape_params,
        expression_params=expression_params,
        pose_params=pose_params,
        neck_pose=neck_pose,
        transl=transl
    )
    vertices = gaussians.point_cloud.transform_vertices_function(
        vertices,
        gaussians._vertices_enlargement
    )

    torch.save(vertices, f'{render_path}/{iteration}_{filename_vertices_save}.pt')
    torch.save(gaussians.faces, f'{render_path}/{iteration}_flame_render_faces.pt')

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = flame_render(
            view, gaussians, pipeline, background, vertices=vertices
        )["render"]
        #img = rendering.cpu().numpy()
        #rendering = flood_fill(img, (0, 0, 0), 1, tolerance = 0.05)
        if gts_path is not None:
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        #rendering = torch.tensor(rendering)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    """
    _render_set(
        gaussians=gaussians,
        shape_params=gaussians._flame_shape,
        expression_params=gaussians._flame_exp,
        pose_params=gaussians._flame_pose,
        neck_pose=gaussians._flame_neck_pose,
        transl=gaussians._flame_trans,
        iteration=iteration,
        views=views,
        pipeline=pipeline,
        render_path=render_path,
        gts_path=gts_path,
        background=background
    )
    """

    name_file = "for_mesh_4"
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), name_file)
    makedirs(render_path, exist_ok=True)

    radian = np.pi / 180.0
    pose_rot = gaussians._flame_pose.clone().detach()
    #pose_rot[0,3] = 10  * radian
    _flame_exp = gaussians._flame_exp.clone().detach()

    #_flame_exp[0, 0] = 2
    #_flame_exp[0, 0] = -3
    #_flame_exp[0, 1] =0
    #_flame_exp[0, 2] = 0
    #_flame_exp[0, 3] = 5
    #_flame_exp[0, 4] = 0
    #_flame_exp[0, 5] = 0
    #_flame_exp[0, 6] = 0
    _flame_exp[0, 7] = 3
    _flame_exp[0, 9] = -3
    _flame_exp[0, 8] = -3
    _flame_exp[0, 10] = 3



    _render_set(
        gaussians=gaussians,
        shape_params=gaussians._flame_shape,
        expression_params=_flame_exp,
        pose_params=pose_rot,
        neck_pose=gaussians._flame_neck_pose,
        transl=gaussians._flame_trans,
        iteration=iteration,
        views=views,
        pipeline=pipeline,
        background=background,
        render_path=render_path,
        gts_path=None,
        filename_vertices_save=name_file
    )


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianFlameModel(dataset.sh_degree)
        s = ('/').join(dataset.source_path.split('/')[-2:])
        dataset.source_path = s
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #if not skip_train:
        #     render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)