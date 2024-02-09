#
# This software is based on renders.py file free for non-commercial, research and evaluation use
# from https://github.com/graphdeco-inria/gaussian-splatting/blob/main/render.py
#
# Hence, This software is also free for non-commercial, research and evaluation use.
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_points_animated_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from flat_splatting.scene.points_gaussian_model import PointsGaussianModel


def transform_hotdog_fly(triangles, t):
    triangles_new = triangles.clone()
    #vertices_new[:, 2] += 0.3 * torch.sin(vertices[:, 0] * torch.pi + t)
    #triangles_new[:, :, 2] += t * (triangles[:, :, 1] ** 2 + triangles[:, :, 1] ** 2) ** (1 / 2) * 0.01
    #triangles_new[:, :, 2] += 0.3 * torch.sin(triangles[:, :,  0] * torch.pi + t)
    triangles_new[:, :, 2] += 0.2 * triangles_new[:, :, 0]
    return triangles_new


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "time_animated_games_02_to_2")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    t = torch.linspace(0, 10 * torch.pi, len(views))
    v1, v2, v3 = gaussians.v1, gaussians.v2, gaussians.v3
    triangles = torch.stack([v1, v2, v3], dim=1)

    verts = torch.cat([v1, v2, v3], dim=0)
    torch.save(verts, 'vertices.pt')
    faces = torch.ones(triangles.shape[0], 3)
    faces[0, 0] = 0
    faces[0, 1] = triangles.shape[0]
    faces[0, 2] = triangles.shape[0]*2
    faces = torch.cumsum(faces, 0).long().cuda()
    torch.save(faces, 'faces.pt')
    b = verts[faces]

    print(triangles[0])


    # chose indexes if you want change partly
    idxs = None

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        new_triangles = transform_hotdog_fly(triangles, t[43])
        v1 = new_triangles[:, 0]
        v2 = new_triangles[:, 1]
        v3 = new_triangles[:, 2]
        #print(new_triangles[0])
        verts = torch.cat([v1, v2, v3], dim=0)
        torch.save(verts, 'vertices_after.pt')

        rendering = render(new_triangles, view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
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
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

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
    parser.add_argument('--gs_type', type=str, default="gs_points_flat")
    parser.add_argument("--num_splats", type=int, default=2)

    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)