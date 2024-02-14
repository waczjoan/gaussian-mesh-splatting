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

import os
import numpy as np
import torch

from games.flame_splatting.utils.graphics_utils import FLAMEPointCloud
from scene.dataset_readers import (
    readColmapSceneInfo,
    readNerfSyntheticInfo,
    readCamerasFromTransforms,
    getNerfppNorm,
    SceneInfo,
    storePly,
)
from games.mesh_splatting.scene.dataset_readers import (
    readNerfSyntheticMeshInfo
)
from games.multi_mesh_splatting.scene.dataset_readers import (
    readColmapMeshSceneInfo
)
from utils.sh_utils import SH2RGB
from games.flame_splatting.FLAME import FLAME
from games.flame_splatting.FLAME.config import FlameConfig

softmax = torch.nn.Softmax(dim=2)


def transform_vertices_function(vertices, c=8):
    vertices = torch.squeeze(vertices)
    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] = -vertices[:, 1]
    vertices *= c
    return vertices


def readNerfSyntheticFlameInfo(
        path, white_background, eval, extension=".png"
):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    print("Reading Mesh object")

    flame_config = FlameConfig()
    model_flame = FLAME(flame_config).to(flame_config.device)

    vertices, _ = model_flame(
            flame_config.f_shape, flame_config.f_exp, flame_config.f_pose,
            neck_pose=flame_config.f_neck_pose, transl=flame_config.f_trans
    )
    vertices = transform_vertices_function(
        vertices,
        c=flame_config.vertices_enlargement
    )

    faces = torch.tensor(model_flame.faces.astype(np.int32))
    faces = torch.squeeze(faces)
    faces = faces.to(flame_config.device).long()

    triangles = vertices[faces]

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts_each_triangle = 100
        num_pts = num_pts_each_triangle * triangles.shape[0]
        print(
            f"Generating random point cloud ({num_pts})..."
        )

        # We create random points inside the bounds traingles
        alpha = torch.rand(
            triangles.shape[0],
            num_pts_each_triangle,
            3
        ).to(flame_config.device)

        xyz = torch.matmul(
            alpha,
            triangles
        )
        xyz = xyz.reshape(num_pts, 3)

        shs = np.random.random((num_pts, 3)) / 255.0

        pcd = FLAMEPointCloud(
            alpha=alpha,
            points=xyz.cpu(),
            colors=SH2RGB(shs),
            normals=np.zeros((num_pts, 3)),
            flame_model=model_flame,
            faces=faces,
            vertices_init=vertices,
            transform_vertices_function=transform_vertices_function,
            flame_model_shape_init=flame_config.f_shape,
            flame_model_expression_init=flame_config.f_exp,
            flame_model_pose_init=flame_config.f_pose,
            flame_model_neck_pose_init=flame_config.f_neck_pose,
            flame_model_transl_init=flame_config.f_trans,
            vertices_enlargement_init=flame_config.vertices_enlargement
        )

        storePly(ply_path, pcd.points, SH2RGB(shs) * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Colmap_Mesh": readColmapMeshSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Blender_Mesh": readNerfSyntheticMeshInfo,
    "Blender_FLAME": readNerfSyntheticFlameInfo
}
