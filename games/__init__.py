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

from arguments import OptimizationParams
from arguments_games import (
    OptimizationParamsMesh,
    OptimizationParamsFlame,
)

from scene.gaussian_model import GaussianModel
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
from games.multi_mesh_splatting.scene.gaussian_multi_mesh_model import GaussianMultiMeshModel
from games.flame_splatting.scene.gaussian_flame_model import GaussianFlameModel
from games.flat_splatting.scene.points_gaussian_model import PointsGaussianModel
from games.flat_splatting.scene.flat_gaussian_model import FlatGaussianModel

optimizationParamTypeCallbacks = {
    "gs": OptimizationParams,
    "gs_multi_mesh": OptimizationParamsMesh,
    "gs_flat": OptimizationParams,
    "gs_mesh": OptimizationParamsMesh,
    "gs_flame": OptimizationParamsFlame
}

gaussianModel = {
    "gs": GaussianModel,
    "gs_flat": FlatGaussianModel,
    "gs_mesh": GaussianMeshModel,
    "gs_multi_mesh": GaussianMultiMeshModel,
    "gs_flame": GaussianFlameModel,
    "gs_points": PointsGaussianModel
}

gaussianModelRender = {
    "gs": GaussianModel,
    "gs_flat": FlatGaussianModel,
    "gs_mesh": GaussianMeshModel,
    "gs_multi_mesh": GaussianMultiMeshModel,
    "gs_flame": GaussianFlameModel,
    "gs_points": PointsGaussianModel
}
