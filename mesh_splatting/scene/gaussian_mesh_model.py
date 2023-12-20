import torch
import numpy as np

from torch import nn

from scene.gaussian_model import GaussianModel
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.sh_utils import RGB2SH
from mesh_splatting.utils.graphics_utils import MeshPointCloud


class GaussianMeshModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_claud = None
        self.alpha = None
        self.softmax = torch.nn.Softmax(dim=1)

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):

        self.point_claud = pcd
        self.spatial_lr_scale = spatial_lr_scale
        pcd_alpha_shape = pcd.alpha.shape

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()

        print("Number of points at initialisation : ",
              alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        # TODO dist2, scales, rots, opacities
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.alpha)).float().cuda()), 0.0000001
        )

        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((alpha_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((alpha_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self.alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))  # check update_alpha
        self._calc_xyz()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        self._xyz = torch.matmul(
            self.point_claud.alpha,
            self.point_claud.triangles
        )

    def update_alpha(self):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0

        """
        self.alpha = self.softmax(self.alpha)
