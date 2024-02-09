import torch
from scene.gaussian_model import GaussianModel, inverse_sigmoid
from utils.general_utils import rot_to_quat_batch, build_rotation
from torch import nn

class PointsGaussianModel(GaussianModel):

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.triangles = torch.empty(0)
        self.eps_s0 = 1e-8

        #self.scaling_activation = lambda x: 1/(1+torch.exp(-x)) * 0.6
        #self.scaling_inverse_activation = lambda x: torch.log((x * 6/10) / (1-x * 6/10))

        self.scaling_activation = lambda x: torch.exp(x)
        self.scaling_inverse_activation = lambda x: torch.log(x)

        #self.scaling_activation = lambda x: 1 / (1 - torch.exp(-x)) * 0.7
        #self.scaling_inverse_activation = lambda x: torch.log(x * 7/10 /(1-x * 7/10))
    def prepare_vertices(self):
        scales = self.get_scaling
        rotation = self._rotation
        R = build_rotation(rotation)
        v1 = self._xyz
        s = scales[:, -2].reshape(-1, 1)
        r = R[:, 1]
        v2 = v1 + s * r
        v3 = v1 + scales[:, -1].reshape(-1, 1) * R[:, 2]
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

        self.triangles = torch.stack([v1, v2, v3], dim = 1)


    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors.
        """

        def dot(v, u):
            return (v * u).sum(dim=-1, keepdim=True)

        def proj(v, u):
            """
            projection of vector v onto subspace spanned by u

            vector u is assumed to be already normalized
            """
            coef = dot(v, u)
            return coef * u

        v1 = self.triangles[:, 0].clone()
        v2 = self.triangles[:, 1].clone()
        v3 = self.triangles[:, 2].clone()

        _s2 = v2 - v1
        _s3 = v3 - v1

        r1 = torch.cross(_s2, _s3)
        s2 = torch.linalg.vector_norm(_s2, dim=-1, keepdim=True) + eps
        _s3_norm = torch.linalg.vector_norm(_s3, dim=-1, keepdim=True) + eps

        r1 = r1 / (torch.linalg.vector_norm(r1, dim=-1, keepdim=True) + eps)
        r2 = _s2 / s2
        r3 = _s3 - proj(_s3, r1) - proj(_s3, r2)
        r3 = r3 / (torch.linalg.vector_norm(r3, dim=-1, keepdim=True) + eps)
        s3 = dot(_s3, r3)

        scales = torch.cat([s2, s3], dim=1)
        self._scaling = self.scaling_inverse_activation(scales)

        rotation = torch.stack([r1, r2, r3], dim=1)
        self._rotation = rot_to_quat_batch(rotation)

    def _prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors.
        """

        v1 = self.triangles[:, 0].clone()
        v2 = self.triangles[:, 1].clone()
        v3 = self.triangles[:, 2].clone()

        _s2 = v2 - v1
        _s3 = v3 - v1
        s2 = torch.linalg.vector_norm(_s2, dim=-1, keepdim=True) + eps
        s3 = torch.linalg.vector_norm(_s3, dim=-1, keepdim=True) + eps
        scales = torch.cat([s2, s3], dim=1)
        self._scaling = self.scaling_inverse_activation(scales)

        r1 = torch.cross(_s2, _s3)
        r1 = r1 / (torch.linalg.vector_norm(r1, dim=-1, keepdim=True) + eps)
        r2 = (v2 - v1) / s2
        r3 = (v3 - v1) / s3
        rotation = torch.stack([r1, r2, r3], dim=1)
        rotation = rotation.transpose(-2, -1)
        self._rotation = rot_to_quat_batch(rotation)

    @property
    def get_scaling(self):
        self.s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s0
        return torch.cat([self.s0, self.scaling_activation(self._scaling[:, [-2, -1]])], dim=1)
