import torch
from scene.gaussian_model import GaussianModel
from utils.general_utils import rot_to_quat_batch, build_rotation

class PointsGaussianModel(GaussianModel):

    def __init__(self, sh_degree: int):
        super().__init__(sh_degree)
        self.triangles = torch.empty(0)
        self.eps_s0 = 1e-8
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

    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors.
        """

        v1 = self.v1.clone()
        v2 = self.v2.clone()
        v3 = self.v3.clone()

        self.triangles = torch.stack([v1, v2, v3], dim = 1)
        a = self.triangles[:, 0]

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
        self._rotation = rot_to_quat_batch(rotation)

    def prepare_scaling_rot_animate(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from
        centroid to 2nd vertex onto subspace spanned by v0 and v1.
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

        self.triangles = torch.stack([self.v1, self.v2, self.v3], dim = 1)

        triangles = self.triangles
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
            dim=1
        )
        v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
        means = torch.mean(triangles, dim=1)
        v1 = triangles[:, 1] - means
        v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
        v1 = v1 / v1_norm
        v2_init = triangles[:, 2] - means
        v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
        v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

        s1 = v1_norm / 2.
        s2 = dot(v2_init, v2) / 2.
        scales = torch.cat([s1, s2], dim=1)
        self._scaling = self.scaling_inverse_activation(scales)

        rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
        self._rotation = rot_to_quat_batch(rotation)

    @property
    def get_scaling(self):
        self.s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s0
        return torch.cat([self.s0, self.scaling_activation(self._scaling[:, [-2, -1]])], dim=1)
