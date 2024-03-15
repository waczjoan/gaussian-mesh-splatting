import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlameConfig:
    def __init__(self):
        self.flame_model_path = 'games/flame_splatting/FLAME/model/flame2023.pkl'
        self.static_landmark_embedding_path = 'games/flame_splatting/FLAME/model/flame_static_embedding.pkl'
        self.dynamic_landmark_embedding_path = 'games/flame_splatting/FLAME/model/flame_dynamic_embedding.npy'
        self.shape_params = 100
        self.expression_params = 50
        self.pose_params = 6
        self.use_face_contour = True
        self.use_3D_translation = True
        self.optimize_eyeballpose = True
        self.optimize_neckpose = True
        self.num_worker = True
        self.batch_size = 1
        self.ring_margin = 0.5
        self.ring_loss_weight = 1.0
        self.device = device
        self.f_shape = torch.zeros(1, 100).float().to(device)
        self.f_exp = torch.zeros(1, 50).float().to(device)
        self.f_pose = torch.zeros(1, 6).float().to(device)
        self.f_neck_pose = torch.zeros(1, 3).float().to(device)
        self.f_trans = torch.zeros(1, 3).float().to(device)
        self.vertices_enlargement = 8.35
