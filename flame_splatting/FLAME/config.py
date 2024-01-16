class FlameConfig:
    def __init__(self, flame_model_path):
        self.flame_model_path = flame_model_path
        self.static_landmark_embedding_path = './FLAME/model/flame_static_embedding.pkl'
        self.dynamic_landmark_embedding_path = './FLAME/model/flame_dynamic_embedding.npy'
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
