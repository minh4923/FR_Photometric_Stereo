import torch
import torch.nn as nn

# Import các Loss thành phần của bạn
from .focalloss.FocalLoss import FocalLoss
from .WeightClassMagLoss import WeightClassMagLoss
# Import MagLinear từ model head cũ để dùng lại
from going_modular.model.head.id import MagLinear 

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class ConcatMultiTaskLoss(torch.nn.Module):
    
    def __init__(self, metadata_path:str, conf:dict):
        super(ConcatMultiTaskLoss, self).__init__()
        
        # --- 1. ID LOSS (Quan trọng nhất) ---
        # Vì model Fusion trả về Embedding (512), ta cần một lớp Head ở đây 
        # để tính Logits và Norm cho MagFace
        self.id_head = MagLinear(512, conf['num_classes'])
        self.id_loss = WeightClassMagLoss(metadata_path)
        
        # --- 2. AUX LOSS (Focal Loss cho các task phụ) ---
        # Giữ nguyên cấu hình Focal Loss tối ưu của bạn
        self.gender_loss = FocalLoss(alpha_weights={0:0.916, 1:0.084}, gamma_weights={0:2, 1:0}, num_classes=2)
        self.spectacles_loss = FocalLoss(alpha_weights={0: 0.28, 1: 0.72}, gamma_weights={0:0, 1:1}, num_classes=2)
        self.facial_hair_loss = FocalLoss(alpha_weights={0:0.3, 1:0.7}, gamma_weights={0:0, 1:1}, num_classes=2)
        self.pose_loss = FocalLoss(alpha_weights={0: 0.0263, 1: 0.9737}, gamma_weights={0:0, 1:2.5}, num_classes=2)
        self.emotion_loss = FocalLoss(alpha_weights={0:0.232, 1:0.768}, gamma_weights={0:0, 1:1}, num_classes=2)
        
        # Trọng số loss
        self.spectacles_weight = conf.get('loss_spectacles_weight', 10)
        self.facial_hair_weight = conf.get('loss_facial_hair_weight', 10)
        self.pose_weight = conf.get('loss_pose_weight', 30)
        self.gender_weight = conf.get('loss_gender_weight', 30)
        self.emotion_weight = conf.get('loss_emotion_weight', 10)
        
        
    def forward(self, x_fused, aux_outputs, y):
        """
        x_fused: [Batch, 512] - Feature Fusion
        aux_outputs: Tuple chứa logits của các nhánh phụ (từ Albedo backbone)
        y: Labels
        """
        
        # Tách nhãn
        label_id = y[:, 0]
        label_gender = y[:, 1]
        label_spectacles = y[:, 2]
        label_facial_hair = y[:, 3]
        label_pose = y[:, 4]
        label_emotion = y[:, 5]
        
        # --- TÍNH ID LOSS ---
        # 1. Chuyển Embedding -> Logits & Norm (thông qua MagLinear)
        x_id_logits, x_id_norm = self.id_head(x_fused)
        
        # 2. Tính Loss MagFace
        loss_id = self.id_loss(x_id_logits, label_id, x_id_norm)
        
        # --- TÍNH AUX LOSS ---
        # aux_outputs có cấu trúc: ((x_spec, _), (x_hair, _), (x_emo, _), (x_pose, _), (x_gender, _), ...)
        # Ta lấy phần tử đầu tiên của mỗi tuple (là logits)
        
        x_spectacles = aux_outputs[0][0]
        x_facial_hair = aux_outputs[1][0]
        x_emotion = aux_outputs[2][0]
        x_pose = aux_outputs[3][0]
        x_gender = aux_outputs[4][0]
        
        loss_spectacles = self.spectacles_loss(x_spectacles, label_spectacles)
        loss_facial_hair = self.facial_hair_loss(x_facial_hair, label_facial_hair)
        loss_pose = self.pose_loss(x_pose, label_pose)
        loss_emotion = self.emotion_loss(x_emotion, label_emotion)
        loss_gender = self.gender_loss(x_gender, label_gender)
        
        # Tổng hợp
        total_loss =    loss_id + \
                        loss_gender * self.gender_weight + \
                        loss_emotion * self.emotion_weight + \
                        loss_pose * self.pose_weight + \
                        loss_facial_hair * self.facial_hair_weight + \
                        loss_spectacles * self.spectacles_weight
        
        loss_dict = {
            'id': loss_id,
            'gender': loss_gender,
            'emotion': loss_emotion,
            'pose': loss_pose,
            'facial_hair': loss_facial_hair,
            'spectacles': loss_spectacles
        }
        
        return total_loss, loss_dict