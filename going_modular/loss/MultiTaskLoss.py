#going_modular/loss/MultiTaskLoss.py
import torch
import torch.nn as nn
from .WeightClassMagLoss import WeightClassMagLoss

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class MultiTaskLoss(nn.Module):
    def __init__(self, metadata_path:str, loss_weight:dict):
        super(MultiTaskLoss, self).__init__()

        # 1. ID Loss: Giữ nguyên MagFace
        self.id_loss = WeightClassMagLoss(metadata_path)

        # 2. Attribute Loss: Dùng CrossEntropy (Gradient sạch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cân bằng nhẹ bằng weight cố định
        self.gender_loss = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(device))
        self.spectacles_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
        self.facial_hair_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device))
        self.pose_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))
        self.emotion_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))

        # 3. UNCERTAINTY WEIGHTING (Vũ khí tối thượng)
        # 11 tham số learnable tương ứng với 11 task con
        self.num_tasks = 11
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(self, logits, y):
        (
            (x_spectacles, x_da_spectacles),
            (x_facial_hair, x_da_facial_hair),
            (x_pose, x_da_pose),
            (x_emotion, x_da_emotion),
            (x_gender, x_da_gender),
            x_id_logits, x_id_norm
        ) = logits

        id, gender, spectacles, facial_hair, pose, emotion = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5]

        # Tính Loss gốc
        l_id = self.id_loss(x_id_logits, id, x_id_norm)
        l_gender = self.gender_loss(x_gender, gender)
        l_da_gender = self.gender_loss(x_da_gender, gender)
        l_emotion = self.emotion_loss(x_emotion, emotion)
        l_da_emotion = self.emotion_loss(x_da_emotion, emotion)
        l_pose = self.pose_loss(x_pose, pose)
        l_da_pose = self.pose_loss(x_da_pose, pose)
        l_hair = self.facial_hair_loss(x_facial_hair, facial_hair)
        l_da_hair = self.facial_hair_loss(x_da_facial_hair, facial_hair)
        l_spec = self.spectacles_loss(x_spectacles, spectacles)
        l_da_spec = self.spectacles_loss(x_da_spectacles, spectacles)

        losses = [l_id, l_gender, l_emotion, l_pose, l_hair, l_spec,
                  l_da_gender, l_da_emotion, l_da_pose, l_da_hair, l_da_spec]

        # Tính Loss cân bằng tự động
        total_loss = 0
        for i, loss_item in enumerate(losses):
            precision = 0.5 * torch.exp(-self.log_vars[i])
            total_loss += precision * loss_item + 0.5 * self.log_vars[i]

        return (
            total_loss, l_id,
            l_gender, l_da_gender, l_emotion, l_da_emotion,
            l_pose, l_da_pose, l_hair, l_da_hair, l_spec, l_da_spec
        )