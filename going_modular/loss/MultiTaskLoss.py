import torch
import torch.nn as nn
from .WeightClassMagLoss import WeightClassMagLoss

# Seed cố định để debug cho dễ
seed = 42
torch.manual_seed(seed)

class MultiTaskLoss(nn.Module):
    def __init__(self, metadata_path:str, loss_weight:dict):
        super(MultiTaskLoss, self).__init__()

        # 1. Main Loss: MagFace (vừa nhận diện, vừa lọc ảnh nhiễu)
        self.id_loss = WeightClassMagLoss(metadata_path)

        # 2. Sub-tasks Loss: Dùng CE thuần
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mấy cái weight này hard-code theo kinh nghiệm (priority: Gender > Emotion > mấy cái kia)
        # Loss = -sum(w * y * log(p))
        self.gender_loss = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(device))
        self.spectacles_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
        self.facial_hair_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device))
        self.pose_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))
        self.emotion_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))

        # 3. Auto-Weighting (Paper: Multi-Task Learning Using Uncertainty - Kendall)
        # Thay vì chỉnh tay tỉ lệ loss (vd: 1.0*ID + 0.5*Gender), ta học tham số s.
        # s = log(sigma^2). Ban đầu s=0 (tức sigma=1).
        self.num_tasks = 11
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))

    def forward(self, logits, y):
        # Unpack đống output lằng nhằng từ model
        (
            (x_spectacles, x_da_spectacles),
            (x_facial_hair, x_da_facial_hair),
            (x_pose, x_da_pose),
            (x_emotion, x_da_emotion),
            (x_gender, x_da_gender),
            x_id_logits, x_id_norm
        ) = logits

        # Lấy label thật
        id, gender, spectacles, facial_hair, pose, emotion = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5]

        # --- Phase 1: Tính loss lẻ tẻ ---
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

        # --- Phase 2: Gộp loss (Formula: L_total = sum( 0.5*exp(-s)*L_i + 0.5*s )) ---
        total_loss = 0
        for i, loss_item in enumerate(losses):
            # Exp(-s) tương đương 1/sigma^2.
            # Nếu task này nhiễu (khó học) -> s tăng -> precision giảm -> Loss task này ít ảnh hưởng hơn.
            precision = 0.5 * torch.exp(-self.log_vars[i])
            
            # Cộng dồn: Phần loss có trọng số + Phần regularize log(sigma)
            total_loss += precision * loss_item + 0.5 * self.log_vars[i]

        return (
            total_loss, l_id,
            l_gender, l_da_gender, l_emotion, l_da_emotion,
            l_pose, l_da_pose, l_hair, l_da_hair, l_spec, l_da_spec
        )