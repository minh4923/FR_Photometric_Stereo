import torch
import torch.nn.functional as F

import pandas as pd

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class WeightClassMagLoss(torch.nn.Module):
    
    def __init__(self, file_path, l_a=10, u_a=110, scale=64, lambda_g=20):
        super(WeightClassMagLoss, self).__init__()
        self.l_a = l_a
        self.u_a = u_a
        self.scale = scale
        self.lambda_g = lambda_g
        # weight class
        self.alpha = self._compute_alpha(file_path)


    def _compute_alpha(self, file_path):
        self.data = pd.read_csv(file_path)
        
        # Đếm số lượng mẫu của từng nhãn
        label_counts = self.data['id'].value_counts().sort_index()
        
        # Tổng số mẫu
        total_samples = label_counts.sum()
        
        # Tính tần suất ngược
        alpha = 1 - (label_counts.values / total_samples) # class nhieu mau -> alpha nho , it mau -> alpha lon
        
        # Chuẩn hóa alpha về khoảng [0, 1]
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
        
        min_alpha, max_alpha = 0.3, 1
        alpha = alpha * (max_alpha - min_alpha) + min_alpha
        
        return torch.tensor(alpha, dtype=torch.float32)
    
    
    # input: là logits thu được của layer MagLinear.
    # target: là tensor label thực tế của cả batch ở dạng index từ 0
    # x_norm: là logits thu được của layer MagLinear
    def forward(self, logits, target, x_norm):
        target = target.long()
        loss_g = 1/(self.u_a**2) * x_norm + 1/(x_norm)

        cos_theta, cos_theta_m = logits
        
        cos_theta = self.scale * cos_theta
        cos_theta_m = self.scale * cos_theta_m
        
        # Khởi tạo 1 tensor chứa các giá trị 0 có kích thước giống như cos_theta
        one_hot = torch.zeros_like(cos_theta)
        # Onehotcoding label. Phương thức scatter_ sẽ thay các giá trị của one_hot tại chỉ mục được xác định bởi target thành 1.
        # dim=1 (dim=0 là chiều batch, 1 là chiều flatten)
        # target.view(-1, 1): chuyển target thành tensor cột có kích thước (batch_size,1)
        # 1.0: giá trị được gán
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        
        # Onehot là ma trận mục tiêu, ta muốn áp dụng cos_theta_m cho lớp mục tiêu này và giữ nguyên giá trị cosin_theta cho các lớp khác.
        # Đây là cách thức áp dụng margin vào cosine similarity để tăng cường phân biệt giữa lớp mục tiêu và các lớp khác.
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss_id = F.cross_entropy(output, target, reduction='none')
        # Trọng số alpha cho từng mẫu
        alpha = self.alpha.to(target.device)[target]
        # Tính tổng loss
        loss = torch.mean((loss_id + self.lambda_g * loss_g) * alpha)
        return loss
    
