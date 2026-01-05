import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatMTLFaceRecognitionV2(nn.Module):
    def __init__(self, model_albedo, model_normal, num_classes):
        super(ConcatMTLFaceRecognitionV2, self).__init__()
        self.model_albedo = model_albedo
        self.model_normal = model_normal
        
        # --- Fusion Layer ---
        # Input: 512 (Albedo) + 512 (Normal) = 1024
        # Output: 512 (Fused Feature)
        self.fusion_fc = nn.Linear(512 + 512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.prelu = nn.PReLU(512)
        
        # Lưu ý: Chúng ta không khai báo ID Head (ArcFace) ở đây 
        # vì nó đã nằm bên trong 2 model con (model_albedo.id_head) 
        # hoặc được xử lý bởi Loss Function tùy vào cách bạn viết Loss.
        # Với End-to-End đơn giản, ta sẽ dùng ID Head của nhánh chính (Albedo) 
        # hoặc trả về embedding để Loss tự xử lý.

    def forward(self, x_albedo, x_normal):
        # 1. Lấy feature embedding từ nhánh Albedo
        # get_embedding trả về tuple nhiều món, món cuối cùng [-1] là ID feature (512)
        emb_albedo = self.model_albedo.get_embedding(x_albedo)[-1]
        
        # 2. Lấy feature embedding từ nhánh Normal
        emb_normal = self.model_normal.get_embedding(x_normal)[-1]
        
        # 3. Nối (Concat) 2 vector lại: [Batch, 512] + [Batch, 512] -> [Batch, 1024]
        combined = torch.cat((emb_albedo, emb_normal), dim=1)
        
        # 4. Trộn thông tin (Fusion) -> Về lại [Batch, 512]
        x_fused = self.fusion_fc(combined)
        x_fused = self.bn(x_fused)
        x_fused = self.prelu(x_fused)
        
        # 5. Lấy kết quả phụ trợ từ nhánh chính (Albedo) để tính các loss phụ (Gender, Emotion...)
        # Ta chạy lại forward của Albedo để lấy full logits
        logits_albedo = self.model_albedo(x_albedo) 
        
        # QUAN TRỌNG: 
        # Trả về x_fused (để tính Loss ID chính) 
        # và logits_albedo (để tính các Loss phụ như Gender, Pose...)
        return x_fused, logits_albedo
    
    def get_embedding(self, x_albedo, x_normal):
        # Hàm dùng khi Test/Inference (chỉ cần vector đặc trưng cuối cùng)
        emb_albedo = self.model_albedo.get_embedding(x_albedo)[-1]
        emb_normal = self.model_normal.get_embedding(x_normal)[-1]
        
        combined = torch.cat((emb_albedo, emb_normal), dim=1)
        x_fused = self.fusion_fc(combined)
        x_fused = self.bn(x_fused)
        x_fused = self.prelu(x_fused)
        
        return x_fused