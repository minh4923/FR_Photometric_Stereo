import torch
import torch.nn as nn

class ConcatMTLFaceRecognitionV2(nn.Module):
    def __init__(self, model_albedo, model_normal, num_classes):
        super(ConcatMTLFaceRecognitionV2, self).__init__()
        self.model_albedo = model_albedo
        self.model_normal = model_normal
        
        # Fusion Layer: 512 + 512 -> 1024 -> 512
        self.fusion_fc = nn.Linear(512 + 512, 512)
        self.bn = nn.BatchNorm1d(512)
        self.prelu = nn.PReLU(512)

    def forward(self, x_albedo, x_normal):
        # Lấy feature embedding từ 2 nhánh
        emb_albedo = self.model_albedo.get_embedding(x_albedo)[-1]
        emb_normal = self.model_normal.get_embedding(x_normal)[-1]
        
        # Nối và nén
        combined = torch.cat((emb_albedo, emb_normal), dim=1)
        x_fused = self.fusion_fc(combined)
        x_fused = self.bn(x_fused)
        x_fused = self.prelu(x_fused)
        
        # Lấy kết quả phụ trợ từ nhánh chính (Albedo) để tính Loss phụ
        logits_albedo = self.model_albedo(x_albedo) 
        
        return x_fused, logits_albedo
    
    def get_embedding(self, x_albedo, x_normal):
        emb_albedo = self.model_albedo.get_embedding(x_albedo)[-1]
        emb_normal = self.model_normal.get_embedding(x_normal)[-1]
        combined = torch.cat((emb_albedo, emb_normal), dim=1)
        x_fused = self.fusion_fc(combined)
        x_fused = self.bn(x_fused)
        x_fused = self.prelu(x_fused)
        return x_fused