import torch
import torch.nn as nn
import timm # Thư viện chứa ConvNeXt V2 chuẩn

# --- 1. GIỮ NGUYÊN CÁC MODULE ATTENTION CỦA BẠN ---
# (Mình copy y nguyên từ mifr.py sang để đảm bảo logic tách feature không đổi)

class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)
        
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in pool_size]))
        
        self.channel = nn.Sequential(
            nn.Conv2d(_channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(_channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Attention channel
        channel_input = self.avg_spp(x) + self.max_spp(x)
        channel_scale = self.channel(channel_input)

        # Attention spatial
        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)

        # Tách feature
        x_non_id = (x * channel_scale + x * spatial_scale) * 0.5
        x_id = x - x_non_id
        
        return x_id, x_non_id

# --- 2. BACKBONE MỚI: CONVNEXT V2 TÍCH HỢP LOGIC MI ---

class MIConvNeXtV2(nn.Module):
    def __init__(self, model_name='convnextv2_tiny', pretrained=True, **kwargs):
        super(MIConvNeXtV2, self).__init__()
        
        # A. Load Backbone ConvNeXt V2 từ thư viện timm
        # 'features_only=True' để lấy feature map cuối cùng (chưa qua lớp classification)
        print(f"--- Loading ConvNeXt V2: {model_name} (Pretrained={pretrained}) ---")
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Lấy thông tin số channel đầu ra của backbone
        # ConvNeXt Tiny/Base thường ra 768 hoặc 1024 channels
        dummy_input = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            last_feature_map = features[-1] # Lấy tầng sâu nhất
            out_channels = last_feature_map.shape[1]
            H, W = last_feature_map.shape[2], last_feature_map.shape[3]
            
        print(f"-> Backbone Output Shape: {out_channels}x{H}x{W}")
        
        # B. Lớp chuyển đổi (Adapter Layer)
        # Hệ thống cũ của bạn (AttentionModule) đang chạy với input 512 channel (từ ResNet18)
        # ConvNeXt ra 768/1024 channel -> Cần nén xuống 512
        self.target_channels = 512
        self.adapter_conv = nn.Conv2d(out_channels, self.target_channels, kernel_size=1, bias=False)
        self.adapter_bn = nn.BatchNorm2d(self.target_channels)
        self.adapter_act = nn.PReLU(self.target_channels)
        
        # C. Các module Attention (Logic cũ của bạn)
        self.spectacles_fsm = AttentionModule(channels=self.target_channels)
        self.facial_hair_fsm = AttentionModule(channels=self.target_channels)
        self.emotion_fsm = AttentionModule(channels=self.target_channels)
        self.pose_fsm = AttentionModule(channels=self.target_channels)
        self.gender_fsm = AttentionModule(channels=self.target_channels)

    def forward(self, x):
        # 1. Trích xuất đặc trưng từ ConvNeXt
        # timm trả về list các feature map, ta lấy cái cuối cùng (xịn nhất)
        features = self.backbone(x)[-1] 
        
        # 2. Nén channel về 512 (Adapter)
        x = self.adapter_conv(features)
        x = self.adapter_bn(x)
        x = self.adapter_act(x)
        
        # 3. Chuỗi tách Feature (Logic cũ của MIResNet)
        x_non_spectacles, x_spectacles = self.spectacles_fsm(x)
        x_non_facial_hair, x_facial_hair = self.facial_hair_fsm(x_non_spectacles)
        x_non_emotion, x_emotion = self.facial_hair_fsm(x_non_facial_hair)
        x_non_pose, x_pose = self.pose_fsm(x_non_emotion)
        x_id, x_gender = self.gender_fsm(x_non_pose)
        
        # 4. Trả về đúng format tuple mà MTLFaceRecognition yêu cầu
        return (
                (x_spectacles, x_non_spectacles),
                (x_facial_hair, x_non_facial_hair),
                (x_emotion, x_non_emotion),
                (x_pose, x_non_pose),
                (x_gender, x_id)
            )

# --- 3. HÀM KHỞI TẠO ---
def create_miconvnextv2(model_name='convnextv2_tiny', **kwargs):
    # Hỗ trợ các phiên bản: atto, femto, pico, nano, tiny, base, large
    # Khuyên dùng 'tiny' hoặc 'base' cho cân bằng hiệu năng/tốc độ
    return MIConvNeXtV2(model_name, **kwargs)

if __name__ == '__main__':
    # Test thử
    model = create_miconvnextv2('convnextv2_tiny')
    x = torch.randn(2, 3, 112, 112)
    output = model(x)
    print("Test Output Structure:")
    print(f"Số lượng nhánh chính: {len(output)}")
    print(f"Shape ID Feature: {output[-1][-1].shape}") # Mong đợi (2, 512, 7, 7)