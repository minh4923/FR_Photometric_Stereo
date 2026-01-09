import argparse # Thư viện xử lý tham số dòng lệnh
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

from going_modular.model.backbone.mifr import create_miresnet 
from going_modular.loss.ConcatMultiTaskLoss import ConcatMultiTaskLoss
from going_modular.dataloader.multitask import create_three_branch_loader 
from going_modular.train_eval.train_universal import fit_universal
from going_modular.utils.ModelCheckPoint import ModelCheckpoint
from going_modular.utils.ExperimentManager import ExperimentManager

# --- WRAPPER (Giữ nguyên) ---
class MIFR_Wrapper(torch.nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.fc_spectacles = torch.nn.Linear(512, 2)
        self.fc_facial_hair = torch.nn.Linear(512, 2)
        self.fc_emotion = torch.nn.Linear(512, 2)
        self.fc_pose = torch.nn.Linear(512, 2)
        self.fc_gender = torch.nn.Linear(512, 2)
        self.fc_id = torch.nn.Linear(512, num_classes) 

    def forward(self, x):
        features = self.backbone(x) 
        (x_spec, _), (x_hair, _), (x_emo, _), (x_pose, _), (x_gender, x_id) = features
        
        gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        flatten = torch.nn.Flatten()
        def process_feat(f): return flatten(gap(f))

        return (
            self.fc_spectacles(process_feat(x_spec)),
            self.fc_facial_hair(process_feat(x_hair)),
            self.fc_pose(process_feat(x_pose)),
            self.fc_emotion(process_feat(x_emo)),
            self.fc_gender(process_feat(x_gender)),
            self.fc_id(process_feat(x_id)),
            torch.norm(process_feat(x_id), dim=1, keepdim=True)
        )

def main():
    # 1. NHẬN THAM SỐ TỪ DÒNG LỆNH
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='albedo', help='Loại dữ liệu: albedo hoặc normal')
    parser.add_argument('--use_sampler', action='store_true', help='Nếu có cờ này thì dùng PK Sampler, không thì Shuffle')
    args = parser.parse_args()

    # Cập nhật Config dựa trên tham số
    CONFIGURATION = {
        'dataset_dir': '/content/drive/MyDrive/Photometric_DB_Full/', 
        'batch_size': 32, 
        'epochs': 120,    
        'learning_rate': 1e-4, 
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'step': 0, 
        # Tự động đổi tên Note theo loại dữ liệu
        'note': f"Single_{args.type.upper()}_{'Sampler' if args.use_sampler else 'Shuffle'}",
        'data_type': args.type # Lưu lại để dùng nếu cần logic riêng
    }

    manager = ExperimentManager(CONFIGURATION)
    manager.log_text(f" TRAINING MODE: {args.type.upper()} | Sampler: {args.use_sampler}")

    # 2. Transform (Logic riêng cho từng loại nếu cần, hiện tại dùng chung)
    train_tf = A.Compose([
        A.Resize(256, 256), A.RandomCrop(224, 224), A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()
    ], additional_targets={'image0': 'image', 'image1': 'image'})
    
    test_tf = A.Compose([
        A.Resize(256, 256), A.CenterCrop(224, 224),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()
    ], additional_targets={'image0': 'image', 'image1': 'image'})

    # 3. Dataloader (Truyền cờ use_sampler vào)
    train_dl, test_dl = create_three_branch_loader(CONFIGURATION, train_tf, test_tf, use_sampler=args.use_sampler)

    # 4. Model & Loss & Optimizer
    backbone_mifr = create_miresnet('miresnet50') 
    model = MIFR_Wrapper(backbone_mifr, num_classes=320)
    
    loss_weights = {
        'loss_spectacles_weight': 1.0, 'loss_facial_hair_weight': 1.0,
        'loss_pose_weight': 1.0, 'loss_gender_weight': 1.0, 'loss_emotion_weight': 1.0,
    }
    
    meta_path = os.path.join(CONFIGURATION['dataset_dir'], 'dataset/train_split.csv') 
    if not os.path.exists(meta_path): meta_path = os.path.join(CONFIGURATION['dataset_dir'], 'train_split.csv')
    criterion = ConcatMultiTaskLoss(meta_path, loss_weights)

    optimizer = optim.Adam(model.parameters(), lr=CONFIGURATION['learning_rate'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1, eta_min=1e-6)
    ckpt_saver = ModelCheckpoint(manager.ckpt_dir) 

    # 5. Train
    fit_universal(CONFIGURATION, model, train_dl, test_dl, criterion, optimizer, scheduler, ckpt_saver, manager)

if __name__ == '__main__':
    main()