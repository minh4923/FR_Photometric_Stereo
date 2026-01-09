import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Import modules
from going_modular.model.backbone.irse import IR_50  
from going_modular.model.head.MTLFaceRecognition import MTLFaceRecognition
from going_modular.loss.ConcatMultiTaskLoss import ConcatMultiTaskLoss
from going_modular.dataloader.multitask import create_three_branch_loader 
from going_modular.train_eval.train_universal import fit_universal
from going_modular.utils.ModelCheckPoint import ModelCheckpoint
from going_modular.utils.ExperimentManager import ExperimentManager # Import Manager

# --- CẤU HÌNH (CONFIG) ---
CONFIGURATION = {
    'dataset_dir': '/content/drive/MyDrive/Photometric_DB_Full/', 
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Thông tin thí nghiệm
    'step': 0, # Bước 0
    'note': 'Single_Albedo_Auxiliary_Learning',
}

def main():
    # 1. Khởi tạo Experiment Manager trước tiên
    # Để lấy đường dẫn checkpoint cho class ModelCheckpoint
    manager = ExperimentManager(CONFIGURATION)
    manager.log_text(" Đang khởi tạo Single Albedo Training...")

    # 2. Chuẩn bị Dataloader
    train_tf = A.Compose([
        A.Resize(112, 112),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ], additional_targets={'image0': 'image', 'image1': 'image'})
    
    test_tf = A.Compose([
        A.Resize(112, 112),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ], additional_targets={'image0': 'image', 'image1': 'image'})

    train_dl, test_dl = create_three_branch_loader(CONFIGURATION, train_tf, test_tf)

    # 3. Khởi tạo Model (Single Task)
    backbone = IR_50(input_size=(112, 112)) 
    model = MTLFaceRecognition(backbone, num_classes=320) 
    
    # 4. Khởi tạo Loss (Auxiliary Learning - Ưu tiên ID)
    loss_weights = {
        'loss_spectacles_weight': 0.1,  
        'loss_facial_hair_weight': 0.1,
        'loss_pose_weight': 0.1,
        'loss_gender_weight': 0.1,
        'loss_emotion_weight': 0.1,
        # ID mặc định là 1.0
    } 
    
    # Đường dẫn file csv train để tính class weight
    meta_path = os.path.join(CONFIGURATION['dataset_dir'], 'dataset/train_split.csv') 
    
    # Kiểm tra file meta có tồn tại không để tránh crash
    if not os.path.exists(meta_path):
        # Fallback: Nếu cấu trúc thư mục khác, thử tìm ở root
        meta_path = os.path.join(CONFIGURATION['dataset_dir'], 'train_split.csv')
        
    criterion = ConcatMultiTaskLoss(meta_path, loss_weights)

    # 5. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=CONFIGURATION['learning_rate'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 6. Checkpoint Saver
    ckpt_saver = ModelCheckpoint(manager.ckpt_dir) 

    # 7. BẮT ĐẦU TRAIN
    # Lưu ý: Cần sửa nhẹ fit_universal để nhận tham số 'manager' truyền vào
    # (Tôi đã sửa code fit_universal ở dưới đây để nhận tham số này)
    fit_universal(
        conf=CONFIGURATION,
        model=model,
        train_dl=train_dl,
        test_dl=test_dl,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_checkpoint=ckpt_saver,
        existing_manager=manager # <--- Truyền manager đã tạo vào
    )

if __name__ == '__main__':
    main()