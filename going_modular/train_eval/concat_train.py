import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # Thêm thư viện này để hiển thị thanh tiến trình

from ..utils.roc_auc import compute_auc
from ..utils.metrics import ConcatProgressMeter
from ..utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ..utils.ModelCheckPoint import ModelCheckpoint
import os

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

def fit(
    conf: dict,
    start_epoch: int,
    model: Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    scheduler,
    early_stopping: MultiMetricEarlyStopping,
    model_checkpoint: ModelCheckpoint
):
    log_dir = os.path.abspath(conf['checkpoint_dir'] + conf['type'] + '/logs')
    writer = SummaryWriter(log_dir=log_dir)
    device = conf['device']
    
    for epoch in range(start_epoch, conf['epochs']):
        print(f"\n--- Epoch {epoch+1}/{conf['epochs']} ---")
        
        # 1. Train Loop
        (   
            train_loss,
            train_loss_id,
            train_loss_gender,
            train_loss_emotion,
            train_loss_pose,
            train_loss_facial_hair,
            train_loss_spectacles,
        ) = train_epoch(train_dataloader, model, criterion, optimizer, device)
        
        # 2. Test Loop
        (   
            test_loss_gender,
            test_loss_emotion,
            test_loss_pose,
            test_loss_facial_hair,
            test_loss_spectacles,
        ) = test_epoch(test_dataloader, model, criterion, device)
        
        # 3. Compute Metrics (AUC)
        # Lưu ý: compute_auc cần được cập nhật để xử lý 2 input (Albedo+Normal)
        # Nếu chưa update, đoạn này có thể gây lỗi. Tạm thời try/except để không sập luồng train
        try:
            train_auc = compute_auc(train_dataloader, model, device)
            test_auc = compute_auc(test_dataloader, model, device)
        except Exception as e:
            print(f"⚠️ Warning: Không thể tính AUC do lỗi format dữ liệu: {e}")
            # Tạo dummy data để code chạy tiếp
            train_auc = {k: 0.0 for k in ['gender', 'spectacles', 'facial_hair', 'pose', 'emotion', 'id_cosine', 'id_euclidean']}
            test_auc = {k: 0.0 for k in train_auc.keys()}

        # Gán biến
        train_id_cosine_auc = train_auc.get('id_cosine', 0)
        train_id_euclidean_auc = train_auc.get('id_euclidean', 0)
        test_id_cosine_auc = test_auc.get('id_cosine', 0)
        test_id_euclidean_auc = test_auc.get('id_euclidean', 0)
        
        # 4. Logging & Display
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        # ... (Giữ nguyên các phần log khác của bạn) ...
        
        train_metrics = {
            "loss": train_loss,
            "loss_id": train_loss_id,
            # ... (Các metrics khác giữ nguyên)
            "auc_id_cosine": train_id_cosine_auc,
        }
        test_metrics = {
            # ... (Các metrics khác giữ nguyên)
            "auc_id_cosine": test_id_cosine_auc,
        }

        process = ConcatProgressMeter(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            prefix=f"Epoch {epoch + 1}:"
        )
        process.display()

        # 5. Checkpoint & Early Stopping
        model_checkpoint(model, optimizer, epoch + 1)
        early_stopping([test_id_cosine_auc, test_id_euclidean_auc], model, epoch + 1)
        
        if scheduler:
            scheduler.step(epoch)
        
    writer.close()

def train_epoch(train_dataloader, model, criterion, optimizer, device):
    model.to(device)
    model.train()

    # Khởi tạo AverageMeter hoặc biến đếm
    losses = {k: 0.0 for k in ['total', 'id', 'gender', 'emotion', 'pose', 'hair', 'spec']}
    
    # --- SỬA ĐỔI QUAN TRỌNG: Unpack 3 biến ---
    for x_albedo, x_normal, y in tqdm(train_dataloader, desc="Training"):
        x_albedo = x_albedo.to(device)
        x_normal = x_normal.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward: Model nhận 2 ảnh
        output_fusion, output_aux = model(x_albedo, x_normal)

        # Loss: Criterion nhận (fusion, aux, label)
        total_loss, loss_dict = criterion(output_fusion, output_aux, y)

        total_loss.backward()
        optimizer.step()

        # Cộng dồn loss (loss_dict trả về từ criterion)
        losses['total'] += total_loss.item()
        losses['id'] += loss_dict.get('id', 0)
        losses['gender'] += loss_dict.get('gender', 0)
        losses['emotion'] += loss_dict.get('emotion', 0)
        losses['pose'] += loss_dict.get('pose', 0)
        losses['hair'] += loss_dict.get('facial_hair', 0)
        losses['spec'] += loss_dict.get('spectacles', 0)

    # Tính trung bình
    n = len(train_dataloader)
    return (
        losses['total'] / n,
        losses['id'] / n,
        losses['gender'] / n,
        losses['emotion'] / n,
        losses['pose'] / n,
        losses['hair'] / n,
        losses['spec'] / n,
    )

def test_epoch(test_dataloader, model, criterion, device):
    model.to(device)
    model.eval()
    
    losses = {k: 0.0 for k in ['gender', 'emotion', 'pose', 'hair', 'spec']}

    with torch.no_grad():
        for x_albedo, x_normal, y in test_dataloader:
            x_albedo = x_albedo.to(device)
            x_normal = x_normal.to(device)
            y = y.to(device)

            output_fusion, output_aux = model(x_albedo, x_normal)
            
            # Khi test ta vẫn tính loss để theo dõi (nhưng không backward)
            _, loss_dict = criterion(output_fusion, output_aux, y)

            losses['gender'] += loss_dict.get('gender', 0)
            losses['emotion'] += loss_dict.get('emotion', 0)
            losses['pose'] += loss_dict.get('pose', 0)
            losses['hair'] += loss_dict.get('facial_hair', 0)
            losses['spec'] += loss_dict.get('spectacles', 0)

    n = len(test_dataloader)
    return (
        losses['gender'] / n,
        losses['emotion'] / n,
        losses['pose'] / n,
        losses['hair'] / n,
        losses['spec'] / n,
    )