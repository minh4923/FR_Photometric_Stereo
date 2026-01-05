import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import traceback # <--- THÊM CÁI NÀY ĐỂ SOI LỖI

from ..utils.roc_auc import compute_auc
from ..utils.metrics import ConcatProgressMeter
from ..utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ..utils.ModelCheckPoint import ModelCheckpoint

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
    log_dir = os.path.join(conf['checkpoint_dir'], conf['type'], 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    device = conf['device']
    
    for epoch in range(start_epoch, conf['epochs']):
        print(f"\n--- Epoch {epoch+1}/{conf['epochs']} ---")
        
        # 1. TRAIN
        train_metrics = train_epoch(train_dataloader, model, criterion, optimizer, device)
        
        # 2. TEST
        test_metrics = test_epoch(test_dataloader, model, criterion, device)
        
        # 3. AUC (PHẦN QUAN TRỌNG NHẤT: BẮT LỖI CHI TIẾT)
        test_id_auc = 0.0
        try:
            # Gọi hàm tính AUC
            test_auc = compute_auc(test_dataloader, model, device)
            
            # Lấy giá trị ID Cosine
            if isinstance(test_auc, dict) and 'id_cosine' in test_auc:
                test_id_auc = test_auc['id_cosine']
            else:
                print(f" Cảnh báo: compute_auc không trả về dict chứa 'id_cosine'. Kết quả nhận được: {test_auc}")
                
        except Exception as e:
            print(f"\n LỖI NGHIÊM TRỌNG KHI TÍNH AUC TẠI EPOCH {epoch+1}:")
            print(f" Lý do: {e}")
            print(" Chi tiết lỗi (Traceback):")
            traceback.print_exc() # In ra dòng code gây lỗi
            test_id_auc = 0.0

        # 4. LOGGING
        writer.add_scalar('Loss/train', train_metrics['total'], epoch+1)
        writer.add_scalar('AUC/id_cosine', test_id_auc, epoch+1)
        
        # Hiển thị bảng đẹp
        # Gộp metric train và test vào để hiển thị
        display_train = train_metrics.copy()
        display_test = test_metrics.copy()
        
        # Thêm AUC vào dict để hiển thị ra bảng
        display_train['auc_id_cosine'] = 0.0 # Train không tính AUC để tiết kiệm time
        display_test['auc_id_cosine'] = test_id_auc
        
        process = ConcatProgressMeter(
            train_metrics=display_train,
            test_metrics=display_test,
            prefix=f"Epoch {epoch + 1}:"
        )
        process.display()

        # 5. Checkpoint & Early Stopping
        model_checkpoint(model, optimizer, epoch + 1)
        early_stopping([-test_id_auc], model, epoch + 1) # Dấu trừ vì ta muốn Maximize AUC
        
        if scheduler:
            scheduler.step(epoch)
        
    writer.close()

def train_epoch(train_dataloader, model, criterion, optimizer, device):
    model.to(device)
    model.train()

    # Khởi tạo dict để lưu tất cả các loại loss
    losses = {k: 0.0 for k in ['total', 'id', 'gender', 'emotion', 'pose', 'facial_hair', 'spectacles']}
    
    for x_albedo, x_normal, y in tqdm(train_dataloader, desc="Training"):
        x_albedo, x_normal, y = x_albedo.to(device), x_normal.to(device), y.to(device)

        optimizer.zero_grad()
        output_fusion, output_aux = model(x_albedo, x_normal)
        total_loss, loss_dict = criterion(output_fusion, output_aux, y)

        total_loss.backward()
        optimizer.step()

        # Cộng dồn loss (An toàn hơn với .get)
        losses['total'] += total_loss.item()
        for k in losses:
            if k != 'total' and k in loss_dict:
                val = loss_dict[k]
                losses[k] += val.item() if torch.is_tensor(val) else val

    # Tính trung bình
    n = len(train_dataloader)
    return {k: v / n for k, v in losses.items()}

def test_epoch(test_dataloader, model, criterion, device):
    model.to(device)
    model.eval()
    
    losses = {k: 0.0 for k in ['total', 'id', 'gender', 'emotion', 'pose', 'facial_hair', 'spectacles']}

    with torch.no_grad():
        for x_albedo, x_normal, y in test_dataloader:
            x_albedo, x_normal, y = x_albedo.to(device), x_normal.to(device), y.to(device)

            output_fusion, output_aux = model(x_albedo, x_normal)
            total_loss, loss_dict = criterion(output_fusion, output_aux, y)

            losses['total'] += total_loss.item()
            for k in losses:
                if k != 'total' and k in loss_dict:
                    val = loss_dict[k]
                    losses[k] += val.item() if torch.is_tensor(val) else val

    n = len(test_dataloader)
    return {k: v / n for k, v in losses.items()}