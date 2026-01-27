
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ..utils.roc_auc import compute_auc
from ..utils.metrics import ProgressMeter
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
    model_checkpoint: ModelCheckpoint,
    existing_manager=None # <--- Tích hợp Manager
):
    # 1. SETUP LOGGING
    if existing_manager:
        log_dir = existing_manager.log_dir
        manager = existing_manager
    else:
        # Fallback nếu không dùng manager
        log_dir = os.path.join(conf['checkpoint_dir'], conf['type'], 'logs')
        class Dummy: 
            def log_text(self, m): print(m)
            def log_metrics(self, e, m): pass
        manager = Dummy()

    writer = SummaryWriter(log_dir=log_dir)
    device = conf['device']
    
    manager.log_text(f"BAT DAU TRAINING: {conf['note']}")
    
    for epoch in range(start_epoch, conf['epochs']):
        manager.log_text(f"\n--- Epoch {epoch+1}/{conf['epochs']} ---")
        
        # 1. TRAIN
        (   
            train_loss, train_loss_id,
            train_loss_gender, train_loss_da_gender,
            train_loss_emotion, train_loss_da_emotion,
            train_loss_pose, train_loss_da_pose,
            train_loss_facial_hair, train_loss_da_facial_hair,
            train_loss_spectacles, train_loss_da_spectacles
        ) = train_epoch(train_dataloader, model, criterion, optimizer, device)
        
    
        (   
            test_loss_gender, test_loss_da_gender,
            test_loss_emotion, test_loss_da_emotion,
            test_loss_pose, test_loss_da_pose,
            test_loss_facial_hair, test_loss_da_facial_hair,
            test_loss_spectacles, test_loss_da_spectacles
        ) = test_epoch(test_dataloader, model, criterion, device)
        
        # 3. AUC
        train_auc = compute_auc(train_dataloader, model, device)
        test_auc = compute_auc(test_dataloader, model, device)
        
        # Mapping biến AUC cho gọn
        train_gender_auc = train_auc.get('gender', 0)
        train_spectacles_auc = train_auc.get('spectacles', 0)
        train_facial_hair_auc = train_auc.get('facial_hair', 0)
        train_pose_auc = train_auc.get('pose', 0)
        train_emotion_auc = train_auc.get('emotion', 0)
        train_id_cosine_auc = train_auc.get('id_cosine', 0)
        train_id_euclidean_auc = train_auc.get('id_euclidean', 0)
        
        test_gender_auc = test_auc.get('gender', 0)
        test_spectacles_auc = test_auc.get('spectacles', 0)
        test_facial_hair_auc = test_auc.get('facial_hair', 0)
        test_pose_auc = test_auc.get('pose', 0)
        test_emotion_auc = test_auc.get('emotion', 0)
        test_id_cosine_auc = test_auc.get('id_cosine', 0)
        test_id_euclidean_auc = test_auc.get('id_euclidean', 0)

        # LOG TENSORBOARD
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalars('Loss/gender', {'train': train_loss_gender, 'test': test_loss_gender}, epoch+1)
        writer.add_scalars('AUC/id_cosine', {'train': train_id_cosine_auc, 'test': test_id_cosine_auc}, epoch+1)
      

        
        train_metrics = {
            "loss": train_loss,
            "loss_id": train_loss_id,
            "loss_gender": train_loss_gender,
            "loss_da_gender": train_loss_da_gender,
            "loss_emotion": train_loss_emotion,
            "loss_da_emotion": train_loss_da_emotion,
            "loss_pose": train_loss_pose,
            "loss_da_pose": train_loss_da_pose,
            "loss_facial_hair": train_loss_facial_hair,
            "loss_da_facial_hair": train_loss_da_facial_hair,
            "loss_spectacles": train_loss_spectacles,
            "loss_da_spectacles": train_loss_da_spectacles,
            
            "auc_gender": train_gender_auc,
            "auc_spectacles": train_spectacles_auc,
            "auc_facial_hair": train_facial_hair_auc,
            "auc_pose": train_pose_auc,
            "auc_emotion": train_emotion_auc,
            "auc_id_cosine": train_id_cosine_auc,
            "auc_id_euclidean": train_id_euclidean_auc,
        }

        test_metrics = {
            "loss_gender": test_loss_gender,
            "loss_da_gender": test_loss_da_gender,
            "loss_emotion": test_loss_emotion,
            "loss_da_emotion": test_loss_da_emotion,
            "loss_pose": test_loss_pose,
            "loss_da_pose": test_loss_da_pose,
            "loss_facial_hair": test_loss_facial_hair,
            "loss_da_facial_hair": test_loss_da_facial_hair,
            "loss_spectacles": test_loss_spectacles,
            "loss_da_spectacles": test_loss_da_spectacles,
            
            "auc_gender": test_gender_auc,
            "auc_spectacles": test_spectacles_auc,
            "auc_facial_hair": test_facial_hair_auc,
            "auc_pose": test_pose_auc,
            "auc_emotion": test_emotion_auc,
            "auc_id_cosine": test_id_cosine_auc,
            "auc_id_euclidean": test_id_euclidean_auc,
        }

        # HIỂN THỊ RA MÀN HÌNH
        process = ProgressMeter(
            train_metrics=train_metrics, 
            test_metrics=test_metrics, 
            prefix=f"Ep {epoch + 1}:"
        )
        process.display()
        
        # GHI LOG VÀO FILE
        log_full = {**train_metrics, **test_metrics}
        manager.log_metrics(epoch+1, log_full)
       
        model_checkpoint(model, optimizer, epoch + 1, test_metrics, scheduler)
        
        # Scheduler Step
        scheduler.step(epoch)
        
    writer.close()
    manager.log_text("TRAINING HOAN TAT.")

# --- GIỮ NGUYÊN LOGIC TRAIN_EPOCH & TEST_EPOCH NHƯ FILE GỐC ---
def train_epoch(train_dataloader, model, criterion, optimizer, device):
    model.to(device); model.train()
    train_loss = 0
    t_id=0; t_gen=0; t_da_gen=0; t_emo=0; t_da_emo=0; 
    t_pose=0; t_da_pose=0; t_hair=0; t_da_hair=0; t_spec=0; t_da_spec=0

    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        
        (total_loss, l_id, l_gen, l_da_gen, l_emo, l_da_emo, 
         l_pose, l_da_pose, l_hair, l_da_hair, l_spec, l_da_spec) = criterion(logits, y)

        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        t_id += l_id.item(); t_gen += l_gen.item(); t_da_gen += l_da_gen.item()
        t_emo += l_emo.item(); t_da_emo += l_da_emo.item()
        t_pose += l_pose.item(); t_da_pose += l_da_pose.item()
        t_hair += l_hair.item(); t_da_hair += l_da_hair.item()
        t_spec += l_spec.item(); t_da_spec += l_da_spec.item()

    n = len(train_dataloader)
    # Trả về full tuple
    return (train_loss/n, t_id/n, t_gen/n, t_da_gen/n, t_emo/n, t_da_emo/n, 
            t_pose/n, t_da_pose/n, t_hair/n, t_da_hair/n, t_spec/n, t_da_spec/n)

def test_epoch(test_dataloader, model, criterion, device):
    model.to(device); model.eval()
    t_gen=0; t_da_gen=0; t_emo=0; t_da_emo=0; 
    t_pose=0; t_da_pose=0; t_hair=0; t_da_hair=0; t_spec=0; t_da_spec=0

    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            (_, _, l_gen, l_da_gen, l_emo, l_da_emo, 
             l_pose, l_da_pose, l_hair, l_da_hair, l_spec, l_da_spec) = criterion(logits, y)

            t_gen += l_gen.item(); t_da_gen += l_da_gen.item()
            t_emo += l_emo.item(); t_da_emo += l_da_emo.item()
            t_pose += l_pose.item(); t_da_pose += l_da_pose.item()
            t_hair += l_hair.item(); t_da_hair += l_da_hair.item()
            t_spec += l_spec.item(); t_da_spec += l_da_spec.item()

    n = len(test_dataloader)
    return (t_gen/n, t_da_gen/n, t_emo/n, t_da_emo/n, 
            t_pose/n, t_da_pose/n, t_hair/n, t_da_hair/n, t_spec/n, t_da_spec/n)