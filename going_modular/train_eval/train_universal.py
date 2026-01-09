import torch
import torch.nn as nn
from tqdm import tqdm
from ..utils.roc_auc import compute_auc
from ..utils.metrics import ConcatProgressMeter
from ..utils.ExperimentManager import ExperimentManager

def train_one_epoch(dataloader, model, criterion, optimizer, device):
    model.train()
    running_metrics = {} 
    count = 0

    # Danh sach ten loss (Khop voi thu tu tra ve cua ham Loss cu)
    # Bao gom ca loss thuong va loss Domain Adaptation (da_)
    loss_keys = [
        'loss', 
        'loss_id', 
        'loss_gender', 'loss_da_gender',
        'loss_emotion', 'loss_da_emotion',
        'loss_pose', 'loss_da_pose',
        'loss_facial_hair', 'loss_da_facial_hair',
        'loss_spectacles', 'loss_da_spectacles'
    ]

    for X, y in tqdm(dataloader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits = model(X)
        losses_tuple = criterion(logits, y)
        
        # Phan tu dau tien luon la Total Loss de backward
        total_loss = losses_tuple[0] 
        total_loss.backward()
        optimizer.step()
        
        count += 1
        # Tu dong map gia tri vao ten
        for i, val in enumerate(losses_tuple):
            if i < len(loss_keys):
                key = loss_keys[i]
                running_metrics[key] = running_metrics.get(key, 0) + val.item()

    return {k: v / count for k, v in running_metrics.items()}

def evaluate(dataloader, model, criterion, device):
    model.eval()
    running_metrics = {}
    count = 0
    
    loss_keys = [
        'loss', # Placeholder
        'loss_id', 
        'loss_gender', 'loss_da_gender',
        'loss_emotion', 'loss_da_emotion',
        'loss_pose', 'loss_da_pose',
        'loss_facial_hair', 'loss_da_facial_hair',
        'loss_spectacles', 'loss_da_spectacles'
    ]

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Evaluating", leave=False):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            losses_tuple = criterion(logits, y)
            
            count += 1
            # Logic xu ly khac biet so luong loss tra ve giua train/test neu co
            current_keys = loss_keys
            if len(losses_tuple) < len(loss_keys): 
                # Fallback logic neu ham test tra ve it loss hon
                pass 

            for i, val in enumerate(losses_tuple):
                if i < len(current_keys):
                    key = current_keys[i]
                    running_metrics[key] = running_metrics.get(key, 0) + val.item()

    return {k: v / count for k, v in running_metrics.items()}

def fit_universal(conf, model, train_dl, test_dl, criterion, optimizer, scheduler, model_checkpoint, existing_manager=None):
    # Khoi tao Manager (hoac dung cai co san)
    if existing_manager:
        manager = existing_manager
    else:
        manager = ExperimentManager(conf)
        
    device = conf['device']
    model.to(device)
    
    manager.log_text(f"BAT DAU TRAINING: {conf['note']}")
    
    for epoch in range(conf['epochs']):
        manager.log_text(f"\n--- Epoch {epoch+1}/{conf['epochs']} ---")
        
        # 1. Train
        train_metrics = train_one_epoch(train_dl, model, criterion, optimizer, device)
        
        # 2. Test Loss
        test_metrics = evaluate(test_dl, model, criterion, device)
        
        # 3. Test AUC (Chi chay tren tap Test)
        auc_results = compute_auc(test_dl, model, device)
        for k, v in auc_results.items():
            test_metrics[f"auc_{k}"] = v

        # 4. Hien thi & Luu
        process = ConcatProgressMeter(train_metrics, test_metrics, prefix=f"Ep {epoch+1}:")
        process.display()
        
        # Gop log
        full_log = {}
        for k, v in train_metrics.items(): full_log[f"train_{k}"] = v
        for k, v in test_metrics.items(): full_log[f"val_{k}"] = v
        
        manager.log_metrics(epoch+1, full_log)
        
        # 5. Checkpoint
        model_checkpoint(model, optimizer, epoch+1)
        if scheduler: scheduler.step(epoch)

    manager.log_text("TRAINING HOAN TAT.")