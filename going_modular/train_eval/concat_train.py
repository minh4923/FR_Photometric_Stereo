import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from ..utils.roc_auc import compute_auc
from ..utils.metrics import ConcatProgressMeter

def fit(conf, start_epoch, model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, early_stopping, model_checkpoint):
    log_dir = os.path.join(conf['checkpoint_dir'], conf['type'], 'logs')
    writer = SummaryWriter(log_dir=log_dir)
    device = conf['device']
    
    for epoch in range(start_epoch, conf['epochs']):
        print(f"\n--- Epoch {epoch+1}/{conf['epochs']} ---")
        
        # TRAIN
        train_metrics = train_epoch(train_dataloader, model, criterion, optimizer, device)
        
        # TEST
        test_metrics = test_epoch(test_dataloader, model, criterion, device)
        
        # AUC (Try/Except để tránh crash)
        try:
            test_auc = compute_auc(test_dataloader, model, device)
            test_id_auc = test_auc.get('id_cosine', 0.0)
        except:
            test_id_auc = 0.0

        # LOGGING
        writer.add_scalar('Loss/train', train_metrics['total'], epoch+1)
        writer.add_scalar('AUC/id_cosine', test_id_auc, epoch+1)
        
        print(f"Train Loss: {train_metrics['total']:.4f} | ID Loss: {train_metrics['id']:.4f} | Val AUC: {test_id_auc:.4f}")

        # Checkpoint
        model_checkpoint(model, optimizer, epoch + 1)
        early_stopping([-test_id_auc], model, epoch + 1)
        
        if scheduler: scheduler.step(epoch)
    writer.close()

def train_epoch(dataloader, model, criterion, optimizer, device):
    model.to(device); model.train()
    losses = {k: 0.0 for k in ['total', 'id']}
    
    for x1, x2, y in tqdm(dataloader, desc="Training"):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        
        out_fusion, out_aux = model(x1, x2)
        loss, loss_dict = criterion(out_fusion, out_aux, y)
        
        loss.backward()
        optimizer.step()
        
        losses['total'] += loss.item()
        losses['id'] += loss_dict['id'].item()

    n = len(dataloader)
    return {k: v/n for k, v in losses.items()}

def test_epoch(dataloader, model, criterion, device):
    model.to(device); model.eval()
    losses = {k: 0.0 for k in ['total']}
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            out_fusion, out_aux = model(x1, x2)
            _, loss_dict = criterion(out_fusion, out_aux, y)
    return losses # Placeholder