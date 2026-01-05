import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def compute_auc(dataloader, model, device):
    model.eval()
    embeddings_list = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3: # FUSION
                x1, x2, y = batch
                x1, x2 = x1.to(device), x2.to(device)
                emb = model.get_embedding(x1, x2)
                embeddings_list.append((y[:, 0], emb))
            else: # SINGLE
                img, y = batch
                img = img.to(device)
                out = model.get_result(img)
                embeddings_list.append((y[:, 0], out[0])) # out[0] là id embedding

    auc_scores = {'id_cosine': 0.0}
    try:
        if len(embeddings_list) > 0:
            ids = torch.cat([x[0] for x in embeddings_list], dim=0).to(device)
            feats = torch.cat([x[1] for x in embeddings_list], dim=0)
            feats = F.normalize(feats, p=2, dim=1)

            cosine_sim = torch.mm(feats, feats.t())
            
            # Ground Truth Matrix
            labels = (ids.unsqueeze(1) == ids.unsqueeze(0)).int()
            triu_indices = torch.triu(torch.ones_like(labels), diagonal=1) == 1
            
            y_true = labels[triu_indices].cpu().numpy()
            y_score = cosine_sim[triu_indices].cpu().numpy()
            
            auc_scores['id_cosine'] = roc_auc_score(y_true, y_score)
    except Exception as e:
        print(f"AUC Error: {e}")

    return auc_scores