import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import traceback

def compute_auc(dataloader, model, device):
    model.eval()
    embeddings_list = []

    print("   -> [AUC] Đang thu thập vector đặc trưng...")
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3: # FUSION
                x1, x2, y = batch
                x1, x2 = x1.to(device), x2.to(device)
                emb = model.get_embedding(x1, x2)
                # Đưa về CPU ngay lập tức
                embeddings_list.append((y[:, 0].cpu(), emb.cpu())) 
            else: # SINGLE
                img, y = batch
                img = img.to(device)
                out = model.get_result(img)
                embeddings_list.append((y[:, 0].cpu(), out[0].cpu()))

    auc_scores = {'id_cosine': 0.0, 'id_euclidean': 0.0}
    
    try:
        if len(embeddings_list) > 0:
            print("   -> [AUC] Đang tính toán ma trận tương đồng trên CPU...")
            # Nối dữ liệu
            ids = torch.cat([x[0] for x in embeddings_list], dim=0)
            feats = torch.cat([x[1] for x in embeddings_list], dim=0)
            
            # Chuẩn hóa
            feats = F.normalize(feats, p=2, dim=1)

            # Tính ma trận (CPU)
            cosine_sim = torch.mm(feats, feats.t())
            euclidean_dist = torch.cdist(feats, feats, p=2)
            
            # Ground Truth
            labels = (ids.unsqueeze(1) == ids.unsqueeze(0)).int()
            triu_indices = torch.triu(torch.ones_like(labels), diagonal=1) == 1
            
            y_true = labels[triu_indices].numpy()
            y_score_cos = cosine_sim[triu_indices].numpy()
            y_score_euc = -euclidean_dist[triu_indices].numpy()
            
            # Check dữ liệu trước khi tính
            unique_classes = np.unique(y_true)
            if len(unique_classes) < 2:
                print(f" [AUC] Cảnh báo: Tập dữ liệu chỉ có 1 loại nhãn (Toàn 0 hoặc toàn 1). Unique: {unique_classes}")
                print("   -> Không thể tính AUC. Trả về 0.0")
            else:
                auc_scores['id_cosine'] = roc_auc_score(y_true, y_score_cos)
                auc_scores['id_euclidean'] = roc_auc_score(y_true, y_score_euc)
                print(f"   -> [AUC] Thành công! Cosine: {auc_scores['id_cosine']:.4f}")
        else:
            print("[AUC] Không thu thập được embeddings nào!")
                
    except Exception as e:
        print(f" [AUC] Lỗi tính toán: {e}")
        traceback.print_exc()

    return auc_scores