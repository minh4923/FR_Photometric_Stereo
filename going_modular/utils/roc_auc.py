# going_modular/utils/roc_auc.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def compute_auc(dataloader, model, device):
    model.eval()

    results = {
        'gender': {'true': [], 'prob': []},
        'spectacles': {'true': [], 'prob': []},
        'facial_hair': {'true': [], 'prob': []},
        'pose': {'true': [], 'prob': []},
        'emotion': {'true': [], 'prob': []}
    }

    embeddings_list = []

    with torch.no_grad():
        for batch in dataloader:
            images, y = batch
            id_label = y[:, 0]

            # Lấy label attributes
            attr_labels = {
                'gender': y[:, 1],
                'spectacles': y[:, 2],
                'facial_hair': y[:, 3],
                'pose': y[:, 4],
                'emotion': y[:, 5]
            }

            images = images.to(device)
            output = model.get_result(images)
            x_id, x_gender, x_pose, x_emotion, x_facial_hair, x_spectacles = output

            embeddings_list.append((id_label, x_id))

            predictions_map = {
                'gender': x_gender,
                'spectacles': x_spectacles,
                'facial_hair': x_facial_hair,
                'pose': x_pose,
                'emotion': x_emotion
            }

            for attr_name, logits in predictions_map.items():
                y_true = attr_labels[attr_name].to(device)
                probs = torch.softmax(logits, dim=1)

                results[attr_name]['true'].append(y_true.cpu().numpy())
                results[attr_name]['prob'].append(probs.cpu().numpy())

    # --- TÍNH TOÁN AUC ---
    auc_scores = {}

    for attr_name, data in results.items():
        y_true_all = np.concatenate(data['true'])
        y_prob_all = np.concatenate(data['prob'])

        try:
            # Kiểm tra số lượng lớp
            n_classes = len(np.unique(y_true_all))
            if n_classes < 2:
                auc_scores[attr_name] = 0.5
            else:
                # --- FIX QUAN TRỌNG CHO BINARY CLASSIFICATION ---
                # Nếu chỉ có 2 lớp (vd: Nam/Nữ), ta chỉ đưa vào xác suất của lớp 1
                if y_prob_all.shape[1] == 2:
                    # Lấy cột thứ 2 (index 1) làm xác suất positive
                    score = roc_auc_score(y_true_all, y_prob_all[:, 1])
                else:
                    # Nếu > 2 lớp (Emotion) thì dùng ovr
                    score = roc_auc_score(y_true_all, y_prob_all, multi_class='ovr')

                auc_scores[attr_name] = score

        except Exception as e:
            # IN LỖI RA MÀN HÌNH ĐỂ DEBUG
            print(f" Lỗi tính AUC '{attr_name}': {e}")
            auc_scores[attr_name] = 0.0

    # --- TÍNH ID AUC (Giữ nguyên) ---
    try:
        all_ids = torch.cat([x[0] for x in embeddings_list], dim=0)
        all_embeddings = torch.cat([x[1] for x in embeddings_list], dim=0)
        all_embeddings_norm = F.normalize(all_embeddings, p=2, dim=1)

        cosine_sim = torch.mm(all_embeddings_norm, all_embeddings_norm.t())
        euclidean_dist = torch.cdist(all_embeddings_norm, all_embeddings_norm, p=2)

        all_ids = all_ids.to(device)
        labels = (all_ids.unsqueeze(1) == all_ids.unsqueeze(0)).int()
        triu_indices = torch.triu(torch.ones_like(labels), diagonal=1) == 1

        final_labels = labels[triu_indices].cpu().numpy()
        final_cosine = cosine_sim[triu_indices].cpu().numpy()
        final_euclidean = -euclidean_dist[triu_indices].cpu().numpy()

        auc_scores['id_cosine'] = roc_auc_score(final_labels, final_cosine)
        auc_scores['id_euclidean'] = roc_auc_score(final_labels, final_euclidean)
    except Exception as e:
        print(f" Lỗi tính AUC ID: {e}")
        auc_scores['id_cosine'] = 0.0
        auc_scores['id_euclidean'] = 0.0

    return auc_scores