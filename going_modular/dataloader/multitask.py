# going_modular/dataloader/multitask.py
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import pandas as pd
from typing import Tuple, List

# --- 1. DATASET CHO SINGLE TASK ---
class PhotometricDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, type_mode='albedo'):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.type_mode = type_mode

        # Map tên loại -> tên file .npy
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normalmap': 'normal_map_new_crop.exr.npy',
            'depthmap': 'depth_map_new_crop.exr.npy',
            'normal': 'normal_map_new_crop.exr.npy',
            'depth': 'depth_map_new_crop.exr.npy'
        }

        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        
        # --- CẢI TIẾN: Lưu danh sách label để dùng cho Sampler ---
        self.labels_list = []
        for _, row in self.df.iterrows():
            self.labels_list.append(self.id_to_label[row['id']])
            
        self.weightclass = {} # Có thể tính class weight nếu cần dùng CrossEntropyLoss

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.labels_list

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session_name = row['session']

        file_suffix = self.file_map.get(self.type_mode, 'albedo_map_new_crop.exr.npy')
        file_path = os.path.join(self.root_dir, str(id_val), str(session_name), file_suffix)

        try:
            image_data = np.load(file_path)
            # Chuyển về HWC nếu đang là CHW hoặc định dạng khác
            if image_data.ndim == 3 and image_data.shape[0] == 3:
                image_data = image_data.transpose(1, 2, 0)
            image_data = image_data.astype(np.float32)
        except Exception:
            # Fallback nếu lỗi đọc file
            image_data = np.zeros((112, 112, 3), dtype=np.float32)

        label_id = self.id_to_label[id_val]
        
        # Lấy label phụ
        gender = int(row.get('Gender', 0))
        spec = int(row.get('Spectacles', 0))
        hair = int(row.get('Facial_Hair', 0))
        pose = int(row.get('Pose', 0))
        emo = int(row.get('Emotion', 0))

        labels = torch.tensor([label_id, gender, spec, hair, pose, emo], dtype=torch.long)

        if self.transform:
            augmented = self.transform(image=image_data)
            image_data = augmented['image']

        if isinstance(image_data, np.ndarray):
            X = torch.from_numpy(image_data).permute(2, 0, 1)
        else:
            X = image_data

        return X, labels

# --- 2. CUSTOM SAMPLER: ĐẢM BẢO UNIQUE ID TRONG BATCH ---
class UniqueIdBatchSampler(Sampler):
    """
    Sampler này đảm bảo trong mỗi batch, mỗi ID chỉ xuất hiện tối đa 1 lần.
    Giúp cân bằng việc lấy mẫu giữa các ID có nhiều ảnh (148) và ít ảnh (5).
    """
    def __init__(self, labels: List[int], batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.unique_labels = list(set(labels))
        
        # Tạo map: Label -> Danh sách các index của ảnh thuộc label đó
        self.label_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_indices:
                self.label_indices[label] = []
            self.label_indices[label].append(idx)

        # Kiểm tra batch size
        if self.batch_size > len(self.unique_labels):
            raise ValueError(f"Batch size ({self.batch_size}) lớn hơn tổng số ID ({len(self.unique_labels)}). Không thể tạo unique batch.")

    def __iter__(self):
        # Tính số lượng batch trong 1 epoch (ước lượng)
        n_batches = len(self.labels) // self.batch_size
        
        for _ in range(n_batches):
            # Bước 1: Chọn ngẫu nhiên 'batch_size' ID khác nhau (replace=False)
            batch_labels = np.random.choice(self.unique_labels, size=self.batch_size, replace=False)
            
            batch_indices = []
            for label in batch_labels:
                # Bước 2: Với mỗi ID đã chọn, lấy ngẫu nhiên 1 ảnh thuộc ID đó
                # Cách này giúp ID có 148 ảnh và ID có 5 ảnh đều có cơ hội xuất hiện ngang nhau
                img_idx = np.random.choice(self.label_indices[label])
                batch_indices.append(img_idx)
            
            yield batch_indices

    def __len__(self):
        return len(self.labels) // self.batch_size

# --- 3. HÀM TẠO DATALOADER ---
def create_multitask_datafetcher(config, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, dict]:
    dataset_dir = config['dataset_dir']
    train_csv = os.path.join(dataset_dir, 'train_split.csv')
    test_csv = os.path.join(dataset_dir, 'probe_split.csv')
    batch_size = config['batch_size']

    type_mode = config.get('type', 'albedo')
    print(f"--- Khởi tạo Single Task: {type_mode} ---")

    train_dataset = PhotometricDataset(train_csv, dataset_dir, train_transform, type_mode)
    test_dataset = PhotometricDataset(test_csv, dataset_dir, test_transform, type_mode)

    print(f"-> Train: {len(train_dataset)} ảnh | Test: {len(test_dataset)} ảnh")

    # --- CẤU HÌNH SAMPLER CHO TRAIN ---
    # Sử dụng UniqueIdBatchSampler cho tập Train để tránh trùng ID trong batch
    train_sampler = UniqueIdBatchSampler(train_dataset.get_labels(), batch_size)

    # Lưu ý: Khi dùng batch_sampler, tham số batch_size trong DataLoader phải bỏ qua (hoặc để 1),
    # shuffle phải là False (vì sampler đã shuffle rồi).
    train_dl = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, # Dùng custom sampler ở đây
        num_workers=2, 
        pin_memory=True
    )

    # Tập Test không cần sampler phức tạp, chỉ cần shuffle=False để đánh giá tuần tự
    test_dl = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    return train_dl, test_dl, train_dataset.weightclass

def create_concatv2_multitask_datafetcher(config, train_transform, test_transform):
    raise NotImplementedError("Fusion Task")

# --- PHẦN THÊM VÀO CUỐI FILE going_modular/dataloader/multitask.py ---

class ConcatPhotometricDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        # Mapping file
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normal': 'normal_map_new_crop.exr.npy'
        }
        
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        
        # Lưu label list để dùng cho Sampler
        self.labels_list = []
        for _, row in self.df.iterrows():
            self.labels_list.append(self.id_to_label[row['id']])
            
        self.weightclass = {}

    def __len__(self):
        return len(self.df)
    
    def get_labels(self):
        return self.labels_list

    def __load_npy(self, path):
        try:
            img = np.load(path)
            # Fix lỗi Depth/Normal 2D nếu có
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            img = img.astype(np.float32)
            return img
        except:
            return np.zeros((112, 112, 3), dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session = str(row['session'])
        
        # Lấy đường dẫn cả 2 loại
        path_albedo = os.path.join(self.root_dir, str(id_val), session, self.file_map['albedo'])
        path_normal = os.path.join(self.root_dir, str(id_val), session, self.file_map['normal'])
        
        img_albedo = self.__load_npy(path_albedo)
        img_normal = self.__load_npy(path_normal)
        
        # Lấy nhãn
        label_id = self.id_to_label[id_val]
        gender = int(row.get('Gender', 0))
        spec = int(row.get('Spectacles', 0))
        hair = int(row.get('Facial_Hair', 0))
        pose = int(row.get('Pose', 0))
        emo = int(row.get('Emotion', 0))
        labels = torch.tensor([label_id, gender, spec, hair, pose, emo], dtype=torch.long)
        
        # Transform (Albumentations xử lý 2 ảnh cùng lúc)
        if self.transform:
            augmented = self.transform(image=img_albedo, image2=img_normal)
            img_albedo = augmented['image']
            img_normal = augmented['image2']
            
        # Convert to Tensor
        if isinstance(img_albedo, np.ndarray):
            img_albedo = torch.from_numpy(img_albedo).permute(2, 0, 1)
            img_normal = torch.from_numpy(img_normal).permute(2, 0, 1)
            
        return img_albedo, img_normal, labels

def create_concat_multitask_datafetcher(config, train_transform, test_transform):
    dataset_dir = config['dataset_dir']
    batch_size = config['batch_size']
    
    train_csv = os.path.join(dataset_dir, 'train_split.csv')
    test_csv = os.path.join(dataset_dir, 'probe_split.csv')
    
    print("--- Khởi tạo Dual-Stream Dataset (Albedo + Normal) ---")
    
    train_ds = ConcatPhotometricDataset(train_csv, dataset_dir, train_transform)
    test_ds = ConcatPhotometricDataset(test_csv, dataset_dir, test_transform)
    
    # Dùng lại UniqueIdBatchSampler thần thánh
    train_sampler = UniqueIdBatchSampler(train_ds.get_labels(), batch_size)
    
    train_dl = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_dl, test_dl, train_ds.weightclass