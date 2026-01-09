import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import pandas as pd
import cv2
from typing import Tuple, List

# --- 1. DATASET CHO SINGLE TASK (Giữ nguyên của bạn) ---
class PhotometricDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, type_mode='albedo'):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.type_mode = type_mode
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normalmap': 'normal_map_new_crop.exr.npy',
            'depthmap': 'depth_map_new_crop.exr.npy',
            'normal': 'normal_map_new_crop.exr.npy', # Alias
            'depth': 'depth_map_new_crop.exr.npy'     # Alias
        }
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        
        self.labels_list = []
        for _, row in self.df.iterrows():
            self.labels_list.append(self.id_to_label[row['id']])
            
        self.weightclass = {} 

    def __len__(self): return len(self.df)
    def get_labels(self): return self.labels_list

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session_name = row['session']
        file_suffix = self.file_map.get(self.type_mode, 'albedo_map_new_crop.exr.npy')
        file_path = os.path.join(self.root_dir, str(id_val), str(session_name), file_suffix)

        try:
            image_data = np.load(file_path)
            if image_data.ndim == 3 and image_data.shape[0] == 3:
                image_data = image_data.transpose(1, 2, 0)
            image_data = image_data.astype(np.float32)
        except Exception:
            image_data = np.zeros((112, 112, 3), dtype=np.float32)

        label_id = self.id_to_label[id_val]
        # Lấy nhãn phụ an toàn
        def get_lbl(key): return int(row.get(key, 0))
        
        labels = torch.tensor([
            label_id, get_lbl('Gender'), get_lbl('Spectacles'), 
            get_lbl('Facial_Hair'), get_lbl('Pose'), get_lbl('Emotion')
        ], dtype=torch.long)

        if self.transform:
            augmented = self.transform(image=image_data)
            image_data = augmented['image']

        if isinstance(image_data, np.ndarray):
            X = torch.from_numpy(image_data).permute(2, 0, 1)
        else:
            X = image_data

        return X, labels

# --- 2. DATASET 3 NHÁNH (BỔ SUNG QUAN TRỌNG) ---
class ThreeBranchDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normal': 'normal_map_new_crop.exr.npy',
            'depth': 'depth_map_new_crop.exr.npy' 
        }
        
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        self.labels_list = [self.id_to_label[row['id']] for _, row in self.df.iterrows()]

    def __len__(self): return len(self.df)
    def get_labels(self): return self.labels_list

    def __load_npy(self, path):
        try:
            if not os.path.exists(path): return None
            img = np.load(path)
            if img.ndim == 2: 
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] == 3: 
                img = img.transpose(1, 2, 0)
            return img.astype(np.float32)
        except: return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session = str(row['session'])
        
        # Đường dẫn 3 file
        p_alb = os.path.join(self.root_dir, str(id_val), session, self.file_map['albedo'])
        p_nor = os.path.join(self.root_dir, str(id_val), session, self.file_map['normal'])
        p_dep = os.path.join(self.root_dir, str(id_val), session, self.file_map['depth'])
        
        i_alb = self.__load_npy(p_alb)
        i_nor = self.__load_npy(p_nor)
        i_dep = self.__load_npy(p_dep)
        
        # Fallback
        if i_alb is None: i_alb = np.zeros((112, 112, 3), dtype=np.float32)
        h, w = i_alb.shape[:2]
        
        def fix_img(img):
            if img is None: return np.zeros((h, w, 3), dtype=np.float32)
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h)) # Resize cho khớp
            return img

        i_nor = fix_img(i_nor)
        i_dep = fix_img(i_dep)

        # Labels
        label_id = self.id_to_label[id_val]
        def get_lbl(key): return int(row.get(key, 0))
        
        labels = torch.tensor([
            label_id, get_lbl('Gender'), get_lbl('Spectacles'), 
            get_lbl('Facial_Hair'), get_lbl('Pose'), get_lbl('Emotion')
        ], dtype=torch.long)

        # Augmentation 3 nhánh
        if self.transform:
            res = self.transform(image=i_alb, image0=i_nor, image1=i_dep)
            i_alb, i_nor, i_dep = res['image'], res['image0'], res['image1']
            
        # To Tensor
        to_ts = lambda x: torch.from_numpy(x).permute(2, 0, 1) if isinstance(x, np.ndarray) else x
        
        return to_ts(i_alb), to_ts(i_nor), to_ts(i_dep), labels

# --- 3. CUSTOM SAMPLER (Giữ nguyên) ---
class UniqueIdBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.unique_labels = list(set(labels))
        self.label_indices = {l: [] for l in self.unique_labels}
        for idx, l in enumerate(self.labels): self.label_indices[l].append(idx)

    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            batch_labels = np.random.choice(self.unique_labels, size=self.batch_size, replace=False)
            batch_indices = [np.random.choice(self.label_indices[l]) for l in batch_labels]
            yield batch_indices

    def __len__(self): return len(self.labels) // self.batch_size

# --- 4. DATA LOADER FACTORIES ---
def create_multitask_datafetcher(config, train_transform, test_transform):
    # Hàm này dùng cho PhotometricDataset (cũ)
    dataset_dir = config['dataset_dir']
    train_ds = PhotometricDataset(os.path.join(dataset_dir, 'train_split.csv'), dataset_dir, train_transform, config.get('type', 'albedo'))
    test_ds = PhotometricDataset(os.path.join(dataset_dir, 'probe_split.csv'), dataset_dir, test_transform, config.get('type', 'albedo'))
    
    # Mặc định dùng Sampler
    train_dl = DataLoader(train_ds, batch_sampler=UniqueIdBatchSampler(train_ds.get_labels(), config['batch_size']), num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    return train_dl, test_dl, train_ds.weightclass

def create_three_branch_loader(conf, train_tf, test_tf, use_sampler=True):
    # Hàm này dùng cho ThreeBranchDataset (Mới, dùng cho run_single_task.py)
    train_csv = os.path.join(conf['dataset_dir'], 'train_split.csv')
    test_csv = os.path.join(conf['dataset_dir'], 'gallery_split.csv') 
    if not os.path.exists(test_csv): test_csv = os.path.join(conf['dataset_dir'], 'probe_split.csv')

    train_ds = ThreeBranchDataset(train_csv, conf['dataset_dir'], train_tf)
    test_ds = ThreeBranchDataset(test_csv, conf['dataset_dir'], test_tf)
    
    if use_sampler:
        print("Dataloader: PK Sampler ACTIVATED")
        train_loader = DataLoader(train_ds, batch_sampler=UniqueIdBatchSampler(train_ds.get_labels(), conf['batch_size']), num_workers=2)
    else:
        print("Dataloader: Random Shuffle ACTIVATED")
        train_loader = DataLoader(train_ds, batch_size=conf['batch_size'], shuffle=True, num_workers=2)
    
    test_loader = DataLoader(test_ds, batch_size=conf['batch_size'], shuffle=False, num_workers=2)
    return train_loader, test_loader