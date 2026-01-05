import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import pandas as pd
from typing import Tuple, List
import cv2 # Cần resize ảnh để tránh lỗi lệch size

# --- 1. DATASET SINGLE TASK (CŨ) ---
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
            'normal': 'normal_map_new_crop.exr.npy',
            'depth': 'depth_map_new_crop.exr.npy'
        }
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        self.labels_list = [self.id_to_label[row['id']] for _, row in self.df.iterrows()]
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
            if image_data.ndim == 2:
                image_data = np.expand_dims(image_data, axis=-1)
                image_data = np.repeat(image_data, 3, axis=-1)
            elif image_data.ndim == 3 and image_data.shape[0] == 3:
                image_data = image_data.transpose(1, 2, 0)
            elif image_data.ndim == 3 and image_data.shape[2] == 1:
                image_data = np.repeat(image_data, 3, axis=-1)
            image_data = image_data.astype(np.float32)
        except Exception:
            image_data = np.zeros((112, 112, 3), dtype=np.float32)

        label_id = self.id_to_label[id_val]
        labels = torch.tensor([label_id, int(row.get('Gender', 0)), int(row.get('Spectacles', 0)), 
                               int(row.get('Facial_Hair', 0)), int(row.get('Pose', 0)), int(row.get('Emotion', 0))], dtype=torch.long)

        if self.transform:
            augmented = self.transform(image=image_data)
            image_data = augmented['image']

        X = torch.from_numpy(image_data).permute(2, 0, 1) if isinstance(image_data, np.ndarray) else image_data
        return X, labels

# --- 2. SAMPLER ---
class UniqueIdBatchSampler(Sampler):
    def __init__(self, labels: List[int], batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.unique_labels = list(set(labels))
        self.label_indices = {label: [] for label in self.unique_labels}
        for idx, label in enumerate(self.labels):
            self.label_indices[label].append(idx)

    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            batch_labels = np.random.choice(self.unique_labels, size=self.batch_size, replace=False)
            batch_indices = [np.random.choice(self.label_indices[label]) for label in batch_labels]
            yield batch_indices

    def __len__(self): return len(self.labels) // self.batch_size

# --- 3. DATASET FUSION (FIX LỖI SIZE) ---
class ConcatPhotometricDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.file_map = {'albedo': 'albedo_map_new_crop.exr.npy', 'normal': 'normal_map_new_crop.exr.npy'}
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        self.labels_list = [self.id_to_label[row['id']] for _, row in self.df.iterrows()]
        self.weightclass = {}

    def __len__(self): return len(self.df)
    def get_labels(self): return self.labels_list

    def __load_npy(self, path):
        try:
            img = np.load(path)
            if img.ndim == 2: img = np.repeat(np.expand_dims(img, axis=-1), 3, axis=-1)
            elif img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1, 2, 0)
            return img.astype(np.float32)
        except: return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session = str(row['session'])
        
        path_alb = os.path.join(self.root_dir, str(id_val), session, self.file_map['albedo'])
        path_nor = os.path.join(self.root_dir, str(id_val), session, self.file_map['normal'])
        
        img_alb = self.__load_npy(path_alb)
        img_nor = self.__load_npy(path_nor)
        
        # FIX LỖI SIZE
        if img_alb is None and img_nor is None:
            img_alb = np.zeros((112, 112, 3), dtype=np.float32)
            img_nor = np.zeros((112, 112, 3), dtype=np.float32)
        elif img_alb is None: img_alb = np.zeros_like(img_nor)
        elif img_nor is None: img_nor = np.zeros_like(img_alb)
        elif img_alb.shape[:2] != img_nor.shape[:2]:
            h, w = img_alb.shape[:2]
            img_nor = cv2.resize(img_nor, (w, h))
            if img_nor.ndim == 2: img_nor = np.repeat(np.expand_dims(img_nor, axis=-1), 3, axis=-1)

        label_id = self.id_to_label[id_val]
        labels = torch.tensor([label_id, int(row.get('Gender', 0)), int(row.get('Spectacles', 0)), 
                               int(row.get('Facial_Hair', 0)), int(row.get('Pose', 0)), int(row.get('Emotion', 0))], dtype=torch.long)
        
        if self.transform:
            augmented = self.transform(image=img_alb, image2=img_nor)
            img_alb, img_nor = augmented['image'], augmented['image2']
            
        if isinstance(img_alb, np.ndarray): img_alb = torch.from_numpy(img_alb).permute(2, 0, 1)
        if isinstance(img_nor, np.ndarray): img_nor = torch.from_numpy(img_nor).permute(2, 0, 1)
            
        return img_alb, img_nor, labels

# --- 4. DATA LOADER FACTORIES ---
def create_multitask_datafetcher(config, train_transform, test_transform):
    dataset_dir = config['dataset_dir']
    train_ds = PhotometricDataset(os.path.join(dataset_dir, 'train_split.csv'), dataset_dir, train_transform, config.get('type', 'albedo'))
    test_ds = PhotometricDataset(os.path.join(dataset_dir, 'probe_split.csv'), dataset_dir, test_transform, config.get('type', 'albedo'))
    train_dl = DataLoader(train_ds, batch_sampler=UniqueIdBatchSampler(train_ds.get_labels(), config['batch_size']), num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl, train_ds.weightclass

def create_concat_multitask_datafetcher(config, train_transform, test_transform):
    dataset_dir = config['dataset_dir']
    train_ds = ConcatPhotometricDataset(os.path.join(dataset_dir, 'train_split.csv'), dataset_dir, train_transform)
    test_ds = ConcatPhotometricDataset(os.path.join(dataset_dir, 'probe_split.csv'), dataset_dir, test_transform)
    train_dl = DataLoader(train_ds, batch_sampler=UniqueIdBatchSampler(train_ds.get_labels(), config['batch_size']), num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, test_dl, train_ds.weightclass