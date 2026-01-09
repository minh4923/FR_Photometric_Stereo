import os
import csv
import sys
from datetime import datetime

class ExperimentManager:
    def __init__(self, conf):
        """
        Khoi tao trinh quan ly thi nghiem.
        conf: Dictionary chua cau hinh (bat buoc phai co 'dataset_dir', 'step', 'note')
        """
        self.conf = conf
        
        # 1. TAO TEN THU MUC
        exp_name = f"{conf['step']:02d}_{conf['note']}"
        self.exp_dir = os.path.join(conf['dataset_dir'], 'experiments', exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        
        # 2. TAO FOLDER
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 3. FILE PATHS
        self.txt_path = os.path.join(self.log_dir, 'console_log.txt')
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        
        # 4. GHI LOG MO DAU
        self.log_text(f"KHOI TAO THI NGHIEM: {exp_name}")
        self.log_text(f"Luu tru tai: {self.exp_dir}")
        self.log_text(f"Thoi gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_text("-" * 50)
        self.log_text("CAU HINH (CONFIGURATION):")
        for k, v in conf.items():
            self.log_text(f"  - {k}: {v}")
        self.log_text("-" * 50)
        
        # Bien co de kiem tra xem da ghi header CSV chua
        self.csv_initialized = False
        self.csv_fieldnames = []

    def log_text(self, message):
        """In ra man hinh VA luu vao file txt"""
        print(message)
        try:
            with open(self.txt_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        except Exception as e:
            print(f"Loi ghi log: {e}")

    def log_metrics(self, epoch, metrics_dict):
        """
        Luu DYNAMIC cac con so vao file CSV.
        Tu dong tao cot dua tren keys cua metrics_dict o lan chay dau tien.
        """
        # Chuan bi du lieu row
        row_data = {'epoch': epoch}
        
        # Format so lieu (lay 4 so le)
        for k, v in metrics_dict.items():
            if isinstance(v, (int, float)):
                row_data[k] = f"{v:.4f}"
            else:
                row_data[k] = v

        # Neu chua khoi tao CSV, dung keys cua row_data lam header
        if not self.csv_initialized:
            # Sap xep de 'epoch' luon dung dau
            self.csv_fieldnames = ['epoch'] + sorted([k for k in row_data.keys() if k != 'epoch'])
            
            # Ghi header neu file chua ton tai hoac rong
            write_header = not os.path.exists(self.csv_path) or os.stat(self.csv_path).st_size == 0
            
            if write_header:
                try:
                    with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                        writer.writeheader()
                except Exception as e:
                    print(f"Loi khoi tao CSV: {e}")
            
            self.csv_initialized = True

        # Ghi du lieu
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
                # Chi ghi nhung cot da dinh nghia trong header
                row_to_write = {k: v for k, v in row_data.items() if k in self.csv_fieldnames}
                writer.writerow(row_to_write)
        except Exception as e:
            print(f"Loi ghi CSV: {e}")
            
    def get_checkpoint_path(self, name="best_model.pth"):
        return os.path.join(self.ckpt_dir, name)