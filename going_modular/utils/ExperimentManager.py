
import os
import csv
import sys
from datetime import datetime

class ExperimentManager:
    def __init__(self, conf):
        """
        Khởi tạo trình quản lý thí nghiệm.
        conf: Dictionary chứa cấu hình (bắt buộc phải có 'dataset_dir', 'step', 'note')
        """
        self.conf = conf
        
        # 1. TẠO TÊN THƯ MỤC TỰ ĐỘNG
        # Ví dụ: 01_Baseline_2Branch_ResNet
        exp_name = f"{conf['step']:02d}_{conf['note']}"
        
        # Đường dẫn: .../experiments/01_...
        self.exp_dir = os.path.join(conf['dataset_dir'], 'experiments', exp_name)
        self.ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
        self.log_dir = os.path.join(self.exp_dir, 'logs')
        
        # 2. TẠO FOLDER (Nếu chưa có)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 3. ĐƯỜNG DẪN FILE LOG
        self.txt_path = os.path.join(self.log_dir, 'console_log.txt')
        self.csv_path = os.path.join(self.log_dir, 'metrics.csv')
        
        # 4. GHI LOG MỞ ĐẦU
        self.log_text(f"KHỞI TẠO THÍ NGHIỆM: {exp_name}")
        self.log_text(f"Lưu trữ tại: {self.exp_dir}")
        self.log_text(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_text("-" * 50)
        self.log_text("CẤU HÌNH (CONFIGURATION):")
        for k, v in conf.items():
            self.log_text(f"  - {k}: {v}")
        self.log_text("-" * 50)
        
        # 5. KHỞI TẠO FILE CSV (Ghi tiêu đề cột)
        # Tự động điều chỉnh cột dựa trên các metric bạn quan tâm
        self.csv_headers = ['epoch', 'train_loss', 'val_loss', 'val_auc_id_cosine']
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

    def log_text(self, message):
        """
        In ra màn hình VÀ lưu vào file txt cùng lúc.
        Giúp bạn không bị mất log khi Colab disconnect.
        """
        print(message) # In ra console Colab
        try:
            with open(self.txt_path, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        except Exception as e:
            print(f"⚠️ Không ghi được log file: {e}")

    def log_metrics(self, epoch, metrics_dict):
        """
        Lưu các con số vào file CSV để vẽ biểu đồ.
        metrics_dict: {'train_loss': 0.5, 'val_auc': 0.9...}
        """
        # Tạo dòng dữ liệu theo đúng thứ tự headers
        row = [epoch]
        row.append(f"{metrics_dict.get('loss', 0):.4f}")           # train_loss
        row.append(f"{metrics_dict.get('val_loss', 0):.4f}")       # val_loss
        row.append(f"{metrics_dict.get('auc_id_cosine', 0):.4f}")  # val_auc
        
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            print(f" Không ghi được CSV: {e}")
            
    def get_checkpoint_path(self, name="best_model.pth"):
        return os.path.join(self.ckpt_dir, name)