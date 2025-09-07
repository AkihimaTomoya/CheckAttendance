# configs/inference_config.py

from configs.base import config
from pathlib import Path
from torchvision import transforms as trans

# -------------------- GHI ĐÈ CẤU HÌNH CHO INFERENCE --------------------

# --- Cấu hình MẶC ĐỊNH cho model ---
config.network = "r50"
config.output = "work_dirs/ms1mv3_r50_onegpu"

# Tên file trọng số bên trong thư mục `output`.
config.model_file = 'model.pt'
config.cpu_model_file = 'model.pt'

# --- Các cấu hình khác cho inference ---
config.fp16 = True
config.threshold = 1.4
config.tta = True
config.update_facebank = False
config.test_transform = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
config.show_score = True
config.face_limit = 10
config.min_face_size = 30
config.facebank_path = Path('facebank')