# configs/infer_config.py

import torch
from configs.base import config
from pathlib import Path
from torchvision import transforms as trans

config.network = "r50"
config.output = "work_dirs/ms1mv3_r50_onegpu"

config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Embedding size
config.embedding_size = 512

config.model_file = 'model.pt'
config.cpu_model_file = 'model.pt'

config.fp16 = True
config.threshold = 1.68
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
