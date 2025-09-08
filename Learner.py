# Learner.py

import torch
from torchvision import transforms as trans
from pathlib import Path
from backbones import get_model

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class face_learner(object):
    def __init__(self, conf, inference=True):
        print(conf)
        self.threshold = conf.threshold

        print(f"Đang khởi tạo model: '{conf.network}'...")
        self.model = get_model(
            conf.network,
            fp16=conf.fp16,
            num_features=conf.embedding_size
        ).to(conf.device)
        print(f"Model '{conf.network}' đã được tạo thành công.")

        self.model.eval()

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=True):
        if not hasattr(conf, 'output') or conf.output is None:
            raise ValueError("Cấu hình lỗi: `config.output` phải được chỉ định và trỏ đến thư mục chứa model.")

        model_path = Path(conf.output) / fixed_str
        if not model_path.is_file():
            raise FileNotFoundError(f"Lỗi: Không tìm thấy file model tại đường dẫn: {model_path}")

        state_dict = torch.load(model_path, map_location=conf.device)
        clean_state_dict = {}
        is_parallel = all(k.startswith('module.') for k in state_dict.keys())

        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        for k, v in state_dict.items():
            name = k[7:] if is_parallel else k
            clean_state_dict[name] = v

        self.model.load_state_dict(clean_state_dict, strict=False)
        print(f"Đã tải thành công trọng số từ: {model_path}")

    def infer(self, conf, faces, target_embs, tta=False):
        with torch.no_grad():
            embs = []
            for img in faces:
                if tta:
                    mirror = trans.functional.hflip(img)
                    emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                    emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))
                else:
                    embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))

            if len(embs) == 0:
                return torch.tensor([]), torch.tensor([])

            source_embs = torch.cat(embs)
            diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            minimum, min_idx = torch.min(dist, dim=1)

            min_idx[minimum > self.threshold] = -1
            return min_idx, minimum
