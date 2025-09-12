import torch
from torchvision import transforms as trans
from pathlib import Path
from backbones import get_model


def l2_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=eps)
    return x / norm


class face_learner(object):
    def __init__(self, conf):
        self.threshold = conf.threshold

        self.model = get_model(
            conf.network,
            fp16=getattr(conf, "fp16", False),
            num_features=getattr(conf, "embedding_size", 512),
        ).to(conf.device)
        self.model.eval()

        # Preload FFM.B if exists
        try:
            if hasattr(self.model, "ffm") and hasattr(self.model.ffm, "B"):
                b_path = Path(getattr(conf, "output", ".")) / "ffm_B.pt"
                if b_path.is_file():
                    B = torch.load(b_path, map_location="cpu")
                    with torch.no_grad():
                        self.model.ffm.B.copy_(B.to(dtype=torch.float32, device=self.model.ffm.B.device))
        except Exception:
            pass

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=True):
        base_dir = getattr(conf, "output", None) or getattr(conf, "model_path", None)
        if base_dir is None:
            raise ValueError("Missing checkpoint directory: need conf.output or conf.model_path")
        model_path = Path(base_dir) / fixed_str
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        raw = torch.load(model_path, map_location=conf.device)
        if isinstance(raw, dict):
            if "model_state_dict" in raw:
                state_dict = raw["model_state_dict"]
            elif "state_dict" in raw:
                state_dict = raw["state_dict"]
            else:
                state_dict = raw
        else:
            raise ValueError("Invalid checkpoint format: expected dict.")

        is_parallel = all(k.startswith("module.") for k in state_dict.keys())
        clean_state_dict = {(k[7:] if is_parallel else k): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict, strict=False)

        # Save FFM.B for future reuse
        try:
            if hasattr(self.model, "ffm") and hasattr(self.model.ffm, "B"):
                b_path = Path(getattr(conf, "output", ".")) / "ffm_B.pt"
                with torch.no_grad():
                    torch.save(self.model.ffm.B.detach().cpu(), b_path)
        except Exception:
            pass

    def _embed_faces(self, conf, faces, tta: bool = False) -> torch.Tensor:
        if len(faces) == 0:
            return torch.empty(0, getattr(conf, "embedding_size", 512), device=conf.device)

        embs = []
        with torch.no_grad():
            for img in faces:
                try:
                    if tta:
                        mirror = trans.functional.hflip(img)
                        inp = conf.test_transform(img).to(conf.device).unsqueeze(0)
                        inp_m = conf.test_transform(mirror).to(conf.device).unsqueeze(0)
                        emb = self.model(inp) + self.model(inp_m)
                    else:
                        inp = conf.test_transform(img).to(conf.device).unsqueeze(0)
                        emb = self.model(inp)

                    # fallback to fp32 if necessary
                    if torch.isnan(emb).any() or torch.isinf(emb).any():
                        prev = getattr(self.model, "fp16", False)
                        try:
                            self.model.fp16 = False
                            emb = self.model(inp.float())
                            if tta:
                                emb = emb + self.model(inp_m.float())
                        finally:
                            self.model.fp16 = prev

                    if torch.isnan(emb).any() or torch.isinf(emb).any():
                        continue

                    emb = l2_norm(emb, dim=1)
                    if torch.isnan(emb).any() or float(torch.norm(emb, p=2)) < 1e-6:
                        continue

                    embs.append(emb)
                except Exception:
                    continue

        if len(embs) == 0:
            return torch.empty(0, getattr(conf, "embedding_size", 512), device=conf.device)
        return torch.cat(embs, dim=0)

    def infer(self, conf, faces, target_embs, tta: bool = False):
        if isinstance(target_embs, torch.Tensor):
            if target_embs.device != conf.device:
                target_embs = target_embs.to(conf.device)
            target_embs = target_embs.float()

        source_embs = self._embed_faces(conf, faces, tta=tta)
        if source_embs.numel() == 0 or (not isinstance(target_embs, torch.Tensor) or target_embs.numel() == 0):
            return torch.tensor([], device=conf.device, dtype=torch.long), torch.tensor([], device=conf.device)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(diff.pow(2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)

        # apply threshold
        min_idx = min_idx.clone()
        min_idx[minimum > self.threshold] = -1
        return min_idx, minimum
