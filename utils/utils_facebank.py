from pathlib import Path
from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F
from PIL import Image

# =========================
# Facebank path helpers
# =========================
def _arch_dir(conf, model=None) -> Path:
    """Return the facebank directory root."""
    return Path(getattr(conf, "facebank_path", Path("facebank")))

def _facebank_file(conf, model=None) -> Path:
    """Return the path to the facebank file."""
    return _arch_dir(conf, model) / "facebank.pt"

# =========================
# Load / Save facebank
# =========================
def _safe_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """Convert to fp32, remove NaN/Inf, and L2-normalize along dim=1."""
    embeddings = embeddings.float()
    embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    if embeddings.numel() == 0:
        return embeddings
    return F.normalize(embeddings, p=2, dim=1)

def load_facebank(conf, model=None) -> Tuple[torch.Tensor, Dict[int, str]]:
    """
    Load facebank from a single canonical location.
    Supports two formats:
      - dict: {"embeddings": Tensor[P,D], "names": {idx: str}}
      - tuple: (embeddings, names_dict)  # legacy
    Returns (embeddings, names_dict); empty tensors/dicts if missing/invalid.
    """
    fb_path = _facebank_file(conf, model)
    if not fb_path.is_file():
        print(f"[facebank] Not found: {fb_path}. Returning empty.")
        return torch.empty(0), {}

    try:
        obj = torch.load(fb_path, map_location=getattr(conf, "device", "cpu"))
        if isinstance(obj, dict):
            embeddings = obj.get("embeddings", torch.empty(0))
            names_dict = obj.get("names", {})
        elif isinstance(obj, tuple) and len(obj) == 2:
            embeddings, names_dict = obj
        else:
            print(f"[facebank] Invalid format at {fb_path}. Returning empty.")
            return torch.empty(0), {}

        if isinstance(embeddings, torch.Tensor) and embeddings.numel() > 0:
            embeddings = _safe_normalize(embeddings)
        return embeddings, names_dict if isinstance(names_dict, dict) else {}
    except Exception as e:
        print(f"[facebank] Load error from {fb_path}: {e}")
        return torch.empty(0), {}

# =========================
# Embedding a single image
# =========================
def _embed_one_image(conf, model, detector, pil_img: Image.Image, tta: bool):
    """
    Detect + align (1 best face), return embedding [1, D] (L2-normalized) or None.
    Force FP32 during embedding to avoid NaNs.
    """
    try:
        res = detector.align_multi(pil_img, getattr(conf, "face_limit", 10), getattr(conf, "min_face_size", 30))
        if res is None:
            return None
        bboxes, faces = res
        if faces is None or len(faces) == 0:
            return None

        # choose top-score face (bboxes: [N,5] with score at -1)
        scores = bboxes[:, -1]
        idx = int(scores.argmax()) if hasattr(scores, "argmax") else 0
        face = faces[idx]

        prev_fp16 = getattr(model, "fp16", None)
        try:
            if hasattr(model, "fp16"):
                model.fp16 = False  # force FP32 temporarily

            if tta:
                from torchvision import transforms as trans
                mirror = trans.functional.hflip(face)
                inp = conf.test_transform(face).to(conf.device).unsqueeze(0).float()
                inp_m = conf.test_transform(mirror).to(conf.device).unsqueeze(0).float()
                with torch.inference_mode():
                    emb = model(inp) + model(inp_m)
            else:
                inp = conf.test_transform(face).to(conf.device).unsqueeze(0).float()
                with torch.inference_mode():
                    emb = model(inp)

            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            emb = F.normalize(emb, p=2, dim=1)
            if torch.isnan(emb).any() or not torch.isfinite(emb).all():
                return None
            if float(torch.norm(emb, p=2)) < 1e-6:
                return None
            return emb  # [1, D]
        finally:
            if prev_fp16 is not None and hasattr(model, "fp16"):
                try:
                    model.fp16 = prev_fp16
                except Exception:
                    pass
    except Exception as e:
        print(f"[facebank._embed_one_image] {e}")
        return None

# =========================
# Build facebank
# =========================
def prepare_facebank(conf, model, detector, tta: bool = True) -> Tuple[torch.Tensor, Dict[int, str]]:
    """
    Build facebank in a single directory:
    - Traverse: facebank/<PersonName>/*.{jpg,png,jpeg,bmp,webp}
    - For each person: embed valid faces -> mean -> L2-norm
    - Save {'embeddings', 'names'} to facebank/facebank.pt
    """
    arch_dir = _arch_dir(conf, model)
    arch_dir.mkdir(parents=True, exist_ok=True)

    identities = [p for p in arch_dir.iterdir() if p.is_dir()]
    identities.sort(key=lambda p: p.name.lower())

    agg_embeddings: List[torch.Tensor] = []
    names_dict: Dict[int, str] = {}
    pid = 0

    if len(identities) == 0:
        print(f"[facebank] No identities at: {arch_dir}")
        torch.save({"embeddings": torch.empty(0), "names": {}}, _facebank_file(conf, model))
        return torch.empty(0), {}

    for person_dir in identities:
        person_name = person_dir.name
        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            img_files += list(person_dir.glob(ext))
        if len(img_files) == 0:
            continue

        embs_person: List[torch.Tensor] = []
        for img_path in img_files:
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            emb = _embed_one_image(conf, model, detector, pil_img, tta=tta)
            if emb is not None:
                embs_person.append(emb)

        if len(embs_person) == 0:
            print(f"[facebank] Skip '{person_name}': no valid face.")
            continue

        # mean embedding -> L2-norm
        person_mat = torch.cat(embs_person, dim=0)  # [K, D]
        mean_emb = F.normalize(person_mat.mean(dim=0, keepdim=True), p=2, dim=1)  # [1, D]

        agg_embeddings.append(mean_emb)
        names_dict[pid] = person_name
        pid += 1

    if len(agg_embeddings) == 0:
        print(f"[facebank] No valid identities after processing.")
        torch.save({"embeddings": torch.empty(0), "names": {}}, _facebank_file(conf, model))
        return torch.empty(0), {}

    final_embeddings = torch.cat(agg_embeddings, dim=0).float()  # [P, D]
    torch.save({"embeddings": final_embeddings, "names": names_dict}, _facebank_file(conf, model))

    print(f"[facebank] Prepared -> {_facebank_file(conf, model)} | {len(names_dict)} ids | embs={tuple(final_embeddings.shape)}")
    return final_embeddings, names_dict
