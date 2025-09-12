# utils/utils_facebank.py
import os
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from PIL import Image
import cv2


# =========================
# Helpers: normalize & draw
# =========================
def l2_norm(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=eps)
    return x / norm

def draw_box_name(box, name, img_rgb):
    """Draw bounding boxes and labels (img_rgb: RGB ndarray)."""
    x1, y1, x2, y2 = map(int, box)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_bgr, str(name), (x1, max(0, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# =========================
# Facebank grouping by model
# =========================
GROUP_MAP = {
    # FFM:
    "r50_ffm": "_ffm",

    # Non-FFM:
    "r50": "_nonffm",
    "r50_custom_v1": "_nonffm",
    "r50_custom_v2": "_nonffm",
    "r100": "_nonffm",
    "r50_fan": "_nonffm",
    "r50_fan_ffm": "_ffm",
}

def facebank_group_of(conf, model=None) -> str:
    name = (getattr(conf, "network", None) or "").lower()
    if name in GROUP_MAP:
        return GROUP_MAP[name]
    if hasattr(model, "use_ffm") and bool(getattr(model, "use_ffm")):
        return "_ffm"
    return "_nonffm"

def _arch_dir(conf, model=None) -> Path:
    base = Path(getattr(conf, "facebank_path", Path("facebank")))
    group = facebank_group_of(conf, model)
    return base / group

def _facebank_file(conf, model=None) -> Path:
    return _arch_dir(conf, model) / "facebank.pt"

def _model_meta(conf, model) -> dict:
    conv1 = getattr(model, "conv1", None)
    return {
        "network": getattr(conf, "network", None),
        "group": facebank_group_of(conf, model),
        "embedding_size": int(getattr(conf, "embedding_size", 512)),
        "tta": bool(getattr(conf, "tta", False)),
        "use_ffm": bool(getattr(model, "use_ffm", False)),
        "conv1_in": int(getattr(conv1, "in_channels", 0)),
        "conv1_out": int(getattr(conv1, "out_channels", 0)),
    }

def _is_meta_compatible(meta: dict, conf, model) -> bool:
    if not isinstance(meta, dict):
        return False
    cur_group = facebank_group_of(conf, model)
    if meta.get("group") != cur_group:
        return False
    if int(meta.get("embedding_size", -1)) != int(getattr(conf, "embedding_size", -1)):
        return False
    if cur_group == "_ffm":
        conv1 = getattr(model, "conv1", None)
        if int(meta.get("conv1_in", -1)) != int(getattr(conv1, "in_channels", 0)):
            return False
    return True


# =========================
# Load / Save facebank
# =========================
def load_facebank(conf, model=None) -> Tuple[torch.Tensor, Dict[int, str]]:
    fb_path = _facebank_file(conf, model)
    if not fb_path.is_file():
        print(f"[facebank] Not found: {fb_path}. Returning empty.")
        return torch.empty(0), {}

    try:
        obj = torch.load(fb_path, map_location=getattr(conf, "device", "cpu"))
        if isinstance(obj, tuple) and len(obj) == 2:
            embeddings, names_dict = obj
            meta = None
        elif isinstance(obj, dict):
            embeddings = obj.get("embeddings", torch.empty(0))
            names_dict = obj.get("names", {})
            meta = obj.get("meta", None)
        else:
            print(f"[facebank] Invalid file format: {fb_path}")
            return torch.empty(0), {}

        if isinstance(embeddings, torch.Tensor) and embeddings.numel() > 0:
            embeddings = F.normalize(embeddings.float(), p=2, dim=1)

        setattr(conf, "_loaded_facebank_meta", meta)
        return embeddings, names_dict
    except Exception as e:
        print(f"[facebank] Load error: {e}")
        return torch.empty(0), {}


def _embed_one_image(conf, model, detector, pil_img: Image.Image, tta: bool) -> torch.Tensor:
    """
    Detect + align (1 face), return embedding [1, D] (đã L2-norm) or None.
    """
    try:
        res = detector.align_multi(pil_img, getattr(conf, "face_limit", 10), getattr(conf, "min_face_size", 30))
        if res is None:
            return None
        bboxes, faces = res
        if faces is None or len(faces) == 0:
            return None

        # chọn 1 mặt lớn nhất
        # ở đây bboxes shape [N,5], lấy score bboxes[:, -1]
        scores = bboxes[:, -1]
        idx = int(scores.argmax()) if hasattr(scores, "argmax") else 0
        face = faces[idx]

        # TTA horizontal flip
        if tta:
            from torchvision import transforms as trans
            mirror = trans.functional.hflip(face)
            inp = conf.test_transform(face).to(conf.device).unsqueeze(0)
            inp_m = conf.test_transform(mirror).to(conf.device).unsqueeze(0)
            with torch.no_grad():
                emb = model(inp) + model(inp_m)
                emb = l2_norm(emb, dim=1)
        else:
            inp = conf.test_transform(face).to(conf.device).unsqueeze(0)
            with torch.no_grad():
                emb = l2_norm(model(inp), dim=1)
        return emb  # [1, D]
    except Exception as e:
        print(f"[facebank._embed_one_image] {e}")
        return None


def prepare_facebank(conf, model, detector, tta: bool = True) -> Tuple[torch.Tensor, Dict[int, str]]:
    """
    Build facebank GROUP (non-FFM, FFM):
    - Traverse the directory: facebank/<group>/<PersonName>/*.jpg|png|jpeg
    - For each person, process multiple images -> detect/align -> embedding -> mean -> L2-norm
    - Save {embeddings, names, meta} to facebank/<group>/facebank.pt
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
        save_path = _facebank_file(conf, model)
        meta = _model_meta(conf, model)
        torch.save({"embeddings": torch.empty(0), "names": {}, "meta": meta}, save_path)
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
            except Exception as _:
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
        save_path = _facebank_file(conf, model)
        meta = _model_meta(conf, model)
        torch.save({"embeddings": torch.empty(0), "names": {}, "meta": meta}, save_path)
        return torch.empty(0), {}

    final_embeddings = torch.cat(agg_embeddings, dim=0)  # [P, D]

    save_path = _facebank_file(conf, model)
    meta = _model_meta(conf, model)
    torch.save({"embeddings": final_embeddings, "names": names_dict, "meta": meta}, save_path)

    print(f"[facebank] Prepared -> {save_path} | {len(names_dict)} ids | embs={tuple(final_embeddings.shape)} | meta={meta}")
    return final_embeddings, names_dict
