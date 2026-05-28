import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import torch
from PIL import Image

from face_detector import FaceDetector
from infer_learner import face_learner
from utils.utils_facebank import (
    _arch_dir, _facebank_file, load_facebank, prepare_facebank,
)

# ──────────────────────────────────────────────────────────────
#  Module state
# ──────────────────────────────────────────────────────────────
conf    = None
yolo    = None
learner = None
targets = torch.empty(0)
names: Dict[int, str] = {}

# ── Runtime setters/getters ───────────────────────────────────

def set_threshold(val: float) -> float:
    global learner
    if learner is not None:
        learner.threshold = float(val)
    return float(get_threshold())

def set_tta(enabled: bool) -> bool:
    global conf
    if conf is not None:
        conf.tta = bool(enabled)
    return bool(getattr(conf, "tta", False))

def get_threshold() -> float:
    return float(learner.threshold) if learner is not None else 0.0

# ── Initialization & facebank ─────────────────────────────────

def _need_rebuild_facebank(embs: torch.Tensor) -> bool:
    if (not isinstance(embs, torch.Tensor)) or embs.numel() == 0:
        return True
    emb_dim = embs.shape[1]
    cfg_dim = int(getattr(conf, "embedding_size", emb_dim))
    return emb_dim != cfg_dim


def initialize(cfg, update_facebank: bool = False) -> None:
    global conf, yolo, learner, targets, names

    conf = cfg
    if not hasattr(conf, "device"):
        conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf.fp16 = False

    yolo = FaceDetector()

    learner = face_learner(conf)
    learner.threshold = float(getattr(conf, "threshold", 1.54))
    model_file = (conf.model_file
                  if getattr(conf, "device", torch.device("cpu")).type != "cpu"
                  else conf.cpu_model_file)
    learner.load_state(conf, model_file)
    learner.model.eval()
    if hasattr(learner.model, "fp16"):
        try:
            learner.model.fp16 = False
        except Exception:
            pass

    if update_facebank:
        t, n = prepare_facebank(conf, learner.model, yolo, tta=getattr(conf, "tta", False))
    else:
        t, n = load_facebank(conf, learner.model)

    if isinstance(t, torch.Tensor):
        t = t.to(conf.device).float()

    if _need_rebuild_facebank(t):
        t, n = prepare_facebank(conf, learner.model, yolo, tta=getattr(conf, "tta", False))
        if isinstance(t, torch.Tensor):
            t = t.to(conf.device).float()

    targets, names = t, n


def reload_facebank() -> Tuple[bool, str]:
    global targets, names
    if conf is None or learner is None or yolo is None:
        return False, "face_verify is not initialized"
    try:
        arch_dir = _arch_dir(conf, learner.model)
        t, n = prepare_facebank(conf, learner.model, yolo, tta=getattr(conf, "tta", False))
        if isinstance(t, torch.Tensor):
            t = t.to(conf.device).float()

        if (not isinstance(t, torch.Tensor)) or t.numel() == 0 or len(n) == 0:
            return False, (
                f"No identities found under: {str(arch_dir)}. "
                f"Put images into subfolders, e.g. {str(arch_dir)}/<PersonName>/*.jpg"
            )
        targets, names = t, n
        msg = (f"Facebank updated: {len(names)} identities, "
               f"{0 if t.numel()==0 else t.shape[0]} embeddings "
               f"at {_facebank_file(conf, learner.model)}")
        return True, msg
    except Exception as e:
        return False, f"Reload facebank failed: {e}"


def facebank_info() -> Dict[str, Any]:
    import os
    if conf is None or learner is None:
        return {"error": "face_verify not initialized"}
    try:
        arch_dir  = _arch_dir(conf, learner.model)
        fb_file   = _facebank_file(conf, learner.model)
        conv1_in  = int(getattr(getattr(learner.model, "conv1", None), "in_channels", 0))
        ids = (list(names.values()) if isinstance(names, dict)
               else (list(names) if isinstance(names, (list, tuple)) else []))
        return {
            "network":         getattr(conf, "network", None),
            "threshold":       float(get_threshold()),
            "use_ffm":         bool(getattr(learner.model, "use_ffm", False)),
            "conv1_in":        conv1_in,
            "facebank_dir":    str(arch_dir),
            "facebank_file":   str(fb_file),
            "facebank_exists": os.path.isfile(fb_file),
            "targets_shape":   None if not isinstance(targets, torch.Tensor) else tuple(targets.shape),
            "num_identities":  len(ids),
            "names":           ids,
        }
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────
#  FaceVerificationApp
# ──────────────────────────────────────────────────────────────

class FaceVerificationApp:
    def __init__(self, width: int = 800, height: int = 800):
        self.width  = width
        self.height = height
        self.last_recognition_results: Dict[str, Any] = {}

    def _targets_on_device(self) -> torch.Tensor:
        global targets
        if isinstance(targets, torch.Tensor) and targets.device != conf.device:
            targets = targets.to(conf.device).float()
        return targets

    def _idx_to_name(self, ridx: int) -> str:
        global names
        if isinstance(names, dict):
            if len(names) == 0:
                return "Unknown"
            if ridx in names:
                return names[ridx]
            keys_sorted = sorted(names.keys())
            if 0 <= ridx < len(keys_sorted):
                return names[keys_sorted[ridx]]
            return "Unknown"
        if isinstance(names, (list, tuple)):
            if 0 <= ridx < len(names):
                return names[ridx]
            return "Unknown"
        return "Unknown"

    def recognize_faces_and_locs(self, frame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Nhận dạng khuôn mặt và trả về kết quả cho caller.
        Không chứa logic ghi file CSV.

        Returns
        -------
        face_locations : list[{"id": str, "bbox": [x1,y1,x2,y2]}]
        recognition_data : dict{ face_id: { name, distance, passed_threshold,
                                            name_top1, threshold, confidence } }
        """
        face_locations:   List[Dict[str, Any]] = []
        recognition_data: Dict[str, Any]       = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res   = yolo.align_multi(image,
                                     getattr(conf, "face_limit", 10),
                                     getattr(conf, "min_face_size", 30))
            if res is None:
                return face_locations, recognition_data

            bboxes, faces = res
            if faces is None or len(faces) == 0:
                return face_locations, recognition_data

            bboxes_draw = bboxes[:, :4].astype(int)

            for idx, bbox in enumerate(bboxes_draw):
                face_locations.append({"id": f"face_{idx}", "bbox": bbox.tolist()})

            tgt = self._targets_on_device()
            # torch.no_grad(): bỏ tính gradient không cần thiết khi inference
            with torch.no_grad():
                results, distances = learner.infer(conf, faces, tgt, getattr(conf, "tta", False))

            thr = float(learner.threshold)
            for idx, _ in enumerate(bboxes_draw):
                face_id = f"face_{idx}"

                distance = (float(distances[idx])
                            if isinstance(distances, torch.Tensor) and distances.numel() > idx
                            else 9.99)

                idx_thr  = (int(results[idx])
                            if isinstance(results, torch.Tensor) and results.numel() > idx
                            else -1)

                name_top1  = self._idx_to_name(idx_thr) if idx_thr >= 0 else "Unknown"
                passed     = idx_thr >= 0 and distance <= thr
                name       = name_top1 if passed else "Unknown"
                confidence = max(0.0, 1.0 - distance / (thr * 2.0))

                recognition_data[face_id] = {
                    "name":             name,
                    "distance":         round(distance, 3),
                    "passed_threshold": passed,
                    "name_top1":        name_top1,
                    "threshold":        thr,
                    "confidence":       round(confidence, 3),
                }

            self.last_recognition_results = recognition_data

        except Exception:
            pass

        return face_locations, recognition_data