# face_verify.py

import cv2
from PIL import Image
import torch

from infer_learner import face_learner
from face_detector import FaceDetector
from utils.utils_facebank import (
    load_facebank, prepare_facebank, _is_meta_compatible,
    _arch_dir, _facebank_file, facebank_group_of
)

# ====== Module state ======
conf = None               # config
yolo = None               # detector
learner = None            # face_learner
targets = torch.empty(0)  # facebank embeddings
names = {}                # id -> name

# Runtime flags
DEBUG_ALWAYS_TOP1 = False  # optionally show Top-1 ignoring threshold


# ----- Runtime setters/getters -----
def set_threshold(val: float):
    global learner
    if learner is not None:
        learner.threshold = float(val)
    return float(get_threshold())

def set_tta(enabled: bool):
    global conf
    if conf is not None:
        conf.tta = bool(enabled)
    return bool(getattr(conf, "tta", False))

def set_debug_top1(enabled: bool):
    global DEBUG_ALWAYS_TOP1
    DEBUG_ALWAYS_TOP1 = bool(enabled)
    return DEBUG_ALWAYS_TOP1

def get_runtime_config():
    return {
        "threshold": float(get_threshold()),
        "tta": bool(getattr(conf, "tta", False)),
        "debug_top1": bool(DEBUG_ALWAYS_TOP1),
    }

def get_threshold() -> float:
    return float(learner.threshold) if learner is not None else 0.0


# ----- Initialization & facebank -----
def initialize(cfg, update_facebank=False):
    """Initialize detector, model, and facebank."""
    global conf, yolo, learner, targets, names
    conf = cfg
    if not hasattr(conf, "device"):
        conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # detector
    yolo = FaceDetector()

    # embedding model
    learner = face_learner(conf)
    learner.threshold = float(getattr(conf, "threshold", 1.54))
    model_file = conf.model_file if conf.device.type != "cpu" else conf.cpu_model_file
    learner.load_state(conf, model_file)
    learner.model.eval()

    # facebank
    if update_facebank:
        t, n = prepare_facebank(conf, learner.model, yolo, tta=getattr(conf, "tta", False))
    else:
        t, n = load_facebank(conf, learner.model)

    if isinstance(t, torch.Tensor):
        t = t.to(conf.device).float()

    # check if rebuild needed
    need_rebuild = False
    meta = getattr(conf, "_loaded_facebank_meta", None)
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        need_rebuild = True
    else:
        try:
            if not _is_meta_compatible(meta, conf, learner.model):
                need_rebuild = True
        except Exception:
            need_reuild = True  # fallback

    if need_rebuild:
        t, n = prepare_facebank(conf, learner.model, yolo, tta=getattr(conf, "tta", False))
        if isinstance(t, torch.Tensor):
            t = t.to(conf.device).float()

    targets, names = t, n


def reload_facebank():
    """Rebuild facebank using current detector + model (honors conf.tta)."""
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


def facebank_info():
    """Return current facebank/model info for UI."""
    import os
    if conf is None or learner is None:
        return {"error": "face_verify not initialized"}

    try:
        arch_dir = _arch_dir(conf, learner.model)
        fb_file = _facebank_file(conf, learner.model)
        meta = getattr(conf, "_loaded_facebank_meta", None)
        conv1_in = int(getattr(getattr(learner.model, "conv1", None), "in_channels", 0))
        group = facebank_group_of(conf, learner.model)
        ids = (list(names.values()) if isinstance(names, dict)
               else (list(names) if isinstance(names, (list, tuple)) else []))
        return {
            "network": getattr(conf, "network", None),
            "threshold": float(get_threshold()),
            "use_ffm": bool(getattr(learner.model, "use_ffm", False)),
            "conv1_in": conv1_in,
            "group": group,
            "facebank_dir": str(arch_dir),
            "facebank_file": str(fb_file),
            "facebank_exists": os.path.isfile(fb_file),
            "targets_shape": None if not isinstance(targets, torch.Tensor) else tuple(targets.shape),
            "num_identities": len(ids),
            "names": ids,
            "meta": meta
        }
    except Exception as e:
        return {"error": str(e)}


# ----- App (merged flows) -----
class FaceVerificationApp:
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
        self.last_recognition_results = {}

    def _targets_on_device(self):
        """Ensure targets are on the correct device before inference."""
        global targets
        if isinstance(targets, torch.Tensor) and targets.device != conf.device:
            targets = targets.to(conf.device).float()
        return targets

    def _idx_to_name(self, ridx: int):
        """Map index in targets (0..P-1) to display name from `names`."""
        global names
        if isinstance(names, dict):
            if len(names) == 0:
                return "Unknown"
            keys_sorted = sorted(names.keys())
            if 0 <= ridx < len(keys_sorted):
                return names[keys_sorted[ridx]]
            if ridx in names:
                return names[ridx]
            if (ridx + 1) in names:
                return names[ridx + 1]
            return "Unknown"

        if isinstance(names, (list, tuple)):
            if 0 <= ridx < len(names):
                return names[ridx]
            if 0 <= (ridx + 1) < len(names):
                return names[ridx + 1]
            return "Unknown"

        return "Unknown"

    def recognize_faces_and_locs(self, frame):
        """
        Single entry point: return (face_locations_list, recognition_data_dict)

        face_locations_list: [{"id": "face_i", "bbox": [x1,y1,x2,y2]}, ...]
        recognition_data_dict: {
            "face_i": {
                "name": str,
                "distance": float,
                "passed_threshold": bool,
                "name_top1": str,
                "threshold": float,
                "confidence": float
            }, ...
        }
        """
        face_locations = []
        recognition_data = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res = yolo.align_multi(image, conf.face_limit, conf.min_face_size)
            if res is None:
                self.last_recognition_results = recognition_data
                return face_locations, recognition_data

            bboxes, faces = res
            if faces is None or len(faces) == 0:
                self.last_recognition_results = recognition_data
                return face_locations, recognition_data

            bboxes = (bboxes[:, :4].astype(int))

            # build locs
            for idx, bbox in enumerate(bboxes):
                face_locations.append({"id": f"face_{idx}", "bbox": bbox.tolist()})

            # infer
            tgt = self._targets_on_device()
            results, distances = learner.infer(conf, faces, tgt, getattr(conf, "tta", False))

            for idx, bbox in enumerate(bboxes):
                face_id = f"face_{idx}"
                name = "Unknown"
                distance = float(distances[idx]) if distances.numel() > 0 else 9.99

                # Top-1 ignoring threshold (for optional display)
                top1_idx = int(results[idx]) if results.numel() > 0 else -1
                if distances.numel() > 0:
                    top1_idx = int(torch.argmin(distances).item()) if distances.ndim == 1 else top1_idx
                name_top1 = self._idx_to_name(top1_idx) if top1_idx >= 0 else "Unknown"

                passed = (results.numel() > 0 and int(results[idx]) >= 0 and distance <= learner.threshold)
                if passed and not DEBUG_ALWAYS_TOP1:
                    name = self._idx_to_name(int(results[idx]))
                elif DEBUG_ALWAYS_TOP1:
                    name = name_top1

                confidence = max(0.0, 1.0 - (distance / (learner.threshold * 2)))
                recognition_data[face_id] = {
                    "name": name,
                    "distance": distance,
                    "passed_threshold": passed,
                    "name_top1": name_top1,
                    "threshold": float(learner.threshold),
                    "confidence": confidence,
                }

            self.last_recognition_results = recognition_data
        except Exception:
            # fail-closed: return empty results
            self.last_recognition_results = recognition_data

        return face_locations, recognition_data
