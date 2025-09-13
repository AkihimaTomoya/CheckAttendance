import cv2
from PIL import Image
import torch
from typing import Dict, List, Tuple, Any

from infer_learner import face_learner
from face_detector import FaceDetector
from utils.utils_facebank import (
    load_facebank, prepare_facebank, _arch_dir, _facebank_file
)

# ====== Module state ======
conf = None               # config
yolo = None               # detector
learner = None            # face_learner
targets = torch.empty(0)  # facebank embeddings
names: Dict[int, str] = {}  # id -> name

# ----- Runtime setters/getters -----
def set_threshold(val: float) -> float:
    """Set the recognition threshold in the learner and return the current value."""
    global learner
    if learner is not None:
        learner.threshold = float(val)
    return float(get_threshold())

def set_tta(enabled: bool) -> bool:
    """Enable/disable test-time augmentation in config and return current state."""
    global conf
    if conf is not None:
        conf.tta = bool(enabled)
    return bool(getattr(conf, "tta", False))

def get_threshold() -> float:
    """Return current threshold; 0.0 if learner not ready."""
    return float(learner.threshold) if learner is not None else 0.0

# ----- Initialization & facebank -----
def _need_rebuild_facebank(embs: torch.Tensor) -> bool:
    """Rebuild if embeddings missing or dimension mismatch."""
    if (not isinstance(embs, torch.Tensor)) or embs.numel() == 0:
        return True
    emb_dim = embs.shape[1]
    cfg_dim = int(getattr(conf, "embedding_size", emb_dim))
    return emb_dim != cfg_dim

def initialize(cfg, update_facebank: bool = False) -> None:
    """Initialize detector, model, and facebank."""
    global conf, yolo, learner, targets, names
    conf = cfg
    if not hasattr(conf, "device"):
        conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Force FP32 at runtime
    conf.fp16 = False

    # Detector
    yolo = FaceDetector()

    # Embedding model (FP32)
    learner = face_learner(conf)
    learner.threshold = float(getattr(conf, "threshold", 1.54))
    model_file = conf.model_file if getattr(conf, "device", torch.device("cpu")).type != "cpu" else conf.cpu_model_file
    learner.load_state(conf, model_file)
    learner.model.eval()
    if hasattr(learner.model, "fp16"):
        try:
            learner.model.fp16 = False
        except Exception:
            pass

    # Facebank load/prepare
    if update_facebank:
        t, n = prepare_facebank(conf, learner.model, yolo, tta=getattr(conf, "tta", False))
    else:
        t, n = load_facebank(conf, learner.model)

    if isinstance(t, torch.Tensor):
        t = t.to(conf.device).float()

    # Check compatibility -> rebuild if needed
    if _need_rebuild_facebank(t):
        t, n = prepare_facebank(conf, learner.model, yolo, tta=getattr(conf, "tta", False))
        if isinstance(t, torch.Tensor):
            t = t.to(conf.device).float()

    targets, names = t, n

def reload_facebank() -> Tuple[bool, str]:
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

def facebank_info() -> Dict[str, Any]:
    """Return current facebank/model info for UI."""
    import os
    if conf is None or learner is None:
        return {"error": "face_verify not initialized"}

    try:
        arch_dir = _arch_dir(conf, learner.model)
        fb_file = _facebank_file(conf, learner.model)
        conv1_in = int(getattr(getattr(learner.model, "conv1", None), "in_channels", 0))
        ids = (list(names.values()) if isinstance(names, dict)
               else (list(names) if isinstance(names, (list, tuple)) else []))
        return {
            "network": getattr(conf, "network", None),
            "threshold": float(get_threshold()),
            "use_ffm": bool(getattr(learner.model, "use_ffm", False)),
            "conv1_in": conv1_in,
            "facebank_dir": str(arch_dir),
            "facebank_file": str(fb_file),
            "facebank_exists": os.path.isfile(fb_file),
            "targets_shape": None if not isinstance(targets, torch.Tensor) else tuple(targets.shape),
            "num_identities": len(ids),
            "names": ids,
        }
    except Exception as e:
        return {"error": str(e)}

# ----- App (merged flows) -----
class FaceVerificationApp:
    def __init__(self, width: int = 800, height: int = 800):
        self.width = width
        self.height = height
        self.last_recognition_results: Dict[str, Any] = {}

    def _targets_on_device(self) -> torch.Tensor:
        """Ensure targets are on the correct device before inference."""
        global targets
        if isinstance(targets, torch.Tensor) and targets.device != conf.device:
            targets = targets.to(conf.device).float()
        return targets

    def _idx_to_name(self, ridx: int) -> str:
        """Map index in targets (0..P-1) to display name from `names`."""
        global names
        if isinstance(names, dict):
            if len(names) == 0:
                return "Unknown"
            # direct index if key exists
            if ridx in names:
                return names[ridx]
            # fallback: sorted by key order
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
        face_locations: List[Dict[str, Any]] = []
        recognition_data: Dict[str, Any] = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res = yolo.align_multi(image, getattr(conf, "face_limit", 10), getattr(conf, "min_face_size", 30))
            if res is None:
                self.last_recognition_results = recognition_data
                return face_locations, recognition_data

            bboxes, faces = res
            if faces is None or len(faces) == 0:
                self.last_recognition_results = recognition_data
                return face_locations, recognition_data

            # bboxes: ndarray (N, 5) with score at -1; we draw first 4 ints
            bboxes_draw = (bboxes[:, :4].astype(int))

            # build locs
            for idx, bbox in enumerate(bboxes_draw):
                face_locations.append({"id": f"face_{idx}", "bbox": bbox.tolist()})

            # infer
            tgt = self._targets_on_device()
            results, distances = learner.infer(conf, faces, tgt, getattr(conf, "tta", False))

            for idx, _ in enumerate(bboxes_draw):
                face_id = f"face_{idx}"
                # distance safeguard
                if isinstance(distances, torch.Tensor) and distances.numel() > idx:
                    distance = float(distances[idx])
                else:
                    distance = 9.99

                # index (thresholded already)
                if isinstance(results, torch.Tensor) and results.numel() > idx:
                    idx_thr = int(results[idx])
                else:
                    idx_thr = -1

                # top1 name (same as idx_thr because infer() already applied threshold for idx)
                name_top1 = self._idx_to_name(idx_thr) if idx_thr >= 0 else "Unknown"

                passed = (idx_thr >= 0 and distance <= float(learner.threshold))
                name = name_top1 if passed else "Unknown"

                # rough confidence proxy from distance
                confidence = max(0.0, 1.0 - (distance / (float(learner.threshold) * 2.0)))

                recognition_data[face_id] = {
                    "name": name,
                    "distance": distance,
                    "passed_threshold": passed,
                    "name_top1": name_top1,
                    "threshold": float(learner.threshold),
                    "confidence": confidence,
                }

                # ---- Terminal logging of recognition result ----
                try:
                    nm = recognition_data[face_id].get("name_top1", recognition_data[face_id].get("name", "Unknown"))
                    dist = float(recognition_data[face_id].get("distance", 0.0))
                    thr = float(learner.threshold)
                    ok = bool(recognition_data[face_id].get("passed_threshold", False))
                    print(f"[recognize] {nm} | dist={dist:.3f} | thr={thr:.3f} | {'PASSED' if ok else 'not passed'}")
                except Exception:
                    pass

            # update last results on success path
            self.last_recognition_results = recognition_data

        except Exception:
            # fail-closed: return empty results
            self.last_recognition_results = recognition_data

        return face_locations, recognition_data
