import csv
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import torch
from PIL import Image

from face_detector import FaceDetector
from infer_learner import face_learner
from utils.utils_facebank import (
    _arch_dir, _facebank_file, load_facebank, prepare_facebank,
)

# ──────────────────────────────────────────────────────────────
#  PresenceLogger
#  Toàn bộ logic theo dõi / CSV nằm ở đây.
#  fastapi-app.py không cần biết gì về CSV.
# ──────────────────────────────────────────────────────────────

class PresenceLogger:
    """
    Theo dõi sự hiện diện từng người qua các frame.

    Chống nhiễu camera bằng sliding-window:
      - Chỉ xác nhận "xuất hiện"  khi tên PASSED liên tiếp >= CONFIRM_FRAMES
      - Chỉ xác nhận "biến mất"   khi PASSED = 0 liên tiếp  >= ABSENT_FRAMES
      Điều này loại trừ trường hợp 1 frame Unknown xen giữa nhiều frame PASSED.

    Vắng mặt > ABSENCE_ALERT_MINUTES => cột absence_alert = True trong CSV.
    """

    CONFIRM_FRAMES        = 4    # frame PASSED liên tiếp để coi là "xuất hiện"
    ABSENT_FRAMES         = 8    # frame không-PASSED liên tiếp để coi là "biến mất"
    ABSENCE_ALERT_MINUTES = 30   # ngưỡng gắn cờ absent_alert

    # Cột CSV
    _HEADERS = [
        "name",
        "event_type",        # "appeared" | "disappeared"
        "datetime",
        "timestamp",
        "absence_alert",     # True nếu khoảng vắng > 30 phút (chỉ có ở "disappeared")
        "absence_minutes",   # thời gian vắng mặt tính bằng phút  (chỉ có ở "disappeared")
    ]

    def __init__(self, log_dir: str = "logs"):
        self._log_path = Path(log_dir) / "presence_log.csv"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()

        # Khởi tạo file CSV (chỉ ghi header nếu chưa có)
        if not self._log_path.exists():
            with open(self._log_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._HEADERS).writeheader()

        # Trạng thái mỗi người:
        #   "window"      : deque[bool]  — lịch sử PASSED/không-PASSED gần nhất
        #   "is_present"  : bool
        #   "appeared_at" : float | None — timestamp lúc xác nhận xuất hiện
        #   "last_seen"   : float | None — timestamp frame PASSED cuối cùng
        self._state: Dict[str, dict] = {}

    # ── public API ────────────────────────────────────────────

    def update(self, recognition_data: Dict[str, Any]) -> None:
        """
        Gọi sau mỗi frame được infer.
        recognition_data = dict trả về từ recognize_faces_and_locs().
        """
        now = time.time()

        # Tập hợp tên PASSED trong frame này (bỏ Unknown)
        passed_names: set = set()
        for rec in recognition_data.values():
            if rec.get("passed_threshold") and rec.get("name", "Unknown") != "Unknown":
                passed_names.add(rec["name"])

        with self._lock:
            # Cập nhật window cho tất cả người đã biết + người mới
            all_names = passed_names | {n for n, s in self._state.items() if s["is_present"]}

            for name in all_names:
                s = self._get_state(name)
                seen_this_frame = (name in passed_names)

                if seen_this_frame:
                    s["last_seen"] = now

                s["window"].append(seen_this_frame)

                if not s["is_present"]:
                    # Kiểm tra xác nhận "xuất hiện"
                    if self._all_true_recent(s["window"], self.CONFIRM_FRAMES):
                        s["is_present"]  = True
                        s["appeared_at"] = now
                        self._write_row(name, "appeared", now)
                else:
                    # Kiểm tra xác nhận "biến mất"
                    if self._all_false_recent(s["window"], self.ABSENT_FRAMES):
                        s["is_present"] = False
                        absence_minutes = self._absence_minutes(s["appeared_at"], s["last_seen"])
                        alert           = absence_minutes is not None and absence_minutes >= self.ABSENCE_ALERT_MINUTES
                        self._write_row(
                            name, "disappeared", s["last_seen"] or now,
                            absence_minutes=absence_minutes,
                            absence_alert=alert,
                        )

    @property
    def csv_path(self) -> str:
        return str(self._log_path)

    # ── internal helpers ──────────────────────────────────────

    def _get_state(self, name: str) -> dict:
        if name not in self._state:
            self._state[name] = {
                "window":      deque(maxlen=max(self.CONFIRM_FRAMES, self.ABSENT_FRAMES) + 2),
                "is_present":  False,
                "appeared_at": None,
                "last_seen":   None,
            }
        return self._state[name]

    @staticmethod
    def _all_true_recent(window: deque, n: int) -> bool:
        """Kiểm tra n phần tử cuối đều True."""
        tail = list(window)[-n:]
        return len(tail) == n and all(tail)

    @staticmethod
    def _all_false_recent(window: deque, n: int) -> bool:
        """Kiểm tra n phần tử cuối đều False."""
        tail = list(window)[-n:]
        return len(tail) == n and not any(tail)

    @staticmethod
    def _absence_minutes(appeared_at: float, last_seen: float) -> float | None:
        if appeared_at is None or last_seen is None:
            return None
        duration_sec = max(0.0, last_seen - appeared_at)
        return round(duration_sec / 60, 2)

    def _write_row(
        self,
        name: str,
        event_type: str,
        ts: float,
        absence_minutes: float | None = None,
        absence_alert: bool = False,
    ) -> None:
        row = {
            "name":            name,
            "event_type":      event_type,
            "datetime":        datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp":       round(ts, 3),
            "absence_alert":   absence_alert   if event_type == "disappeared" else "",
            "absence_minutes": absence_minutes if event_type == "disappeared" else "",
        }
        with open(self._log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._HEADERS).writerow(row)


# ──────────────────────────────────────────────────────────────
#  Module state
# ──────────────────────────────────────────────────────────────
conf    = None
yolo    = None
learner = None
targets = torch.empty(0)
names: Dict[int, str] = {}

_presence_logger: PresenceLogger | None = None

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
    global conf, yolo, learner, targets, names, _presence_logger

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

    # Khởi tạo logger (log_dir có thể override qua conf nếu cần)
    log_dir = str(getattr(conf, "log_dir", "logs"))
    _presence_logger = PresenceLogger(log_dir=log_dir)


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
        Nhận dạng khuôn mặt, cập nhật presence logger, trả về kết quả cho caller.

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
                self._log_and_store(recognition_data)
                return face_locations, recognition_data

            bboxes, faces = res
            if faces is None or len(faces) == 0:
                self._log_and_store(recognition_data)
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
                # print đã bị bỏ — stdout flush mỗi frame là bottleneck lớn

            self._log_and_store(recognition_data)

        except Exception:
            self._log_and_store(recognition_data)

        return face_locations, recognition_data

    # ── internal ─────────────────────────────────────────────

    def _log_and_store(self, recognition_data: Dict[str, Any]) -> None:
        """Cập nhật presence logger và lưu kết quả cuối."""
        self.last_recognition_results = recognition_data
        if _presence_logger is not None:
            try:
                _presence_logger.update(recognition_data)
            except Exception:
                pass
