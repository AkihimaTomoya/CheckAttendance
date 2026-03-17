import logging
import asyncio
import base64
import json
import time
import argparse
import torch
import cv2
import numpy as np
import uvicorn

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.utils_config import get_config

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_app")

# ─── Model hints ──────────────────────────────────────────────────────────────
MODEL_NAME_HINTS = [
    ("ms1mv3_r50",    "r50"),
    ("res50_custom2", "r50_custom_v2"),
    ("res50_custom",  "r50_custom_v1"),
    ("res50_fan",     "r50_fan"),
    ("res50_ffm",     "r50_ffm"),
    ("r100",          "r100"),
]
MODEL_BEST_THRESHOLDS = {
    "r50":           1.7,
    "res50_ffm":     1.56,
    "res50_fan":     1.7,
    "res50_custom2": 1.7,
    "res50_custom":  1.7,
    "r100":          1.7,
}

def apply_model_overrides(cfg, model_dir: str):
    dir_name = Path(model_dir).name.lower()
    for key, net in MODEL_NAME_HINTS:
        if key in dir_name:
            cfg.network = net
            if key in MODEL_BEST_THRESHOLDS:
                cfg.threshold = MODEL_BEST_THRESHOLDS[key]
            log.info(f"Model override: network={net}, threshold={cfg.threshold}")
            return

# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FastAPI Face Recognition Server")
parser.add_argument("-c", "--config",    default="configs/infer.py")
parser.add_argument("-d", "--model-dir", type=str, default=None)
parser.add_argument("--host",            default="0.0.0.0")
parser.add_argument("--port",            type=int, default=5050)
args = parser.parse_args()

config = get_config(args.config)
if not hasattr(config, "device"):
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {config.device}")

if args.model_dir:
    config.output = args.model_dir
    apply_model_overrides(config, args.model_dir)

try:
    import importlib
    face_verify = importlib.import_module("face_verify")
    log.info("Initializing face_verify (building facebank)…")
    face_verify.initialize(config, update_facebank=True)
    FaceVerificationApp   = face_verify.FaceVerificationApp
    face_verification_app = FaceVerificationApp()
    log.info(f"face_verify ready. threshold={face_verify.get_threshold():.3f}")
except Exception:
    log.exception("Failed to initialize face_verify")
    face_verification_app = None

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Face Recognition API", version="1.0.0")

try:
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
except Exception:
    pass

ui_config = {
    "show_bbox":  True,
    "show_label": True,
    "threshold":  float(face_verify.get_threshold()) if face_verification_app
                  else float(getattr(config, "threshold", 1.54)),
}

class SetThreshold(BaseModel):
    threshold: float
class SetTTA(BaseModel):
    tta: bool
class UIConfig(BaseModel):
    show_bbox: bool
    show_label: bool

# ─── WebSocket Connection Manager ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        log.info(f"WS connected. Total: {len(self.active_connections)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)
        log.info(f"WS disconnected. Total: {len(self.active_connections)}")

    async def send(self, msg: dict, ws: WebSocket):
        try:
            await ws.send_text(json.dumps(msg))
        except Exception:
            self.disconnect(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for c in self.active_connections:
            try:
                await c.send_text(json.dumps(msg))
            except Exception:
                dead.append(c)
        for c in dead:
            self.disconnect(c)

manager = ConnectionManager()

# Thread pool: 1 worker duy nhất để GPU không bị race condition giữa 2 threads
_executor = ThreadPoolExecutor(max_workers=1)

# ─── REST endpoints ───────────────────────────────────────────────────────────
@app.post("/config")
async def update_ui_config(new_config: UIConfig):
    global ui_config
    ui_config.update(new_config.dict())
    ui_config["threshold"] = float(face_verify.get_threshold()) if face_verification_app else ui_config.get("threshold", 1.54)
    await manager.broadcast({"type": "config", "data": ui_config})
    return {"status": "success", "config": ui_config}

@app.get("/config")
async def get_ui_config():
    ui_config["threshold"] = float(face_verify.get_threshold()) if face_verification_app else ui_config.get("threshold", 1.54)
    return {"status": "success", "config": ui_config}

@app.post("/reload-facebank")
async def reload_facebank_endpoint():
    if not face_verification_app:
        return {"status": "error", "message": "Face verification app not initialized"}
    log.info("Reloading facebank…")
    ok, message = face_verify.reload_facebank()
    log.info(f"Facebank reload: {'OK' if ok else 'FAIL'} — {message}")
    return {"status": "success" if ok else "error", "message": message}

@app.get("/status")
async def get_status():
    return {
        "status":             "success",
        "app_initialized":    face_verification_app is not None,
        "active_connections": len(manager.active_connections),
        "config":             ui_config,
    }

@app.post("/set-threshold")
async def set_threshold_api(req: SetThreshold):
    try:
        val = face_verify.set_threshold(req.threshold)
        log.info(f"Threshold set to {val:.3f}")
        await manager.broadcast({"type": "config", "data": {"threshold": float(val)}})
        return {"status": "success", "threshold": float(val)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/set-tta")
async def set_tta_api(req: SetTTA):
    try:
        cur = face_verify.set_tta(req.tta)
        log.info(f"TTA set to {cur}")
        return {"status": "success", "tta": bool(cur)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/presence")
async def get_presence_log_path():
    csv_path = getattr(face_verify._presence_logger, "csv_path", None) if face_verification_app else None
    return {"status": "success", "csv_path": csv_path}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        html_path = Path("templates/fastapi-index.html")
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Face Recognition Server</h1><p>Template missing.</p>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><pre>{e}</pre>")

# ─── WebSocket ────────────────────────────────────────────────────────────────
# FIX CHÍNH: tách receiver và processor thành 2 coroutine chạy song song.
#
# Vấn đề cũ:
#   while True:
#       data = await receive_text()   # đợi frame
#       await process_frame(...)      # đợi inference xong (~200-400ms)
#   → trong lúc inference, client vẫn gửi frame mới vào buffer WS
#   → buffer tích lũy N frame cũ, server xử lý từng frame theo thứ tự
#   → bbox hiển thị kết quả của frame vài giây trước, không phản ánh
#      góc nhìn mới dù chờ 5-10s
#
# Fix:
#   receiver() — luôn đọc buffer WS nhanh nhất có thể, chỉ giữ frame MỚI NHẤT
#   processor() — lấy frame từ queue, infer, gửi kết quả
#   Queue maxsize=1 + drain: khi frame mới về, frame cũ chưa xử lý bị DROP
#   → server LUÔN infer frame gần nhất, không bao giờ tụt hậu

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await manager.send({"type": "config", "data": ui_config}, websocket)

    # Queue size=1: chỉ giữ 1 frame pending — frame mới luôn thay frame cũ
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    frame_count  = 0
    drop_count   = 0
    infer_count  = 0

    async def receiver():
        nonlocal frame_count, drop_count
        while True:
            raw = await websocket.receive_text()
            frame_count += 1
            # Nếu queue đang chứa frame chưa xử lý → drop nó, thay bằng frame mới
            if not queue.empty():
                try:
                    queue.get_nowait()
                    drop_count += 1
                except asyncio.QueueEmpty:
                    pass
            await queue.put(raw)

    async def processor():
        nonlocal infer_count
        loop = asyncio.get_event_loop()
        while True:
            raw = await queue.get()

            if not face_verification_app:
                await manager.send({"type": "error", "data": "app not initialized"}, websocket)
                continue

            # Decode ảnh
            try:
                encoded  = raw.split(",", 1)[1] if "," in raw else raw
                img_data = base64.b64decode(encoded)
                np_arr   = np.frombuffer(img_data, np.uint8)
                frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                log.warning("Frame decode failed")
                continue

            if frame is None:
                log.warning("cv2.imdecode returned None")
                continue

            h, w = frame.shape[:2]
            t0 = time.perf_counter()

            try:
                locs, recognition_data = await loop.run_in_executor(
                    _executor,
                    face_verification_app.recognize_faces_and_locs,
                    frame,
                )
            except Exception:
                log.exception("Inference error")
                continue

            infer_ms = (time.perf_counter() - t0) * 1000
            infer_count += 1

            # Log mỗi 30 frame: tổng/drop/infer + kết quả nhận dạng
            if infer_count % 30 == 0:
                names_found = [v.get("name", "?") for v in recognition_data.values()]
                log.info(
                    f"frames={frame_count} dropped={drop_count} infer={infer_count} "
                    f"| last infer {infer_ms:.0f}ms | {w}x{h} "
                    f"| faces={len(locs)} {names_found}"
                )

            # Log mỗi frame nếu có mặt (để debug bbox)
            for face_id, rec in recognition_data.items():
                loc = next((l for l in locs if l["id"] == face_id), None)
                bbox_str = str(loc["bbox"]) if loc else "n/a"
                log.debug(
                    f"  {face_id}: {rec.get('name','?')} "
                    f"dist={rec.get('distance', 0):.3f} "
                    f"passed={rec.get('passed_threshold')} "
                    f"bbox={bbox_str}"
                )

            await manager.send({
                "type": "frame_result",
                "locs": locs,
                "data": recognition_data,
                "meta": {"threshold": float(face_verify.get_threshold())},
            }, websocket)

    receiver_task  = asyncio.create_task(receiver())
    processor_task = asyncio.create_task(processor())

    try:
        await asyncio.gather(receiver_task, processor_task)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("WS session error")
    finally:
        receiver_task.cancel()
        processor_task.cancel()
        manager.disconnect(websocket)
        log.info(
            f"Session closed — frames={frame_count} dropped={drop_count} infer={infer_count}"
        )

# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status":          "healthy",
        "timestamp":       time.time(),
        "app_initialized": face_verification_app is not None,
    }

if __name__ == "__main__":
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)
    except Exception:
        log.exception("uvicorn crashed")
