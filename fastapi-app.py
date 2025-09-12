from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64, cv2, numpy as np, uvicorn, argparse, time, traceback, json, torch
from pathlib import Path
from typing import List

from utils.utils_config import get_config

# Model hints for auto threshold override
MODEL_NAME_HINTS = [
    ("ms1mv3_r50",    "r50"),
    ("res50_custom2", "r50_custom_v2"),
    ("res50_custom",  "r50_custom_v1"),
    ("res50_fan",     "r50_fan"),
    ("res50_ffm",     "r50_ffm"),
    ("r100",          "r100"),
]

MODEL_THRESHOLDS = {
    "r50":           1.7,
    "res50_ffm":     1.56,
    "res50_fan":     1.7,
    "res50_custom2": 1.7,
    "res50_custom":  1.7,
    "r100":          1.7,
}

def apply_model_overrides(cfg, model_dir: str):
    """Infer network/threshold from folder name and override cfg."""
    dir_name = Path(model_dir).name.lower()
    for key, net in MODEL_NAME_HINTS:
        if key in dir_name:
            cfg.network = net
            if key in MODEL_THRESHOLDS:
                cfg.threshold = MODEL_THRESHOLDS[key]
            return

# --- CLI ---
parser = argparse.ArgumentParser(description="FastAPI Face Recognition Server")
parser.add_argument("-c", "--config", default="configs/infer.py", help="Path to inference config.")
parser.add_argument("-d", "--model-dir", type=str, default=None, help="Override config.output.")
parser.add_argument("--host", default="0.0.0.0", help="Bind host")
parser.add_argument("--port", type=int, default=5050, help="Bind port")
args = parser.parse_args()

# Load config
config = get_config(args.config)
if not hasattr(config, "device"):
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optional model-dir override
if args.model_dir:
    config.output = args.model_dir
    apply_model_overrides(config, args.model_dir)

# Import face_verify and init
try:
    import importlib
    face_verify = importlib.import_module("face_verify")
    face_verify.initialize(config, update_facebank=False)

    FaceVerificationApp = face_verify.FaceVerificationApp
    face_verification_app = FaceVerificationApp()
except Exception as e:
    traceback.print_exc()
    face_verification_app = None

# --- FastAPI App ---
app = FastAPI(title="Face Recognition API", version="1.0.0")

# Static files (optional)
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass

# UI config sent to client
ui_config = {
    "show_bbox": True,
    "show_label": True,
    "threshold": float(face_verify.get_threshold()) if face_verification_app else float(getattr(config, "threshold", 1.54)),
}

class SetThreshold(BaseModel):
    threshold: float
class SetTTA(BaseModel):
    tta: bool
class SetDebugTop1(BaseModel):
    debug_top1: bool

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self): self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try: await websocket.send_text(json.dumps(message))
        except Exception:
            self.disconnect(websocket)
    async def broadcast(self, message: dict):
        disconnected = []
        for conn in self.active_connections:
            try: await conn.send_text(json.dumps(message))
            except Exception:
                disconnected.append(conn)
        for c in disconnected: self.disconnect(c)

manager = ConnectionManager()

class UIConfig(BaseModel):
    show_bbox: bool
    show_label: bool

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
    ok, message = face_verify.reload_facebank()
    status = "success" if ok else "error"
    return {"status": status, "message": message}

@app.get("/status")
async def get_status():
    return {
        "status": "success",
        "app_initialized": face_verification_app is not None,
        "active_connections": len(manager.active_connections),
        "config": ui_config
    }

@app.get("/debug")
async def debug_state():
    try:
        from face_verify import learner, FaceVerificationApp
        model = learner.model if learner is not None else None
        conv1 = getattr(model, "conv1", None)
        conv1_in = int(getattr(conv1, "in_channels", 0)) if conv1 else None
        use_ffm = bool(getattr(model, "use_ffm", False)) if model else None
        app_tmp = FaceVerificationApp()
        tgt = app_tmp._targets_on_device()
        tshape = None if not isinstance(tgt, torch.Tensor) else tuple(tgt.shape)
        return {
            "network": getattr(config, "network", None),
            "threshold": float(face_verify.get_threshold()),
            "use_ffm": use_ffm,
            "conv1_in": conv1_in,
            "targets_shape": tshape,
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/facebank-info")
async def facebank_info_endpoint():
    try:
        info = face_verify.facebank_info()
        return {"status": "success", "data": info}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/runtime-config")
async def runtime_config():
    try:
        cfg = face_verify.get_runtime_config()
        return {"status": "success", "data": cfg}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/set-threshold")
async def set_threshold_api(req: SetThreshold):
    try:
        val = face_verify.set_threshold(req.threshold)
        await manager.broadcast({"type": "config", "data": {"threshold": float(val)}})
        return {"status": "success", "threshold": float(val)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/set-tta")
async def set_tta_api(req: SetTTA):
    try:
        cur = face_verify.set_tta(req.tta)
        return {"status": "success", "tta": bool(cur)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/set-debug-top1")
async def set_debug_top1_api(req: SetDebugTop1):
    try:
        cur = face_verify.set_debug_top1(req.debug_top1)
        return {"status": "success", "debug_top1": bool(cur)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Serve HTML
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        html_path = Path("templates/fastapi-index.html")
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        else:
            return HTMLResponse(content="<h1>Face Recognition Server</h1><p>Template missing.</p>")
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error</h1><pre>{e}</pre>")

# --- Frame processing ---
frame_skip_counter = 0

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    await manager.send_personal_message({"type": "config", "data": ui_config}, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if not face_verification_app:
                await manager.send_personal_message({"type": "error", "data": "Face verification app not initialized"}, websocket)
                continue
            await process_frame(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        traceback.print_exc()
        manager.disconnect(websocket)

async def process_frame(websocket: WebSocket, data: str):
    global frame_skip_counter
    try:
        # Decode base64 image
        encoded = data.split(",", 1)[1] if "," in data else data
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Skip to reduce load (process every 2nd frame)
        frame_skip_counter += 1
        if frame_skip_counter % 2 != 0:
            return

        # Get locations + recognition in one pass
        locs, recognition_data = face_verification_app.recognize_faces_and_locs(frame)

        # Send single combined packet (avoids client-side race)
        await manager.send_personal_message({
            "type": "frame_result",
            "locs": locs,
            "data": recognition_data,
            "meta": {"threshold": float(face_verify.get_threshold())}
        }, websocket)

        # Optional: let client toggle spinners
        await manager.send_personal_message({"type": "infer_status", "data": "done"}, websocket)

    except Exception:
        traceback.print_exc()


# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "app_initialized": face_verification_app is not None,
    }

if __name__ == "__main__":
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=True)
    except Exception:
        traceback.print_exc()
