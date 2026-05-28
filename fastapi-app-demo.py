import csv
import logging
import asyncio
import base64
import json
import argparse
import torch
import cv2
import numpy as np
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utils.utils_config import get_config

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("face_app")

# --- PATHS & AUTO-INIT ---
BASE_DIR = Path(__file__).parent
CFG_DIR = BASE_DIR / "cfg"
LOGS_DIR = BASE_DIR / "logs"
USERS_PATH = CFG_DIR / "users.json"
ATT_CONFIG_PATH = CFG_DIR / "attendance_config.json"

CFG_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

if not USERS_PATH.exists():
    USERS_PATH.write_text(json.dumps({
        "admin": {"password": "admin", "role": "admin", "display_name": "Quản trị viên", "face_name": "Admin"},
        "SV001": {"password": "123", "role": "student", "display_name": "Nguyễn Văn An", "face_name": "An"}
    }, indent=2, ensure_ascii=False), encoding="utf-8")

if not ATT_CONFIG_PATH.exists():
    ATT_CONFIG_PATH.write_text(json.dumps({
        "course_id": "CS101", "course_name": "Lập trình Web", "class_name": "K24A",
        "class_start": "08:00", "class_end": "10:00"
    }, indent=2, ensure_ascii=False), encoding="utf-8")

# --- MODEL HINTS & CLI ---
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
            if key in MODEL_BEST_THRESHOLDS: cfg.threshold = MODEL_BEST_THRESHOLDS[key]
            return

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="configs/infer.py")
parser.add_argument("-d", "--model-dir", type=str, default=None)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=5050)
args = parser.parse_args()

config = get_config(args.config)
if not hasattr(config, "device"): config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.model_dir: apply_model_overrides(config, args.model_dir)

# --- INIT AI ---
try:
    import importlib
    face_verify = importlib.import_module("face_verify")
    face_verify.initialize(config, update_facebank=True)
    face_verification_app = face_verify.FaceVerificationApp()
except Exception as e:
    log.error(f"AI Module Error: {e}")
    face_verification_app = None

# --- FASTAPI SETUP ---
app = FastAPI()
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

class LoginReq(BaseModel): username: str; password: str
class NewStudent(BaseModel): uid: str; password: str; display_name: str; face_name: str
class LogEntry(BaseModel): student_id: str; display_name: str; face_name: str; event: str; total_absent_sec: int = 0

def get_users(): return json.loads(USERS_PATH.read_text(encoding="utf-8"))
def save_users(d): USERS_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

# --- API ENDPOINTS ---
@app.post("/api/login")
async def api_login(req: LoginReq):
    users = get_users()
    u = req.username.strip()
    if u in users and users[u].get("password") == req.password:
        user_info = users[u].copy()
        user_info.pop("password", None)
        user_info["uid"] = u
        return {"status": "ok", "user": user_info}
    return {"status": "error", "message": "Sai tài khoản hoặc mật khẩu!"}

@app.get("/api/users")
async def api_get_users():
    return {k: {**v, "password": None} for k, v in get_users().items()}

@app.post("/api/users")
async def api_add_user(student: NewStudent):
    users = get_users()
    uid = student.uid.strip().upper()
    if uid in users: return {"status": "error", "message": f"Mã sinh viên '{uid}' đã tồn tại!"}
    users[uid] = {"password": student.password, "role": "student", "display_name": student.display_name.strip(), "face_name": student.face_name.strip()}
    save_users(users)
    return {"status": "ok", "message": f"Đã thêm sinh viên {uid}."}

@app.delete("/api/users/{uid}")
async def api_delete_user(uid: str):
    users = get_users()
    if uid not in users: return {"status": "error", "message": "Không tìm thấy!"}
    if users[uid].get("role") == "admin": return {"status": "error", "message": "Không thể xóa Admin!"}
    del users[uid]
    save_users(users)
    return {"status": "ok", "message": f"Đã xóa {uid}."}

@app.get("/attendance-config")
async def get_att_config():
    return json.loads(ATT_CONFIG_PATH.read_text(encoding="utf-8"))

@app.post("/log-entry")
async def post_log_entry(entry: LogEntry):
    path = LOGS_DIR / f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file: w.writerow(["timestamp", "student_id", "display_name", "event", "total_absent_sec"])
        w.writerow([datetime.now().strftime("%H:%M:%S"), entry.student_id, entry.display_name, entry.event, entry.total_absent_sec])
    return {"status": "ok"}

# NÚT RELOAD FACEBANK
@app.post("/api/reload-facebank")
async def api_reload_facebank():
    if not face_verification_app:
        return {"status": "error", "message": "AI chưa được khởi tạo!"}
    try:
        ok, msg = face_verify.reload_facebank()
        return {"status": "ok" if ok else "error", "message": msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- HTML ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def get_index(): return (BASE_DIR / "templates/index.html").read_text(encoding="utf-8")
@app.get("/student", response_class=HTMLResponse)
async def get_student(): return (BASE_DIR / "templates/student.html").read_text(encoding="utf-8")
@app.get("/admin", response_class=HTMLResponse)
async def get_admin(): return (BASE_DIR / "templates/admin.html").read_text(encoding="utf-8")

# --- WEBSOCKET ---
_executor = ThreadPoolExecutor(max_workers=1)
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue = asyncio.Queue(maxsize=1)
    async def processor():
        while True:
            raw = await queue.get()
            if not face_verification_app: continue
            try:
                img_data = base64.b64decode(raw.split(",")[1])
                frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                locs, rec_data = await asyncio.get_event_loop().run_in_executor(_executor, face_verification_app.recognize_faces_and_locs, frame)
                await ws.send_text(json.dumps({"type": "frame_result", "locs": locs, "data": rec_data}))
            except Exception as e: log.error(f"WS Proc Error: {e}")

    pt = asyncio.create_task(processor())
    try:
        while True:
            raw = await ws.receive_text()
            if not queue.empty(): queue.get_nowait()
            await queue.put(raw)
    except: pt.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)