# fastapi-app.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import uvicorn
import argparse
from pathlib import Path
import time
import traceback
from face_verify import FaceVerificationApp
from utils.utils_config import get_config

# --- PHẦN KHỞI TẠO CẤU HÌNH ---
parser = argparse.ArgumentParser(description="FastAPI Face Recognition Server")
parser.add_argument('-c', '--config', default='configs/infer_config.py',
                    help='Đường dẫn đến file cấu hình cho inference.')
parser.add_argument('-d', '--model-dir', type=str, default=None,
                    help='Ghi đè đường dẫn đến thư mục model (config.output).')
args = parser.parse_args()
config = get_config(args.config)

if args.model_dir:
    print(f"[INFO] Ghi đè `config.output` bằng đường dẫn từ dòng lệnh: '{args.model_dir}'")
    config.output = args.model_dir
    dir_name = Path(args.model_dir).name
    if 'res50_custom2' in dir_name:
        config.network = 'r50_custom_v2'
    elif 'res50_custom' in dir_name:
        config.network = 'r50_custom_v1'
    elif 'res50_fan' in dir_name:
        config.network = 'r50_fan_ffm'
    elif 'r100' in dir_name:
        config.network = 'r100'
    print(f"[INFO] Tự động đặt `config.network` thành: '{config.network}'")

print("\n--- Cấu hình Inference cuối cùng cho FastAPI ---")
print(f"Kiến trúc model: {config.network}")
print(f"Thư mục model:   {config.output}")
print(f"File trọng số:    {config.model_file}")
print(f"Ngưỡng nhận dạng: {config.threshold}")
print("-------------------------------------------------\n")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Khởi tạo ứng dụng nhận dạng
face_verification_app = FaceVerificationApp()

# --- CÁC API ENDPOINT ---
ui_config = {"show_bbox": True, "show_label": True}


class UIConfig(BaseModel):
    show_bbox: bool
    show_label: bool


class ThresholdConfig(BaseModel):
    threshold: float


@app.post("/config")
async def update_ui_config(new_config: UIConfig):
    global ui_config
    ui_config.update(new_config.dict())
    print(f"Cập nhật cấu hình UI: {ui_config}")
    return {"status": "success", "config": ui_config}


@app.post("/threshold")
async def update_threshold(threshold_config: ThresholdConfig):
    old_threshold = face_verification_app.conf.threshold
    face_verification_app.conf.threshold = threshold_config.threshold
    print(f"Threshold updated from {old_threshold} to {threshold_config.threshold}")
    return {"status": "success", "old_threshold": old_threshold, "new_threshold": threshold_config.threshold}


@app.get("/threshold")
async def get_threshold():
    return {"threshold": face_verification_app.conf.threshold}


@app.post("/reload-facebank")
async def reload_facebank_endpoint():
    success, message = face_verification_app.reload_facebank()
    if success:
        return {"status": "success", "message": message}
    else:
        return {"status": "error", "message": message}


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/fastapi-index.html") as f:
        return HTMLResponse(content=f.read())


# Serve the HTML page
@app.get("/")
async def get_index():
    try:
        with open("templates/fastapi-index.html", "r") as f:
            content = f.read()
            return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<html><body><h1>Error: Template file not found</h1></body></html>")

last_recognition_time = 0
recognition_interval = 2
# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)

    # Send configuration
    await websocket.send_json({"type": "config", "data": config})

    try:
        while True:
            data = await websocket.receive_text()
            await process_frame(websocket, data)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# Process video frames
async def process_frame(websocket: WebSocket, data: str):
    global last_recognition_time
    current_time = time.time()

    try:
        # Decode the image
        if ',' in data:
            _, encoded = data.split(',', 1)
        else:
            encoded = data
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Face locations update - always
        face_locations = camera.get_face_locations(frame)
        await websocket.send_json({"type": "face_locations", "data": face_locations})

        # Recognition update - every 2 seconds
        if current_time - last_recognition_time >= recognition_interval:
            recognition_data = camera.recognize_faces(frame)
            await websocket.send_json({"type": "recognition_data", "data": recognition_data})
            last_recognition_time = current_time

    except Exception as e:
        print(f"Frame processing error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
