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
import json
from typing import List
from face_verify import FaceVerificationApp
from utils.utils_config import get_config

# --- PHẦN KHỞI TẠO CẤU HÌNH ---
parser = argparse.ArgumentParser(description="FastAPI Face Recognition Server")
parser.add_argument('-c', '--config', default='configs/infer.py',
                    help='Đường dẫn đến file cấu hình cho inference.')
parser.add_argument('-d', '--model-dir', type=str, default=None,
                    help='Ghi đè đường dẫn đến thư mục model (config.output).')
parser.add_argument('--host', default='0.0.0.0', help='Host để bind server')
parser.add_argument('--port', type=int, default=5050, help='Port để bind server')
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

app = FastAPI(title="Face Recognition API", version="1.0.0")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    print("Static files mounted successfully")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")

# Khởi tạo ứng dụng nhận dạng
try:
    face_verification_app = FaceVerificationApp()
    print("Face verification app initialized successfully")
except Exception as e:
    print(f"Error initializing face verification app: {e}")
    traceback.print_exc()
    face_verification_app = None

# --- CÁC API ENDPOINT ---
ui_config = {"show_bbox": True, "show_label": True}


# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


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

    # Broadcast config update to all connected clients
    await manager.broadcast({"type": "config", "data": ui_config})

    return {"status": "success", "config": ui_config}


@app.get("/config")
async def get_ui_config():
    return {"status": "success", "config": ui_config}


@app.post("/threshold")
async def update_threshold(threshold_config: ThresholdConfig):
    if not face_verification_app:
        return {"status": "error", "message": "Face verification app not initialized"}

    try:
        from face_verify import learner
        old_threshold = learner.threshold
        learner.threshold = threshold_config.threshold
        print(f"Threshold updated from {old_threshold} to {threshold_config.threshold}")
        return {"status": "success", "old_threshold": old_threshold, "new_threshold": threshold_config.threshold}
    except Exception as e:
        print(f"Error updating threshold: {e}")
        return {"status": "error", "message": f"Failed to update threshold: {str(e)}"}


@app.get("/threshold")
async def get_threshold():
    if not face_verification_app:
        return {"status": "error", "message": "Face verification app not initialized"}

    try:
        from face_verify import learner
        return {"status": "success", "threshold": learner.threshold}
    except Exception as e:
        print(f"Error getting threshold: {e}")
        return {"status": "error", "message": f"Failed to get threshold: {str(e)}"}


@app.post("/reload-facebank")
async def reload_facebank_endpoint():
    if not face_verification_app:
        return {"status": "error", "message": "Face verification app not initialized"}

    try:
        success, message = face_verification_app.reload_facebank()
        status = "success" if success else "error"
        return {"status": status, "message": message}
    except Exception as e:
        print(f"Error in reload facebank endpoint: {e}")
        traceback.print_exc()
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}


@app.get("/status")
async def get_status():
    return {
        "status": "success",
        "app_initialized": face_verification_app is not None,
        "active_connections": len(manager.active_connections),
        "config": ui_config
    }


# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        html_path = Path("templates/fastapi-index.html")
        if html_path.exists():
            with open(html_path, "r", encoding='utf-8') as f:
                content = f.read()
                return HTMLResponse(content=content)
        else:
            # Fallback HTML if template file not found
            fallback_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Face Recognition - Server Running</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                    .error { color: #e74c3c; }
                    .info { color: #3498db; }
                </style>
            </head>
            <body>
                <h1>Face Recognition Server</h1>
                <p class="error">Template file not found!</p>
                <p class="info">Please make sure 'templates/fastapi-index.html' exists.</p>
                <p>Server is running on: <strong>http://localhost:5050</strong></p>
            </body>
            </html>
            """
            return HTMLResponse(content=fallback_html)
    except Exception as e:
        error_html = f"""
        <html>
        <body>
            <h1>Error Loading Page</h1>
            <p>Error: {str(e)}</p>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)


# Global variables for frame processing
last_recognition_time = 0
recognition_interval = 2.0  # seconds
frame_skip_counter = 0


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    # Send initial configuration
    await manager.send_personal_message({"type": "config", "data": ui_config}, websocket)

    try:
        while True:
            data = await websocket.receive_text()
            if face_verification_app:
                await process_frame(websocket, data)
            else:
                await manager.send_personal_message({
                    "type": "error",
                    "data": "Face verification app not initialized"
                }, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected normally")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        manager.disconnect(websocket)


# Process video frames
async def process_frame(websocket: WebSocket, data: str):
    global last_recognition_time, frame_skip_counter
    current_time = time.time()

    try:
        # Decode the image
        if ',' in data:
            header, encoded = data.split(',', 1)
        else:
            encoded = data

        try:
            img_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as decode_error:
            print(f"Error decoding frame: {decode_error}")
            return

        if frame is None:
            print("Received null frame")
            return

        # Skip some frames to reduce processing load
        frame_skip_counter += 1
        if frame_skip_counter % 2 != 0:  # Process every 2nd frame
            return

        try:
            # Face locations update - always (but throttled)
            face_locations = face_verification_app.get_face_locations(frame)
            await manager.send_personal_message({"type": "face_locations", "data": face_locations}, websocket)

            # Recognition update - every N seconds
            if current_time - last_recognition_time >= recognition_interval:
                print(f"Running recognition at {current_time:.2f}")
                recognition_data = face_verification_app.recognize_faces(frame)
                await manager.send_personal_message({"type": "recognition_data", "data": recognition_data}, websocket)
                last_recognition_time = current_time

        except Exception as processing_error:
            print(f"Error processing frame: {processing_error}")
            traceback.print_exc()

    except Exception as e:
        print(f"Frame processing error: {e}")
        traceback.print_exc()


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "app_initialized": face_verification_app is not None
    }


if __name__ == "__main__":
    print(f"\n🚀 Starting FastAPI Face Recognition Server...")
    print(f"📡 Server will be available at: http://{args.host}:{args.port}")
    print(f"🎯 Access the web interface at: http://localhost:{args.port}")
    print(f"📊 API docs at: http://localhost:{args.port}/docs")
    print(f"🔍 Health check at: http://localhost:{args.port}/health")

    if not face_verification_app:
        print("⚠️  Warning: Face verification app failed to initialize!")
    else:
        print("✅ Face verification app initialized successfully")

    print("-" * 50)

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        traceback.print_exc()
