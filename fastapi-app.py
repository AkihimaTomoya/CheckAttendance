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

# --- PHẦN KHỞI TẠO CẤU HÌNH (Giữ nguyên từ file mới của bạn) ---
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
face_verification_app = FaceVerificationApp(config)

# --- CÁC API ENDPOINT (Giữ nguyên) ---
ui_config = {"show_bbox": True, "show_label": True}


class UIConfig(BaseModel):
    show_bbox: bool
    show_label: bool


@app.post("/config")
async def update_ui_config(new_config: UIConfig):
    global ui_config
    ui_config.update(new_config.dict())
    print(f"Cập nhật cấu hình UI: {ui_config}")
    # Gửi cấu hình mới đến tất cả các client đang kết nối (tùy chọn)
    return {"status": "success", "config": ui_config}


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


# --- WEBSOCKET ENDPOINT (Cập nhật với logic mới) ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client đã kết nối.")
    await websocket.send_json({"type": "config", "data": ui_config})

    last_recognition_time = 0
    recognition_interval = 2  # Giây

    try:
        while True:
            data = await websocket.receive_text()
            img_data = base64.b64decode(data.split(',')[1])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            current_time = time.time()
            # Quyết định xem có cần chạy nhận dạng trong lần này không
            should_recognize = (current_time - last_recognition_time) >= recognition_interval

            # Gọi hàm tích hợp mới
            face_locations, recognition_data = face_verification_app.detect_and_recognize(
                frame,
                recognize=should_recognize
            )

            # Luôn gửi vị trí
            await websocket.send_json({"type": "face_locations", "data": face_locations})

            # Chỉ gửi dữ liệu nhận dạng khi nó được thực hiện
            if should_recognize:
                await websocket.send_json({"type": "recognition_data", "data": recognition_data})
                last_recognition_time = current_time

    except WebSocketDisconnect:
        print("Client đã ngắt kết nối.")
    except Exception as e:
        print(f"Lỗi WebSocket: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")