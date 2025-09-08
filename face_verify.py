# face_verify.py

import cv2
from PIL import Image
import traceback
import argparse
import torch
from pathlib import Path
from torchvision import transforms as trans
from utils.utils_config import get_config
from Learner import face_learner
from utils.utils_facebank import load_facebank, draw_box_name, prepare_facebank
from face_detector import FaceDetector

parser = argparse.ArgumentParser(description='Chạy ứng dụng nhận dạng khuôn mặt')
parser.add_argument('-c', '--config', default='configs/infer.py', help='Đường dẫn đến file cấu hình.')
parser.add_argument('-d', '--model-dir', type=str, default=None, help='Ghi đè thư mục model.')
parser.add_argument('--score', action='store_true', help='Hiển thị điểm số confidence.')
parser.add_argument('--tta', action='store_true', help='Bật Test Time Augmentation.')
args = parser.parse_args()
conf = get_config(args.config)

# Sửa lỗi: thiết lập device mặc định nếu không có
if not hasattr(conf, 'device'):
    conf.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo = FaceDetector()
print('Face detector loaded')

if args.model_dir:
    print(f"[INFO] Ghi đè `config.output` bằng: '{args.model_dir}'")
    conf.output = args.model_dir
    dir_name = Path(args.model_dir).name
    if 'res50_custom2' in dir_name:
        conf.network = 'r50_custom_v2'
    elif 'res50_custom' in dir_name:
        conf.network = 'r50_custom_v1'
    elif 'res50_fan' in dir_name:
        conf.network = 'r50_fan_ffm'
    elif 'r100' in dir_name:
        conf.network = 'r100'
    print(f"[INFO] Tự động đặt `config.network` thành: '{conf.network}'")

print("\n--- Cấu hình Inference cuối cùng ---")
print(f"Kiến trúc model: {conf.network}")
print(f"Thư mục model:   {conf.output}")
print(f"File trọng số:    {conf.model_file}")
print(f"Ngưỡng nhận dạng: {conf.threshold}")
print("------------------------------------\n")

learner = face_learner(conf, inference=True)
learner.threshold = conf.threshold

model_file = conf.model_file if conf.device.type != 'cpu' else conf.cpu_model_file
learner.load_state(conf, model_file)
learner.model.eval()
print('Learner loaded')

# Load facebank để có thể sử dụng trong FaceVerificationApp
if conf.update_facebank:
    print("Initial facebank update enabled. Preparing facebank...")
    targets, names = prepare_facebank(conf, learner.model, yolo, tta=conf.tta)
    print('Facebank updated on startup.')
else:
    print("Loading existing facebank...")
    targets, names = load_facebank(conf)
    print('Facebank loaded on startup.')

if names:
    print(f'Found {len(names)} identities in facebank: {list(names.values()) if isinstance(names, dict) else names}')
else:
    print('Facebank is empty.')


class FaceVerificationApp:
    def __init__(self):
        self.height = 800
        self.width = 800
        self.image = None
        self.last_recognition_results = {}  # {face_id: {"name": name, "confidence": conf}}
        self._load_initial_facebank()

    def _load_initial_facebank(self):
        global targets, names
        if conf.update_facebank:
            print("Initial facebank update enabled. Preparing facebank...")
            self.targets, self.names = prepare_facebank(conf, learner.model, yolo, tta=conf.tta)
            print('Facebank updated on startup.')
        else:
            print("Loading existing facebank...")
            self.targets, self.names = load_facebank(conf)
            print('Facebank loaded on startup.')

        # Cập nhật biến global
        targets = self.targets
        names = self.names

        if self.names:
            names_list = list(self.names.values()) if isinstance(self.names, dict) else self.names
            print(f'Found {len(self.names)} identities in facebank: {names_list}')
        else:
            print('Facebank is empty.')

    def reload_facebank(self):
        global targets, names
        try:
            print("\n[API Request] Reloading facebank...")
            self.targets, self.names = prepare_facebank(conf, learner.model, yolo, tta=conf.tta)

            # Cập nhật biến global
            targets = self.targets
            names = self.names

            names_list = list(self.names.values()) if isinstance(self.names, dict) else self.names
            print(f"Facebank reloaded successfully. Found {len(self.names)} identities: {names_list}\n")
            return True, f"Successfully reloaded. Found {len(self.names)} identities."
        except Exception as e:
            print(f"[ERROR] Failed to reload facebank: {e}")
            traceback.print_exc()
            return False, "An error occurred while reloading the facebank."

    def get_face_locations(self, frame):
        face_locations = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            try:
                res = yolo.align_multi(image, conf.face_limit, conf.min_face_size)

                if res is not None:
                    bboxes, faces = res

                    print(
                        f"YOLO detection: bboxes shape: {bboxes.shape if bboxes is not None else None}, faces length: {len(faces) if faces is not None else None}")

                    if faces is not None and len(faces) > 0 and bboxes is not None and bboxes.size > 0:
                        # Fix: Xử lý bboxes đúng cách
                        if bboxes.ndim == 2:
                            # Nếu có confidence score, loại bỏ cột cuối
                            if bboxes.shape[1] > 4:
                                bboxes = bboxes[:, :4]

                        bboxes = bboxes.astype(int)

                        for idx, bbox in enumerate(bboxes):
                            face_id = f"face_{idx}"
                            face_locations.append({
                                "id": face_id,
                                "bbox": bbox.tolist()  # Convert numpy array to list [x1, y1, x2, y2]
                            })
            except (ValueError, AttributeError) as e:
                print(f"YOLO error: {e} - skipping this frame")
                return []
        except Exception as e:
            print(f"Error in get_face_locations: {e}")
            traceback.print_exc()

        return face_locations

    def recognize_faces(self, frame):
        recognition_data = {}

        if self.targets is None or len(self.targets) == 0:
            print("No facebank available for recognition")
            return recognition_data

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res = yolo.align_multi(image, conf.face_limit, conf.min_face_size)

            if res is not None:
                bboxes, faces = res
                print(
                    f"Recognition - bboxes shape: {bboxes.shape if bboxes is not None else None}, faces count: {len(faces) if faces is not None else None}")

                if faces is not None and len(faces) > 0 and bboxes is not None and bboxes.size > 0:
                    # Fix: Xử lý bboxes đúng cách
                    if bboxes.ndim == 2:
                        # Nếu có confidence score, loại bỏ cột cuối
                        if bboxes.shape[1] > 4:
                            bboxes = bboxes[:, :4]

                    bboxes = bboxes.astype(int)

                    try:
                        # Thực hiện inference
                        results, scores = learner.infer(conf, faces, self.targets, args.tta)
                        print(f"Inference results: {results}, scores: {scores}")

                        for idx, bbox in enumerate(bboxes):
                            face_id = f"face_{idx}"
                            name = "Unknown"
                            confidence = 0.0

                            if idx < len(results) and idx < len(scores):
                                result_idx = results[idx]
                                confidence = float(scores[idx])

                                print(
                                    f"Face {idx}: result_idx={result_idx}, confidence={confidence:.3f}, threshold={learner.threshold}")

                                # Fix: Điều kiện nhận dạng chính xác
                                if result_idx != -1 and confidence >= learner.threshold:
                                    # Xử lý names dạng dict hoặc list
                                    if isinstance(self.names, dict):
                                        name = self.names.get(result_idx, "Unknown")
                                    else:
                                        if result_idx < len(self.names):
                                            name = self.names[result_idx]
                                        else:
                                            name = "Unknown"

                                    print(f"Recognized: {name} with confidence {confidence:.3f}")
                                else:
                                    print(
                                        f"Below threshold or no match: result_idx={result_idx}, confidence={confidence:.3f}")

                            recognition_data[face_id] = {
                                "name": name,
                                "confidence": confidence
                            }

                        self.last_recognition_results = recognition_data
                        print(f"Final recognition data: {self.last_recognition_results}")

                    except Exception as inference_error:
                        print(f"Inference error: {inference_error}")
                        traceback.print_exc()
                        # Tạo recognition data mặc định cho các face được detect
                        for idx in range(len(bboxes)):
                            face_id = f"face_{idx}"
                            recognition_data[face_id] = {
                                "name": "Processing...",
                                "confidence": 0.0
                            }

        except Exception as e:
            print(f"Error in recognize_faces: {e}")
            traceback.print_exc()

        return recognition_data

    def process_frame(self, frame):
        """Legacy method for backward compatibility"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res = yolo.align_multi(image, conf.face_limit, conf.min_face_size)

            if res is not None:
                bboxes, faces = res
                if faces is not None and len(faces) > 0:
                    # Fix: Xử lý bboxes đúng cách
                    if bboxes.ndim == 2:
                        if bboxes.shape[1] > 4:
                            bboxes = bboxes[:, :4]

                    bboxes = bboxes.astype(int)

                    results, scores = learner.infer(conf, faces, self.targets, conf.tta)

                    for idx, bbox in enumerate(bboxes):
                        result_idx = results[idx]
                        confidence = float(scores[idx])

                        if result_idx == -1 or confidence < learner.threshold:
                            name = "Unknown"
                        else:
                            # Xử lý names dạng dict hoặc list
                            if isinstance(self.names, dict):
                                name = self.names.get(result_idx, "Unknown")
                            else:
                                if result_idx < len(self.names):
                                    name = self.names[result_idx]
                                else:
                                    name = "Unknown"

                        if args.score:
                            rgb_frame = draw_box_name(bbox, f"{name}_{confidence:.2f}", rgb_frame)
                        else:
                            rgb_frame = draw_box_name(bbox, name, rgb_frame)
                else:
                    cv2.putText(rgb_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(rgb_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception as e:
            print(f"Error in process_frame: {e}")
            traceback.print_exc()

        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return processed_frame
