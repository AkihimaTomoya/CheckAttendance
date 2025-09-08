# face_verify.py

import cv2
from PIL import Image
import traceback
import argparse
from pathlib import Path
from torchvision import transforms as trans
from utils.utils_config import get_config
from Learner import face_learner
from utils.utils_facebank import load_facebank, draw_box_name, prepare_facebank
from face_detector import FaceDetector

parser = argparse.ArgumentParser(description='Chạy ứng dụng nhận dạng khuôn mặt')
parser.add_argument('-c', '--config', default='configs/infer_config.py', help='Đường dẫn đến file cấu hình.')
parser.add_argument('-d', '--model-dir', type=str, default=None, help='Ghi đè thư mục model.')
args = parser.parse_args()
conf = get_config(args.config)
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

class FaceVerificationApp:
    def __init__(self):
        self.height = 800
        self.width = 800
        self.image = None
        self.last_recognition_results = {}  # {face_id: {"name": name, "confidence": conf}}
        self._load_initial_facebank()

    def _load_initial_facebank(self):
        if conf.update_facebank:
            print("Initial facebank update enabled. Preparing facebank...")
            self.targets, self.names = prepare_facebank(conf, learner.model, yolo, tta=conf.tta)
            print('Facebank updated on startup.')
        else:
            print("Loading existing facebank...")
            self.targets, self.names = load_facebank(conf)
            print('Facebank loaded on startup.')

        if self.names:
            print(f'Found {len(self.names)} identities in facebank: {self.names}')
        else:
            print('Facebank is empty.')

    def reload_facebank(self):
        try:
            print("\n[API Request] Reloading facebank...")
            self.targets, self.names = prepare_facebank(conf, learner.model, yolo, tta=conf.tta)
            print(f"Facebank reloaded successfully. Found {len(self.names)} identities: {self.names}\n")
            return True, f"Successfully reloaded. Found {len(self.names)} identities."
        except Exception as e:
            print(f"[ERROR] Failed to reload facebank: {e}")
            traceback.print_exc()
            return False, "An error occurred while reloading the facebank."

    def process_frame(self, frame):
        # Cũ
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res = yolo.align_multi(image, conf.face_limit, conf.min_face_size)
            print("YOLO result:", res)
            if res is not None:
                bboxes, faces = res
                if faces is not None and len(faces) > 0:
                    bboxes = bboxes[:, :-1].astype(int)
                    bboxes = bboxes + [-1, -1, 1, 1]
                    print("Bounding boxes:", bboxes)
                    results, score = learner.infer(conf, faces, self.targets, args.tta)
                    for idx, bbox in enumerate(bboxes):

                        if float('{:.2f}'.format(score[idx])) < .80:
                            name = "Unknown"
                        else:
                            name = names[results[idx] + 1]
                        if args.score:
                            rgb_frame = draw_box_name(bbox, f"{name}_{score[idx]:.2f}",
                                                      rgb_frame)
                        else:
                            rgb_frame = draw_box_name(bbox, name, rgb_frame)
                else:
                    # Nếu không có khuôn mặt, vẽ thông báo
                    cv2.putText(rgb_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                cv2.putText(rgb_frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except Exception as e:
            print(e)
            pass
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return processed_frame

    def get_face_locations(self, frame):
        face_locations = []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            try:
                res = yolo.align_multi(image, conf.face_limit, conf.min_face_size)

                if res is not None:
                    bboxes, faces = res

                    # Debug
                    print(
                        f"MTCNN detection: bboxes shape: {bboxes.shape if bboxes is not None else None}, faces length: {len(faces) if faces is not None else None}")

                    if faces is not None and len(faces) > 0 and bboxes is not None and bboxes.size > 0:
                        bboxes = bboxes[:, :-1].astype(int)
                        bboxes = bboxes + [-1, -1, 1, 1]

                        # Chỉ trả về các tọa độ của bounding box
                        for idx, bbox in enumerate(bboxes):
                            face_id = f"face_{idx}"
                            face_locations.append({
                                "id": face_id,
                                "bbox": bbox.tolist()  # Convert numpy array to list [x1, y1, x2, y2]
                            })
            except ValueError as e:
                print(f"MTCNN error: {e} - skipping this frame")
        except Exception as e:
            print(f"Error in get_face_locations: {e}")
            import traceback
            traceback.print_exc()  # In ra stack trace đầy đủ

        return face_locations

    def recognize_faces(self, frame):
        recognition_data = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            res = yolo.align_multi(image, conf.face_limit, conf.min_face_size)

            if res is not None:
                bboxes, faces = res

                if faces is not None and len(faces) > 0 and bboxes is not None and bboxes.size > 0:
                    bboxes = bboxes[:, :-1].astype(int)
                    bboxes = bboxes + [-1, -1, 1, 1]
                    results, score = learner.infer(conf, faces, targets, args.tta)

                    # Lưu trữ kết quả nhận dạng cho mỗi khuôn mặt
                    for idx, bbox in enumerate(bboxes):
                        face_id = f"face_{idx}"
                        name = "Unknown"
                        confidence = float('{:.2f}'.format(score[idx]))

                        if confidence >= 0.80:
                            name = names[results[idx] + 1]

                        recognition_data[face_id] = {
                            "name": name,
                            "confidence": confidence
                        }

                    # Cập nhật kết quả nhận dạng
                    self.last_recognition_results = recognition_data
                    print(self.last_recognition_results)
        except Exception as e:
            print(f"Error in recognize_faces: {e}")
            import traceback
            traceback.print_exc()  # In ra stack trace đầy đủ

        return recognition_data
