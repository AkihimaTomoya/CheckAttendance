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


class FaceVerificationApp:
    def __init__(self, conf):
        self.conf = conf
        self.detector = FaceDetector()
        print('Face detector loaded')

        self.learner = face_learner(self.conf, inference=True)
        self.learner.threshold = self.conf.threshold

        model_file = self.conf.model_file if self.conf.device.type != 'cpu' else self.conf.cpu_model_file
        self.learner.load_state(self.conf, model_file)
        self.learner.model.eval()
        print('Learner loaded')

        self._load_initial_facebank()

    def _load_initial_facebank(self):
        if self.conf.update_facebank:
            print("Initial facebank update enabled. Preparing facebank...")
            self.targets, self.names = prepare_facebank(self.conf, self.learner.model, self.detector, tta=self.conf.tta)
            print('Facebank updated on startup.')
        else:
            print("Loading existing facebank...")
            self.targets, self.names = load_facebank(self.conf)
            print('Facebank loaded on startup.')

        if self.names:
            print(f'Found {len(self.names)} identities in facebank: {self.names}')
        else:
            print('Facebank is empty.')

    def _detect_and_align_faces(self, image):
        try:
            res = self.detector.align_multi(image, self.conf.face_limit, self.conf.min_face_size)
            if res is None:
                return None, None
            bboxes, faces = res
            return bboxes, faces
        except Exception as e:
            print(f"Error during face detection/alignment: {e}")
            traceback.print_exc()
            return None, None

    def reload_facebank(self):
        try:
            print("\n[API Request] Reloading facebank...")
            self.targets, self.names = prepare_facebank(self.conf, self.learner.model, self.detector, tta=self.conf.tta)
            print(f"Facebank reloaded successfully. Found {len(self.names)} identities: {self.names}\n")
            return True, f"Successfully reloaded. Found {len(self.names)} identities."
        except Exception as e:
            print(f"[ERROR] Failed to reload facebank: {e}")
            traceback.print_exc()
            return False, "An error occurred while reloading the facebank."

    def detect_and_recognize(self, frame, recognize=False):
        """
        Thực hiện cả phát hiện và nhận dạng trong một lần để đảm bảo tính nhất quán.
        :param frame: Khung hình đầu vào (OpenCV BGR)
        :param recognize: Cờ để quyết định có chạy mô hình nhận dạng (chậm) hay không
        :return: (list) face_locations, (dict) recognition_data
        """
        face_locations = []
        recognition_data = {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        try:
            # 1. Phát hiện và căn chỉnh khuôn mặt (CHỈ CHẠY 1 LẦN)
            bboxes, faces = self._detect_and_align_faces(image)

            if faces is None or len(faces) == 0:
                return face_locations, recognition_data

            # 2. Luôn trả về vị trí
            for idx, bbox in enumerate(bboxes):
                face_id = f"face_{idx}"
                face_locations.append({
                    "id": face_id,
                    "bbox": (bbox[:-1].astype(int)).tolist()
                })

            # 3. Chỉ nhận dạng nếu cờ `recognize` là True
            if recognize:
                results, score = self.learner.infer(self.conf, faces, self.targets, self.conf.tta)

                for idx, _ in enumerate(bboxes):
                    face_id = f"face_{idx}"
                    confidence = float('{:.2f}'.format(score[idx]))

                    if confidence < 0.80:
                        name = "Unknown"
                    else:
                        name = self.names.get(results[idx].item(), "Unknown")

                    recognition_data[face_id] = {
                        "name": name,
                        "confidence": confidence
                    }
        except Exception as e:
            print(f"Error in detect_and_recognize: {e}")
            traceback.print_exc()

        return face_locations, recognition_data

    def process_frame(self, frame, show_bbox=True, show_label=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = frame.copy()
        try:
            image = Image.fromarray(rgb_frame)
            bboxes, faces = self._detect_and_align_faces(image)
            if faces is None or len(faces) == 0:
                return processed_frame

            results, score = self.learner.infer(self.conf, faces, self.targets, self.conf.tta)

            for idx, bbox in enumerate(bboxes):
                if show_bbox:
                    bbox_coords = bbox[:-1].astype(int) + [-1, -1, 1, 1]
                    display_text = ""
                    if show_label:
                        confidence = score[idx]
                        if confidence < 0.80:
                            name = "Unknown"
                        else:
                            name = self.names.get(results[idx].item(), "Unknown")

                        display_text = name
                        if self.conf.show_score:
                            display_text = f"{name} ({confidence:.2f})"
                    processed_frame = draw_box_name(bbox_coords, display_text, processed_frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            traceback.print_exc()
        return processed_frame

    def get_face_locations(self, frame):
        face_locations = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            bboxes, _ = self._detect_and_align_faces(image)

            if bboxes is not None and len(bboxes) > 0:
                for idx, bbox in enumerate(bboxes):
                    face_id = f"face_{idx}"
                    face_locations.append({
                        "id": face_id,
                        "bbox": (bbox[:-1].astype(int)).tolist()
                    })
        except Exception as e:
            print(f"Error in get_face_locations: {e}")
            traceback.print_exc()
        return face_locations

    def recognize_faces(self, frame):
        recognition_data = {}
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            image = Image.fromarray(rgb_frame)
            bboxes, faces = self._detect_and_align_faces(image)

            if faces is not None and len(faces) > 0:
                results, score = self.learner.infer(self.conf, faces, self.targets, self.conf.tta)

                for idx, _ in enumerate(bboxes):
                    face_id = f"face_{idx}"
                    confidence = float('{:.2f}'.format(score[idx]))

                    if confidence < 0.80:
                        name = "Unknown"
                    else:
                        name = self.names.get(results[idx].item(), "Unknown")

                    recognition_data[face_id] = {
                        "name": name,
                        "confidence": confidence
                    }
        except Exception as e:
            print(f"Error in recognize_faces: {e}")
            traceback.print_exc()
        return recognition_data

    def run_on_webcam(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        print("\n--- Starting webcam feed ---")
        print("Press 'q' to quit.")
        print("Press 'r' to reload facebank.")
        print("----------------------------\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Face Verification', processed_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed, shutting down.")
                break
            elif key == ord('r'):
                # NHẤN 'r' ĐỂ TẢI LẠI FACEBANK
                self.reload_facebank()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chạy ứng dụng nhận dạng khuôn mặt')
    parser.add_argument('-c', '--config', default='configs/infer_config.py', help='Đường dẫn đến file cấu hình.')
    parser.add_argument('-d', '--model-dir', type=str, default=None, help='Ghi đè thư mục model.')
    args = parser.parse_args()

    config = get_config(args.config)

    if args.model_dir:
        print(f"[INFO] Ghi đè `config.output` bằng: '{args.model_dir}'")
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

    print("\n--- Cấu hình Inference cuối cùng ---")
    print(f"Kiến trúc model: {config.network}")
    print(f"Thư mục model:   {config.output}")
    print(f"File trọng số:    {config.model_file}")
    print(f"Ngưỡng nhận dạng: {config.threshold}")
    print("------------------------------------\n")

    app = FaceVerificationApp(config)
    app.run_on_webcam()