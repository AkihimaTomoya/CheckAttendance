import numpy as np
from PIL import Image
from ultralytics_custom import YOLO


class FaceDetector:
    def __init__(self):
        model_path = 'ultralytics_custom/models/yolo/detect/custom_1_0.6.pt'
        self.model = YOLO(model_path)
        self.model.conf = 0.6
        self.model.iou = 0.5
        self.model.classes = [0]
        self.model.agnostic_nms = True
        self.model.max_det = 1000

    def detect_faces(self, image, min_face_size=20.0, thresholds=[0.5, 0.6], nms_thresholds=[0.5]):
        """
        Detect faces using YOLO.

        Args:
            image (PIL.Image): Input image.
            min_face_size (float): Minimum size of face to be considered.

        Returns:
            bounding_boxes: ndarray (n, 5) - [x1, y1, x2, y2, score]
            landmarks: empty ndarray (n, 10) for compatibility
        """
        img_np = np.array(image)
        results = self.model.predict(img_np, conf=thresholds[1], iou=nms_thresholds[0],
                                     classes=[0], agnostic_nms=True, max_det=1000)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.zeros((0, 5)), np.zeros((0, 10))

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        valid = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            if score >= thresholds[0] and (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
                valid.append([x1, y1, x2, y2, score])

        if not valid:
            return np.zeros((0, 5)), np.zeros((0, 10))

        bounding_boxes = np.array(valid)
        landmarks = np.zeros((len(bounding_boxes), 10))  # No landmarks from YOLO
        return bounding_boxes, landmarks

    def align(self, img):
        """
        Align a single face by cropping and resizing to 112x112.

        Returns:
            PIL.Image: Aligned face or original image if none found.
        """
        boxes, _ = self.detect_faces(img)
        if len(boxes) == 0:
            return img

        x1, y1, x2, y2, _ = boxes[0]
        face = np.array(img)[int(y1):int(y2), int(x1):int(x2)]
        face_resized = Image.fromarray(face).resize((112, 112))
        return face_resized

    def align_multi(self, img, limit=None, min_face_size=30.0):
        """
        Align multiple faces from image.

        Returns:
            boxes (n, 5) and list of PIL.Image aligned faces (112x112)
        """
        boxes, _ = self.detect_faces(img, min_face_size)
        if limit:
            boxes = boxes[:limit]

        faces = []
        img_np = np.array(img)
        for box in boxes:
            x1, y1, x2, y2, _ = box
            face = img_np[int(y1):int(y2), int(x1):int(x2)]
            face_img = Image.fromarray(face).resize((112, 112))
            faces.append(face_img)

        return boxes, faces
