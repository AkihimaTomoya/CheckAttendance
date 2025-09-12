import numpy as np
from PIL import Image
from ultralytics_custom import YOLO


class FaceDetector:
    """
    Thin wrapper around a YOLO face detector that provides:
      - detect_faces: returns [x1,y1,x2,y2,score] (Nx5) and empty 10-point landmarks for compatibility
      - align:        returns the first detected face cropped to 112x112
      - align_multi:  returns (boxes, list[PIL.Image 112x112]) for multiple faces
    """
    def __init__(
        self,
        model_path: str = "ultralytics_custom/models/yolo/detect/custom_1_0.6.pt",
        conf: float = 0.6,
        iou: float = 0.45,
        classes = [0],       # None = all classes; default is face class id 0 in custom model
        max_det: int = 100,
    ):
        self.model = YOLO(model_path)
        self.model.conf = conf
        self.model.iou = iou
        self.model.classes = classes
        self.model.agnostic_nms = True
        self.model.max_det = max_det

    @staticmethod
    def _expand_square(box, img_w: int, img_h: int, scale: float = 1.30):
        """Expand a box to a square with `scale`, clipped to image bounds."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return 0, 0, img_w, img_h
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(w, h) * scale
        nx1 = int(max(0, cx - side / 2))
        ny1 = int(max(0, cy - side / 2))
        nx2 = int(min(img_w, cx + side / 2))
        ny2 = int(min(img_h, cy + side / 2))
        return nx1, ny1, nx2, ny2

    def detect_faces(self, image, min_face_size: float = 20.0, conf_th: float | None = None, iou_th: float | None = None):
        """
        Returns:
            bounding_boxes: np.ndarray of shape (N, 5) with [x1, y1, x2, y2, score]
            landmarks:      np.ndarray of shape (N, 10) (empty for compatibility)
        """
        img_np = np.array(image)
        H, W = img_np.shape[:2]
        conf_use = self.model.conf if conf_th is None else conf_th
        iou_use = self.model.iou if iou_th is None else iou_th

        # First pass with configured classes
        results = self.model.predict(
            img_np,
            conf=conf_use,
            iou=iou_use,
            classes=self.model.classes,
            agnostic_nms=True,
            max_det=self.model.max_det,
            verbose=False,
        )

        # If empty, retry with classes=None (in case class id differs)
        if (len(results) == 0) or (len(results[0].boxes) == 0):
            if self.model.classes is not None:
                results = self.model.predict(
                    img_np,
                    conf=max(0.15, conf_use),
                    iou=iou_use,
                    classes=None,
                    agnostic_nms=True,
                    max_det=self.model.max_det,
                    verbose=False,
                )

        if len(results) == 0 or len(results[0].boxes) == 0:
            return np.zeros((0, 5)), np.zeros((0, 10))

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        valid = []
        for (x1, y1, x2, y2), sc in zip(boxes_xyxy, scores):
            if sc >= conf_use and (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
                valid.append([x1, y1, x2, y2, sc])

        if not valid:
            return np.zeros((0, 5)), np.zeros((0, 10))

        bounding_boxes = np.array(valid, dtype=np.float32)
        landmarks = np.zeros((len(bounding_boxes), 10), dtype=np.float32)
        return bounding_boxes, landmarks

    def align(self, img):
        """Return first detected face (PIL) cropped to 112x112, or None."""
        boxes, _ = self.detect_faces(img)
        if len(boxes) == 0:
            return None

        img_np = np.array(img)
        H, W = img_np.shape[:2]
        x1, y1, x2, y2, _ = boxes[0]
        nx1, ny1, nx2, ny2 = self._expand_square((x1, y1, x2, y2), W, H, scale=1.30)
        face = img_np[ny1:ny2, nx1:nx2]
        if face.size == 0:
            return None
        return Image.fromarray(face).resize((112, 112))

    def align_multi(self, img, limit: int | None = None, min_face_size: float = 30.0):
        """
        Returns:
            (boxes (N,5), faces: list[PIL.Image 112x112]) or None if no faces.
        """
        boxes, _ = self.detect_faces(img, min_face_size=min_face_size)
        if len(boxes) == 0:
            return None

        if limit:
            boxes = boxes[:limit]

        img_np = np.array(img)
        H, W = img_np.shape[:2]
        faces = []

        for x1, y1, x2, y2, _ in boxes:
            nx1, ny1, nx2, ny2 = self._expand_square((x1, y1, x2, y2), W, H, scale=1.30)
            face = img_np[ny1:ny2, nx1:nx2]
            if face.size == 0:
                continue
            faces.append(Image.fromarray(face).resize((112, 112)))

        if len(faces) == 0:
            return None
        return boxes, faces
