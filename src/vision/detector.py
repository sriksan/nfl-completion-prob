"""YOLOv8 object detection wrapper for NFL broadcast frames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class YOLODetector:
    """Lazy-loaded YOLOv8 detector for players and ball in video frames."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        self.model_name = model_name
        self.confidence = confidence
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run YOLOv8 on a BGR frame and return person/ball detections."""
        model = self._load_model()
        results = model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in (COCO_PERSON_CLASS, COCO_SPORTS_BALL_CLASS):
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(box.conf[0]),
                        class_id=class_id,
                        class_name=names[class_id],
                    )
                )
        return detections

    def detect_from_path(self, image_path: Path) -> tuple[np.ndarray, list[Detection]]:
        """Load an image and run detection."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return frame, self.detect(frame)

    def detect_video_frame(
        self,
        video_path: Path,
        frame_index: int = 0,
    ) -> tuple[Optional[np.ndarray], list[Detection]]:
        """Extract a single frame from video and run YOLOv8."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, []

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        cap.release()

        if not success:
            return None, []
        return frame, self.detect(frame)
