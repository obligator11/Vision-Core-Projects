import cv2
import numpy as np

class DrawingUtils:
    """Utility class to render architectural annotations on raw image frames safely."""
    
    @staticmethod
    def draw_polyline(frame: cv2.Mat, coordinates: np.ndarray, closed: bool, color: tuple[int, int, int], thickness: int = 1) -> None:
        """Draws sequence tracking connected contours across historical indices checkpoints safely."""
        pts = coordinates.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], closed, color, thickness, lineType=cv2.LINE_AA)

    @staticmethod
    def draw_bounding_box_with_label(frame: cv2.Mat, top_left: tuple[int, int], bottom_right: tuple[int, int], label: str, color: tuple[int, int, int]) -> None:
        """Constructs safe boundaries boxes containing explicit dynamic notification banners overlay strings."""
        cv2.rectangle(frame, top_left, bottom_right, color, 2, cv2.LINE_AA)
        
        # Overlay structural text tag backdrops
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        text_w, text_h = text_size[0], text_size[1]
        cv2.rectangle(frame, top_left, (top_left[0] + text_w + 10, top_left[1] - text_h - 10), color, -1)
        cv2.putText(frame, label, (top_left[0] + 5, top_left[1] - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)