import torch 

VIDEO_SOURCE = "machinery_sample.mp4"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


YOLO_MODEL = 'yolo26n.pt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MOTION_THRESHOLD = 25
GAUSSIAN_BLUR_KERNAL = 21

ANOMALY_WINDOW_SIZE = 150
ANOMALY_SIGMA_MULTIPLIER = 3.0

