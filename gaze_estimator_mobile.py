import os
import sys
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms

# -------------------------------------------------------
# Locate ONNX model inside gaze-estimation/weights
# -------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "gaze-estimation", "weights")
MODEL_PATH = os.path.join(WEIGHTS_DIR, "mobilenetv2_gaze.onnx")

if not os.path.isfile(MODEL_PATH):
    print(f"❌ ONNX model not found at {MODEL_PATH}")
    sys.exit(1)

print(f"✅ Loading gaze model from {MODEL_PATH}")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# -------------------------------------------------------
# Preprocessing transform
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# -------------------------------------------------------
# Public helper functions
# -------------------------------------------------------
def estimate_gaze_mobile(frame_bgr: np.ndarray, face_box: tuple) -> np.ndarray:
    """
    Estimate gaze (yaw, pitch) using the ONNX MobileGaze model.
    Args:
        frame_bgr: Current BGR frame from OpenCV
        face_box: (x, y, w, h) bounding box in pixel coords
    Returns:
        numpy array [yaw, pitch] in degrees
    """
    x, y, w, h = face_box
    if w <= 0 or h <= 0:
        return np.array([0.0, 0.0])

    crop = frame_bgr[y:y+h, x:x+w]
    if crop.size == 0:
        return np.array([0.0, 0.0])

    img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).numpy().astype(np.float32)

    ort_out = session.run(None, {input_name: tensor})[0].squeeze()
    return ort_out

def is_distracted_by_gaze_mobile(
    yaw_pitch: np.ndarray,
    yaw_thresh: float = 5.0,
    up_thresh: float = 1.5,
    down_thresh: float = 3.0
) -> bool:
    """
    Returns True if absolute yaw exceeds yaw_thresh,
    or pitch above up_thresh (looking up),
    or pitch below -down_thresh (looking down).
    """
    yaw, pitch = float(yaw_pitch[0]), float(yaw_pitch[1])
    return (
        abs(yaw) > yaw_thresh or
        pitch > up_thresh or
        pitch < -down_thresh
    )