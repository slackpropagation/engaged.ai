# attention_tracker.py

import sys
import os
import cv2
import threading
import numpy as np

# Ensure local modules and repo are importable
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, 'gaze-estimation'))

import mediapipe as mp
from distraction_eye import is_distracted_by_eye_position
from distraction_pose import is_distracted_by_head_tilt
from distraction_idle import IdleDistractionDetector
from challenge_popup import show_challenge
from logger import SessionLogger
from gaze_estimator_mobile import estimate_gaze_mobile, is_distracted_by_gaze_mobile

# Initialize MediaPipe FaceMesh with iris refinement
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Initialize detectors and logger
idle_detector = IdleDistractionDetector(timeout_seconds=5)

session_logger = SessionLogger()
# ---- Gaze voting state ----
gaze_counter = 0          # consecutive distracted frames
gaze_required = 3         # frames required to confirm state change
gaze_status = False       # True = currently distracted by gaze

# Prepare video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera index 0")
    sys.exit(1)
else:
    print("✅ Using camera index 0")

print("Press Esc to exit.")

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror for selfie view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Process with FaceMesh
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Track idle (face missing)
    face_present = bool(results.multi_face_landmarks)
    idle_detector.update_activity(face_present)
    if idle_detector.is_idle():
        cv2.putText(frame, "🔴 Idle / No Face Detected", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        session_logger.log_distraction("idle")
        threading.Thread(target=show_challenge).start()

    # If face landmarks found, check distractions
    if face_present:
        for face_landmarks in results.multi_face_landmarks:
            # Compute bounding box for gaze estimation
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            box = (
                int(x_min * width),
                int(y_min * height),
                int((x_max - x_min) * width),
                int((y_max - y_min) * height)
            )

            # ----- Gaze-based distraction with voting -----
            gaze_vec = estimate_gaze_mobile(frame, box)
            if is_distracted_by_gaze_mobile(gaze_vec):
                gaze_counter = min(gaze_counter + 1, gaze_required)
            else:
                gaze_counter = max(gaze_counter - 1, 0)

            # Update gaze_status when counter crosses threshold
            if not gaze_status and gaze_counter >= gaze_required:
                gaze_status = True
                session_logger.log_distraction("gaze")
                threading.Thread(target=show_challenge).start()
            elif gaze_status and gaze_counter == 0:
                gaze_status = False

            # Draw current gaze state label
            if gaze_status:
                cv2.putText(frame, "🔴 Gaze Distraction", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "🟢 Focused", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # Eye position distraction
            if is_distracted_by_eye_position(face_landmarks, width):
                cv2.putText(frame, "🔴 Eye Distraction", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                session_logger.log_distraction("eye")
                threading.Thread(target=show_challenge).start()
                break

            # Head tilt distraction
            if is_distracted_by_head_tilt(face_landmarks, width, height):
                cv2.putText(frame, "🔴 Head Tilt Detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                session_logger.log_distraction("head_tilt")
                threading.Thread(target=show_challenge).start()
                break

            # Focused
            cv2.putText(frame, "🟢 Focused", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # Display frame
    cv2.imshow("Engaged.ai - Attention Tracker", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Esc key
        break

# Cleanup
cap.release()
session_logger.end_session()
cv2.destroyAllWindows()
