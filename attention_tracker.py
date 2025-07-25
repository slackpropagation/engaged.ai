# attention_tracker.py
import cv2
import mediapipe as mp
from distraction_eye import is_distracted_by_eye_position
from distraction_pose import is_distracted_by_head_tilt
from distraction_idle import IdleDistractionDetector
from challenge_popup import show_challenge
from logger import SessionLogger
import threading

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True  # Enables iris landmarks (468–478)
)

idle_detector = IdleDistractionDetector(timeout_seconds=5)
session_logger = SessionLogger()

import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open camera at index 0.")
    exit()
else:
    print("✅ Using camera index 0")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view display
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    height, width, _ = frame.shape

    face_detected = bool(results.multi_face_landmarks)
    idle_detector.update_activity(face_detected)

    if idle_detector.is_idle():
        cv2.putText(frame, "🔴 Idle / No Face", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        session_logger.log_distraction("idle")
        threading.Thread(target=show_challenge).start()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if is_distracted_by_eye_position(face_landmarks, width):
                cv2.putText(frame, "🔴 Eye Distraction", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                session_logger.log_distraction("eye")
                threading.Thread(target=show_challenge).start()
            elif is_distracted_by_head_tilt(face_landmarks, width, height):
                cv2.putText(frame, "🔴 Head Tilt", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                session_logger.log_distraction("head_tilt")
                threading.Thread(target=show_challenge).start()
            else:
                cv2.putText(frame, "🟢 Focused", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Engaged.ai - Attention Tracker", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to quit
        break

cap.release()
session_logger.end_session()
cv2.destroyAllWindows()