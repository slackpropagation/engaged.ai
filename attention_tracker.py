# attention_tracker.py
import cv2
import mediapipe as mp
from distraction_eye import is_distracted_by_eye_position

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

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

    # Estimate attention direction
    distracted = True
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if not is_distracted_by_eye_position(face_landmarks, width):
                distracted = False

    if distracted:
        cv2.putText(frame, "🔴 Possibly Distracted", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "🟢 Focused", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Engaged.ai - Attention Tracker", frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to quit
        break

cap.release()
cv2.destroyAllWindows()