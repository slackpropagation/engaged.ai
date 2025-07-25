

def is_distracted_by_eye_position(face_landmarks, width):
    """
    Determines if the user is distracted based on the nose position relative to the eyes.

    Args:
        face_landmarks: Landmarks for the detected face from MediaPipe.
        width: Width of the video frame.

    Returns:
        True if the user is likely distracted (looking away), False if focused.
    """
    # Nose tip (landmark 1), left eye (33), right eye (263)
    nose = face_landmarks.landmark[1]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]

    # Convert to pixel coordinates
    nose_x = int(nose.x * width)
    left_eye_x = int(left_eye.x * width)
    right_eye_x = int(right_eye.x * width)

    # Check if nose is roughly centered between the eyes
    center_eye_x = (left_eye_x + right_eye_x) // 2
    return abs(nose_x - center_eye_x) >= 40