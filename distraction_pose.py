def is_distracted_by_head_tilt(face_landmarks, width, height):
    """
    Determines if the user's head is significantly tilted, suggesting distraction.

    Args:
        face_landmarks: Landmarks for the detected face from MediaPipe.
        width: Width of the video frame.
        height: Height of the video frame.

    Returns:
        True if the user's head is tilted (distracted), False otherwise.
    """
    left_ear = face_landmarks.landmark[234]
    right_ear = face_landmarks.landmark[454]

    # Convert to pixel coordinates
    left_ear_y = int(left_ear.y * height)
    right_ear_y = int(right_ear.y * height)

    # If the vertical difference between ears is large, user is tilting head
    return abs(left_ear_y - right_ear_y) > 40
