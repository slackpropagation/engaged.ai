def is_distracted_by_eye_position(face_landmarks, width):
    """
    Determines if the user is distracted based on iris position within the eye frame.
    Tracks both horizontal and vertical gaze.
    """
    def is_gaze_off(iris, inner, outer, top, bottom):
        iris_x = iris.x
        iris_y = iris.y

        eye_left = min(inner.x, outer.x)
        eye_right = max(inner.x, outer.x)
        eye_top = min(top.y, bottom.y)
        eye_bottom = max(top.y, bottom.y)

        x_ratio = (iris_x - eye_left) / (eye_right - eye_left + 1e-6)
        y_ratio = (iris_y - eye_top) / (eye_bottom - eye_top + 1e-6)

        return (
            x_ratio < 0.3 or x_ratio > 0.7 or
            y_ratio < 0.3 or y_ratio > 0.7
        )

    left_iris = face_landmarks.landmark[468]
    right_iris = face_landmarks.landmark[473]

    left_eye_inner = face_landmarks.landmark[133]
    left_eye_outer = face_landmarks.landmark[33]
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]

    right_eye_inner = face_landmarks.landmark[362]
    right_eye_outer = face_landmarks.landmark[263]
    right_eye_top = face_landmarks.landmark[386]
    right_eye_bottom = face_landmarks.landmark[374]

    return (
        is_gaze_off(left_iris, left_eye_inner, left_eye_outer, left_eye_top, left_eye_bottom) or
        is_gaze_off(right_iris, right_eye_inner, right_eye_outer, right_eye_top, right_eye_bottom)
    )