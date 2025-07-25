import time

class IdleDistractionDetector:
    def __init__(self, timeout_seconds=5):
        """
        Initializes the idle detector with a timeout threshold.
        Args:
            timeout_seconds: How many seconds of no face detection before considered idle.
        """
        self.timeout = timeout_seconds
        self.last_active_time = time.time()
        
    def update_activity(self, face_detected: bool):
        """
        Updates the last seen active time if a face is detected.
        Args:
            face_detected: Boolean indicating if face is currently detected.
        """
        if face_detected:
            self.last_active_time = time.time()
            
    def is_idle(self):
        """
        Returns True if the time since last detected face exceeds the threshold.
        """
        return (time.time() - self.last_active_time) > self.timeout