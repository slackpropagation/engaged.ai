import time
import json
import os

class SessionLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = os.path.join(self.log_dir, f"session_{timestamp}.json")
        self.log_data = {
            "start_time": time.time(),
            "distractions": [],
            "engagement_points": 0
        }
        
    def log_distraction(self, distraction_type):
        self.log_data["distractions"].append({
            "time": time.time(),
            "type": distraction_type
        })
        
        
    def add_points(self, points):
        self.log_data["engagement_points"] += points
        
    def end_session(self):
        self.log_data["end_time"] = time.time()
        with open(self.filepath, "w") as f:
            json.dump(self.log_data, f, indent=2)