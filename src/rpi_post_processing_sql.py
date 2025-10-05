from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import csv
import sqlite3
from datetime import datetime

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.results_file = "results.csv"

        # --- CSV setup ---
        with open(self.results_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "result"])
            writer.writeheader()

        # --- SQLite setup ---
        self.db_path = "results.db"
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                result TEXT
            )
        """)
        self.conn.commit()

    def log_result(self, result):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {"timestamp": timestamp, "result": result}

        # --- Log to CSV ---
        with open(self.results_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "result"])
            writer.writerow(row)

        # --- Log to SQLite ---
        self.cursor.execute("INSERT INTO detections (timestamp, result) VALUES (?, ?)", 
                            (timestamp, result))
        self.conn.commit()

        print(row)
        return row

# -----------------------------------------------------------------------------------------------
# Callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Center line x-coordinate
    center_x = width // 2
    detected_buckets = []
    detected_coupons = []

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()  # [x_min, y_min, x_max, y_max]
        confidence = detection.get_confidence()

        x_min, y_min, x_max, y_max = bbox
        obj_center_x = int((x_min + x_max) / 2)

        if label == "bucket":
            detected_buckets.append(obj_center_x)
        elif label == "coupon":
            detected_coupons.append(obj_center_x)

    # Check if objects cross the center line
    result = None
    if any(abs(x - center_x) < 10 for x in detected_buckets):  # bucket crosses line
        if any(abs(x - center_x) < 10 for x in detected_coupons):  # coupon also present
            result = "OK"
            user_data.counter += 1
        else:
            result = "NG"

    if result:
        user_data.log_result(result)

    if user_data.use_frame and frame is not None:
        # Draw center blue line
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)

        # Show counter on top-right corner
        cv2.putText(frame, f"Counter: {user_data.counter}", (width - 250, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert to BGR for display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
