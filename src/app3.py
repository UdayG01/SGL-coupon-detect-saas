import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
# from gpiozero import LED
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import requests
import csv
from datetime import datetime
from pathlib import Path

class SKUInputWindow:
    """Tkinter window for SKU code input"""
    def _init_(self):
        self.sku_code = None
        self.window = tk.Tk()
        self.window.title("SKU Code Input")
        self.window.geometry("400x200")
        
        # Center the window
        self.window.eval('tk::PlaceWindow . center')
        
        # Create and configure widgets
        frame = ttk.Frame(self.window, padding="20")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(frame, text="Enter SKU Code:", font=('Arial', 12)).grid(row=0, column=0, pady=10)
        
        self.sku_entry = ttk.Entry(frame, font=('Arial', 12), width=30)
        self.sku_entry.grid(row=1, column=0, pady=10)
        self.sku_entry.focus()
        
        # Bind Enter key to submit
        self.sku_entry.bind('<Return>', lambda e: self.submit())
        
        ttk.Button(frame, text="Start Detection", command=self.submit).grid(row=2, column=0, pady=20)
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def submit(self):
        sku = self.sku_entry.get().strip()
        if sku:
            self.sku_code = sku
            self.window.quit()
            self.window.destroy()
        else:
            messagebox.showwarning("Invalid Input", "Please enter a valid SKU code")
    
    def on_closing(self):
        self.window.quit()
        self.window.destroy()
    
    def run(self):
        self.window.mainloop()
        return self.sku_code


class user_app_callback_class(app_callback_class):
    def _init_(self, sku_code):
        super()._init_()
        
        # Configuration
        self.sku_code = sku_code
        self.target_object = "coupon"
        self.containing_object = "bucket"
        
        # Detection line configuration (vertical line at center)
        self.detection_line_x = 0.5  # Normalized x-coordinate (center of frame)
        self.line_tolerance = 0.05   # Tolerance for crossing detection
        
        # Tracking variables for each bucket
        self.bucket_tracking = {}  # {bucket_id: {'has_crossed': bool, 'has_coupon': bool, 'coupon_crossed': bool}}
        
        # Debouncing variables
        self.bucket_cross_frames = {}  # {bucket_id: frame_count}
        self.coupon_in_bucket_frames = {}  # {bucket_id: frame_count}
        
        # Thresholds
        self.cross_threshold = 3  # Frames needed to confirm crossing
        self.coupon_threshold = 3  # Frames needed to confirm coupon presence
        
        # State tracking
        self.processed_buckets = set()  # Track already processed buckets
        
        # LED control
        # self.green_led = LED(18)
        # self.red_led = LED(14)
        # self.red_led.off()
        # self.green_led.on()
        
        # CSV file setup
        self.csv_filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_csv()
        
        # API endpoint
        self.api_url = "https://localhost:1880/result"
    
    def init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['SKU_CODE', 'TIMESTAMP', 'RESULT'])
    
    def log_result(self, result):
        """Log detection result to CSV"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.sku_code, timestamp, result])
        print(f"Logged: SKU={self.sku_code}, Time={timestamp}, Result={result}")
    
    def send_api_request(self, result_bool):
        """Send POST request to API"""
        try:
            payload = {"result": str(result_bool).lower()}
            response = requests.post(self.api_url, json=payload, timeout=2, verify=False)
            print(f"API Response: {response.status_code}")
        except Exception as e:
            print(f"API Request failed: {e}")
    
    def is_bbox_inside(self, inner_bbox, outer_bbox):
        """Check if inner bounding box is inside outer bounding box"""
        inner_xmin = inner_bbox.xmin()
        inner_ymin = inner_bbox.ymin()
        inner_xmax = inner_xmin + inner_bbox.width()
        inner_ymax = inner_ymin + inner_bbox.height()
        
        outer_xmin = outer_bbox.xmin()
        outer_ymin = outer_bbox.ymin()
        outer_xmax = outer_xmin + outer_bbox.width()
        outer_ymax = outer_ymin + outer_bbox.height()
        
        # Check if inner box is contained within outer box
        return (inner_xmin >= outer_xmin and 
                inner_xmax <= outer_xmax and 
                inner_ymin >= outer_ymin and 
                inner_ymax <= outer_ymax)
    
    def has_crossed_line(self, bbox):
        """Check if bounding box center has crossed the detection line"""
        center_x = bbox.xmin() + (bbox.width() / 2)
        return abs(center_x - self.detection_line_x) < self.line_tolerance
    
    def process_bucket_result(self, bucket_id, has_coupon):
        """Process final result for a bucket"""
        result = "OK" if has_coupon else "NG"
        
        # Update LEDs
        if has_coupon:
            # self.green_led.on()
            # self.red_led.off()
            pass
        else:
            # self.green_led.off()
            # self.red_led.on()
            pass
        
        # Log to CSV
        self.log_result(result)
        
        # Send API request
        self.send_api_request(has_coupon)
        
        # Mark as processed
        self.processed_buckets.add(bucket_id)
        
        print(f"\n{'='*50}")
        print(f"BUCKET {bucket_id} PROCESSED: {result}")
        print(f"Coupon detected: {has_coupon}")
        print(f"{'='*50}\n")


def app_callback(pad, info, user_data):
    """Main callback function for processing video frames"""
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data.increment()
    
    # Get frame information
    format, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
    
    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    
    # Separate buckets and coupons
    buckets = []
    coupons = []
    
    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        
        if confidence > 0.5:
            if label == user_data.containing_object:
                buckets.append(detection)
            elif label == user_data.target_object:
                coupons.append(detection)
    
    # Process each bucket
    for bucket in buckets:
        bucket_bbox = bucket.get_bbox()
        
        # Get or create bucket ID (using position as simple ID)
        bucket_center_x = bucket_bbox.xmin() + (bucket_bbox.width() / 2)
        bucket_id = int(bucket_center_x * 1000)  # Simple ID based on position
        
        # Skip if already processed
        if bucket_id in user_data.processed_buckets:
            continue
        
        # Initialize tracking for this bucket
        if bucket_id not in user_data.bucket_tracking:
            user_data.bucket_tracking[bucket_id] = {
                'has_crossed': False,
                'has_coupon': False,
                'coupon_crossed': False
            }
            user_data.bucket_cross_frames[bucket_id] = 0
            user_data.coupon_in_bucket_frames[bucket_id] = 0
        
        # Check if bucket is crossing the line
        if user_data.has_crossed_line(bucket_bbox):
            user_data.bucket_cross_frames[bucket_id] += 1
            
            # Check for coupons inside this bucket
            coupon_found = False
            for coupon in coupons:
                coupon_bbox = coupon.get_bbox()
                if user_data.is_bbox_inside(coupon_bbox, bucket_bbox):
                    coupon_found = True
                    user_data.coupon_in_bucket_frames[bucket_id] += 1
                    
                    # Check if coupon is also crossing the line
                    if user_data.has_crossed_line(coupon_bbox):
                        if user_data.coupon_in_bucket_frames[bucket_id] >= user_data.coupon_threshold:
                            user_data.bucket_tracking[bucket_id]['coupon_crossed'] = True
                    break
            
            if not coupon_found:
                user_data.coupon_in_bucket_frames[bucket_id] = 0
            
            # Confirm bucket crossing with debouncing
            if user_data.bucket_cross_frames[bucket_id] >= user_data.cross_threshold:
                if not user_data.bucket_tracking[bucket_id]['has_crossed']:
                    user_data.bucket_tracking[bucket_id]['has_crossed'] = True
                    has_coupon = user_data.bucket_tracking[bucket_id]['coupon_crossed']
                    user_data.process_bucket_result(bucket_id, has_coupon)
        else:
            # Reset if not near line
            if user_data.bucket_cross_frames[bucket_id] > 0:
                user_data.bucket_cross_frames[bucket_id] = max(0, user_data.bucket_cross_frames[bucket_id] - 1)
    
    # Draw detection line on frame (optional visualization)
    if user_data.use_frame and frame is not None:
        line_x = int(user_data.detection_line_x * width)
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 2)
        cv2.putText(frame, "DETECTION LINE", (line_x + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw detection count
        cv2.putText(frame, f"Buckets: {len(buckets)} | Coupons: {len(coupons)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)
    
    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # Get SKU code from user via Tkinter window
    print("Starting SKU Input Window...")
    input_window = SKUInputWindow()
    sku_code = input_window.run()
    
    if not sku_code:
        print("No SKU code entered. Exiting...")
        exit(0)
    
    print(f"SKU Code entered: {sku_code}")
    print("Starting detection pipeline...")
    
    # Create callback instance with SKU code
    user_data = user_app_callback_class(sku_code)
    
    # Start GStreamer detection app
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()