import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import hailo
import cv2
import numpy as np
import requests
import csv
import os
from datetime import datetime
import threading 
import time
import schedule

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

import tkinter as tk
from tkinter import ttk, messagebox

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

#from ui import CouponDetectorApp
from ui2 import CouponDetectorApp
import subprocess



class SKUInputWindow:
    """Tkinter window for SKU code input"""
    def __init__(self):
        self.sku_code = None
        self.window = tk.Tk()
        self.window.title("SKU Code Input")
        self.window.geometry("400x200")

        # Center window
        self.window.eval('tk::PlaceWindow . center')

        frame = ttk.Frame(self.window, padding="20")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(frame, text="Enter SKU Code:", font=('Arial', 12)).grid(row=0, column=0, pady=10)

        self.sku_entry = ttk.Entry(frame, font=('Arial', 12), width=30)
        self.sku_entry.grid(row=1, column=0, pady=10)
        self.sku_entry.focus()

        # Bind Enter key
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
    def __init__(self, sku_code):
        super().__init__()
        self.sku_code = sku_code

        # Detection line config
        self.detection_line_x = 0.5   # center of frame
        self.line_tolerance = 0.05

        # Debounce thresholds
        self.cross_threshold = 3

        # Counters
        self.bucket_cross_frames = 0
        self.coupon_cross_frames = 0

        # CSV log
        self.csv_filename = f"/home/admin/Desktop/Detection_Logs/detection_log_{datetime.now().strftime('%d_%m_%Y')}.csv"
        self.init_csv()

        # API
        self.api_url = "http://localhost:1880/result"
        
        # Email
        self.sender_email = "uday.gupta@renataiot.com"
        self.receiver_email = "dispatch@siddharthpetro.com"
        self.app_password = "zrvy bygn nsif xbvg"
        
        # Cooldown
        self.cooldown_frames = 0
        self.cooldown_limit = 30

    def init_csv(self):
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['SKU_CODE', 'TIMESTAMP', 'RESULT'])

    def log_result(self, result):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.sku_code, ts, result])
        print(f"Logged: {result} at {ts}")

    def send_api_request(self, result_bool):
        try:
            payload = {"result": str(result_bool).lower()}
            r = requests.post(self.api_url, json=payload, timeout=2)
            print(f"API sent: {payload}, status={r.status_code}")
        except Exception as e:
            print(f"API error: {e}")

    def has_crossed_line(self, bbox):
        center_x = bbox.xmin() + (bbox.width() / 2)
        return abs(center_x - self.detection_line_x) < self.line_tolerance

    def reset_counters(self):
        """Reset state after one bucket processed"""
        self.bucket_cross_frames = 0
        self.coupon_cross_frames = 0
        self.cooldown_frames = self.cooldown_limit
    
    def send_email_text(self, timestamp, result, cc_emails=["uday.gupta@renataiot.com"]):
        """Send a test email with optional CC."""

        # Ensure cc_emails is a list
        if cc_emails is None:
            cc_emails = []
        
        msg = MIMEText(f"SKU Code: {self.sku_code}\nTimestamp: {timestamp}\nResult: {result}")
        msg["Subject"] = "NG Detection Result"
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email
        if cc_emails:
            msg["Cc"] = ", ".join(cc_emails)
        
        # Collect all recipients (To + CC)
        recipients = [self.receiver_email] + cc_emails

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())
            print("✅ Email sent successfully")
        except Exception as e:
            print("❌ Error sending email:", e)

    def send_email_daily_log(self):
        """Send an email with CSV attachment."""
        msg = MIMEMultipart()
        msg["Subject"] = f"RenataAI Coupon Detection Report-{datetime.now().strftime('%Y%m%d')}"
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email

        # Email body
        body = """
        PFA today's coupon detection report.
        """
        msg.attach(MIMEText(body, "plain"))

        # Attach CSV if it exists
        if os.path.exists(self.csv_filename):
            with open(self.csv_filename, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(self.csv_filename)}")
            msg.attach(part)
        else:
            print(f"⚠️ CSV file '{self.csv_filename}' not found, sending email without attachment.")

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.app_password)
                server.send_message(msg)
            print("✅ Email with CSV sent successfully")
        except Exception as e:
            print("❌ Error sending email:", e)


    def process_frame(self, buckets, coupons):
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return
            
        bucket_crossing = any(self.has_crossed_line(b.get_bbox()) for b in buckets)
        coupon_crossing = any(self.has_crossed_line(c.get_bbox()) for c in coupons)

        # Update counters
        if bucket_crossing:
            self.bucket_cross_frames += 1
        else:
            self.bucket_cross_frames = max(0, self.bucket_cross_frames - 1)

        if coupon_crossing:
            self.coupon_cross_frames += 1
        else:
            self.coupon_cross_frames = max(0, self.coupon_cross_frames - 1)

        # Debounced decision
        if self.coupon_cross_frames >= self.cross_threshold:
            # OK
            self.log_result("OK")
            self.send_api_request(True)
            self.reset_counters()

        elif self.bucket_cross_frames >= self.cross_threshold and self.coupon_cross_frames == 0:
            # NG
            self.log_result("NG")
            self.send_api_request(False)
            self.reset_counters()
            #self.send_email_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "NG")
    
    

def get_zoom_for_sku(sku_code):
        """Return zoom value based on SKU weight"""
        sku_code_lower = sku_code.lower()
        if "1kg" in sku_code_lower or "0.5kg" in sku_code_lower:
            return 150
        else:
            return 100

def set_camera_zoom(zoom_value):
    """Set the zoom of the camera using v4l2-ctl"""
    try:
        cmd = ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl", f"zoom_absolute={zoom_value}"]
        subprocess.run(cmd, check=True)
        print(f"✅ Zoom set to {zoom_value}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to set zoom: {e}")
        

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    # Get detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    buckets, coupons = [], []
    for det in detections:
        if det.get_confidence() > 0.5:
            if det.get_label() == "bucket":
                buckets.append(det)
            elif det.get_label() == "coupon":
                coupons.append(det)

    user_data.process_frame(buckets, coupons)

    # Visualization
    fmt, w, h = get_caps_from_pad(pad)
    if user_data.use_frame and fmt is not None:
        frame = get_numpy_from_buffer(buffer, fmt, w, h)
        line_x = int(user_data.detection_line_x * w)
        cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK


#def schedule_email(user_data):
    # Schedule the function every day at 17:30 (5:30 PM)
#    schedule.every().day.at("17:30").do(user_data.send_email_daily_log)

 #   while True:
  #      schedule.run_pending()
   #     time.sleep(120)

if __name__ == "__main__":
    # Tkinter input
    print("Starting SKU Input Window...")
    input_window = CouponDetectorApp()
    sku_code = input_window.run()

    if not sku_code:
        print("No SKU code entered. Exiting...")
        exit(0)

    
    # Set camera zoom dynamically based on SKU weight
    zoom_value = get_zoom_for_sku(sku_code)
    set_camera_zoom(zoom_value)

    print(f"SKU Code entered: {sku_code}")
    print(f"📷 Camera zoom adjusted to {zoom_value} for {sku_code}")
    print("Starting detection pipeline...")

    user_data = user_app_callback_class(sku_code)

    # Start scheduler in a background thread
    #threading.Thread(target=schedule_email, args=(user_data,), daemon=True).start()

    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
