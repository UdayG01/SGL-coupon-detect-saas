import cv2
import numpy as np
import hailo_platform as hpf
import json
import csv
import time
import os
import threading
import tkinter as tk
from tkinter import messagebox
from datetime import datetime




# ======================
# Paths
# ======================
MODEL_PATH = "/home/admin/Coupon_Detection_Siddharth_Grease/hailo-rpi5-examples/yolov8n.hef"
CONFIG_PATH = "/home/admin/Coupon_Detection_Siddharth_Grease/hailo-rpi5-examples/custom_config.json"

# ======================
# Load config
# ======================
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DETECTION_THRESHOLD = config.get("detection_threshold", 0.5)
LABELS = config.get("labels", [])

# ======================
# Utility: Daily CSV filename
# ======================
def get_csv_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return f"detections_{today}.csv"

def init_csv():
    csv_file = get_csv_filename()
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["S.No", "SKU_CODE", "Timestamp", "Result"])
    return csv_file

# ======================
# Post-processing utils
# ======================
def sigmoid(x): return 1 / (1 + np.exp(-x))

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter_area
    return inter_area / (union + 1e-6)

def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def yolov8_postprocess(output, input_shape, orig_shape, labels, conf_thres=0.5, iou_thres=0.5):
    num_classes = len(labels)
    pred = output[0]  # remove batch dimension
    boxes, scores, class_ids = [], [], []
    for row in pred:
        x, y, w, h = row[0:4]
        obj_conf = sigmoid(row[4])
        class_scores = sigmoid(row[5:]) * obj_conf
        class_id = np.argmax(class_scores)
        conf = class_scores[class_id]
        if conf < conf_thres: continue
        x1 = (x - w / 2) / input_shape[1] * orig_shape[1]
        y1 = (y - h / 2) / input_shape[0] * orig_shape[0]
        x2 = (x + w / 2) / input_shape[1] * orig_shape[1]
        y2 = (y + h / 2) / input_shape[0] * orig_shape[0]
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(class_id)
    if len(boxes) == 0: return []
    boxes, scores, class_ids = np.array(boxes), np.array(scores), np.array(class_ids)
    keep = nms(boxes, scores, iou_thres)
    results = []
    for i in keep:
        results.append([class_ids[i], scores[i], int(boxes[i][0]), int(boxes[i][1]),
                        int(boxes[i][2]), int(boxes[i][3])])
    return results

import smtplib
from email.mime.text import MIMEText

def send_email(sku_code, timestamp, result, receiver_email=None, cc_emails=["uday.gupta@renataiot.com"]):
    """Send a test email with optional CC."""

    SENDER_EMAIL = "uday.renataiot@gmail.com"
    APP_PASSWORD = "gwpc fwfi lnrw pphm"
    
    # Ensure cc_emails is a list
    if cc_emails is None:
        cc_emails = []
    
    msg = MIMEText(f"SKU Code: {sku_code}\nTimestamp: {timestamp}\nResult: {result}")
    msg["Subject"] = "NG Detection Result"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    if cc_emails:
        msg["Cc"] = ", ".join(cc_emails)
    
    # Collect all recipients (To + CC)
    recipients = [receiver_email] + cc_emails

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        print("✅ Email sent successfully")
    except Exception as e:
        print("❌ Error sending email:", e)


# ======================
# Detection Thread
# ======================
class DetectionThread(threading.Thread):
    def __init__(self, app_ref):
        super().__init__()
        self.app_ref = app_ref
        self.running = True
        self.paused = False
        self.counter = 0
        self.last_result = "None"
        self.s_no = 0

    def run(self):
        hef = hpf.HEF(MODEL_PATH)
        with hpf.VDevice() as target:
            configure_params = hpf.ConfigureParams.create_from_hef(hef)
            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()
            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_info = hef.get_output_vstream_infos()[0]
            input_shape = input_vstream_info.shape
            output_shape = output_vstream_info.shape
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Cannot open camera")
                return
            with network_group.activate(network_group_params):
                with hpf.InferVStreams(network_group,
                                       hpf.InputVStreamParams.make_from_network_group(
                                           network_group, quantized=False, format_type=hpf.FormatType.FLOAT32),
                                       hpf.OutputVStreamParams.make_from_network_group(
                                           network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
                                       ) as infer_pipeline:
                    while self.running:
                        if self.paused:
                            time.sleep(0.1)
                            continue
                        ret, frame = cap.read()
                        if not ret: break
                        resized = cv2.resize(frame, (input_shape[1], input_shape[0]))
                        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                        input_data = np.expand_dims(resized.astype(np.float32), axis=0)
                        results = infer_pipeline.infer({input_vstream_info.name: input_data})
                        output_data = results[output_vstream_info.name]
                        detections = yolov8_postprocess(output_data, input_shape, frame.shape, LABELS,
                                                        conf_thres=DETECTION_THRESHOLD)
                        height, width, _ = frame.shape
                        center_x = width // 2
                        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 2)
                        bucket_crossed, coupon_inside = False, False
                        for det in detections:
                            cls_id, conf, x1, y1, x2, y2 = det
                            label = LABELS[cls_id]
                            color = (0, 255, 0) if label == "coupon" else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            obj_center_x = (x1 + x2) // 2
                            if label == "bucket" and abs(obj_center_x - center_x) < 10:
                                bucket_crossed = True
                            if label == "coupon" and bucket_crossed:
                                coupon_inside = True
                        result = None
                        if bucket_crossed:
                            result = "OK" if coupon_inside else "NG"
                            if result == "OK": self.counter += 1
                            self.s_no += 1
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            csv_file = init_csv()
                            with open(csv_file, mode="a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([self.s_no, self.app_ref.sku_code.get(), timestamp, result])
                            self.last_result = result
                            if result == "NG":
                                send_email(self.app_ref.sku_code.get(), timestamp, result, receiver_email="add_something_here@gmail.com")
                        panel_x = width - 300
                        cv2.rectangle(frame, (panel_x, 0), (width, 100), (50, 50, 50), -1)
                        color = (0, 255, 0) if self.last_result == "OK" else (0, 0, 255)
                        cv2.putText(frame, f"Result: {self.last_result}", (panel_x + 10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(frame, f"Counter: {self.counter}", (panel_x + 10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.imshow("YOLOv8 Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
            cap.release()
            cv2.destroyAllWindows()

    def stop(self): self.running = False
    def pause(self): self.paused = True
    def resume(self): self.paused = False

# ======================
# Tkinter App
# ======================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Coupon Detection App")
        self.sku_code = tk.StringVar()
        self.thread = None
        tk.Label(root, text="Enter SKU Code:").pack(pady=5)
        tk.Entry(root, textvariable=self.sku_code).pack(pady=5)
        tk.Button(root, text="Start Detection", command=self.start_detection).pack(pady=5)
        tk.Button(root, text="Pause Detection", command=self.pause_detection).pack(pady=5)
        tk.Button(root, text="Resume Detection", command=self.resume_detection).pack(pady=5)
        tk.Button(root, text="Stop Detection", command=self.stop_detection).pack(pady=5)
        self.status_label = tk.Label(root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(pady=5)

    def start_detection(self):
        if not self.sku_code.get():
            messagebox.showwarning("Warning", "Please enter SKU Code before starting.")
            return
        if self.thread and self.thread.is_alive():
            messagebox.showinfo("Info", "Detection already running.")
            return
        self.thread = DetectionThread(self)
        self.thread.start()
        self.status_label.config(text="Status: Running")

    def pause_detection(self):
        if self.thread:
            self.thread.pause()
            self.status_label.config(text="Status: Paused")

    def resume_detection(self):
        if self.thread:
            if not self.sku_code.get():
                messagebox.showwarning("Warning", "Enter SKU Code before resuming.")
                return
            self.thread.resume()
            self.status_label.config(text="Status: Running")

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.thread.join()
            self.status_label.config(text="Status: Stopped")

# ======================
# Run
# ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
