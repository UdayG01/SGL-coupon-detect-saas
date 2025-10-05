import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict
import math

class BucketDetectionSystem:
    def __init__(self, model_path, video_path, output_path=None):
        """
        Initialize the bucket detection system
        
        Args:
            model_path: Path to your trained YOLOv8 model
            video_path: Path to input video file
            output_path: Optional path to save output video
        """
        # Load the trained YOLOv8 model
        self.model = YOLO(model_path)
        
        # Video setup
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Virtual line for counting (vertical line at center)
        self.counting_line_x = self.width // 2
        
        # Tracking variables
        self.bucket_tracker = {}  # Track bucket positions for line crossing
        self.total_buckets_passed = 0
        self.ok_buckets_count = 0
        self.track_id_counter = 0
        
        # Class names from your data.yaml
        self.class_names = {0: 'bucket', 1: 'coupon'}  # coupon = contact card
        
        # Video writer setup
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None
            
        print(f"Initialized detection system:")
        print(f"Video dimensions: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Counting line at x={self.counting_line_x}")

    def calculate_distance(self, center1, center2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def assign_track_id(self, bucket_center, frame_buckets):
        """Simple tracking: assign track ID based on proximity to previous positions"""
        min_distance = float('inf')
        assigned_id = None
        
        for track_id, prev_data in self.bucket_tracker.items():
            if prev_data['active']:
                distance = self.calculate_distance(bucket_center, prev_data['center'])
                if distance < min_distance and distance < 100:  # threshold for same bucket
                    min_distance = distance
                    assigned_id = track_id
        
        if assigned_id is None:
            # New bucket detected
            assigned_id = self.track_id_counter
            self.track_id_counter += 1
            
        return assigned_id

    def check_line_crossing(self, track_id, current_center, has_contact_card):
        """Check if bucket has crossed the counting line"""
        if track_id in self.bucket_tracker:
            prev_center = self.bucket_tracker[track_id]['center']
            
            # Check if bucket crossed the line (from left to right)
            if (prev_center[0] < self.counting_line_x and 
                current_center[0] >= self.counting_line_x):
                
                # Bucket crossed the line
                if not self.bucket_tracker[track_id]['counted']:
                    self.total_buckets_passed += 1
                    self.bucket_tracker[track_id]['counted'] = True
                    
                    if has_contact_card:
                        self.ok_buckets_count += 1
                        print(f"✅ OK Bucket #{track_id} passed (contains contact card)")
                    else:
                        print(f"🚨 ALERT: Empty Bucket #{track_id} passed (no contact card)")
                    
                    return True
        return False

    def process_detections(self, frame, results):
        """Process YOLO detections and implement logic"""
        # Get detections
        boxes = results[0].boxes
        
        if boxes is None:
            return frame
            
        # Separate buckets and coupons (contact cards)
        bucket_detections = []
        coupon_detections = []
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if confidence > 0.25:  # Confidence threshold
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                if class_id == 0:  # bucket
                    bucket_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'confidence': confidence
                    })
                elif class_id == 1:  # coupon (contact card)
                    coupon_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'confidence': confidence
                    })
        
        # Mark all existing buckets as inactive for this frame
        for track_id in self.bucket_tracker:
            self.bucket_tracker[track_id]['active'] = False
        
        # Process each bucket detection
        for bucket in bucket_detections:
            x1, y1, x2, y2 = bucket['bbox']
            bucket_center = bucket['center']
            
            # Check if this bucket contains a contact card
            has_contact_card = False
            for coupon in coupon_detections:
                cx1, cy1, cx2, cy2 = coupon['bbox']
                coupon_center = coupon['center']
                
                # Check if coupon center is inside bucket bbox (with some tolerance)
                if (x1 - 20 <= coupon_center[0] <= x2 + 20 and 
                    y1 - 20 <= coupon_center[1] <= y2 + 20):
                    has_contact_card = True
                    break
            
            # Assign track ID
            track_id = self.assign_track_id(bucket_center, bucket_detections)
            
            # Update tracker
            self.bucket_tracker[track_id] = {
                'center': bucket_center,
                'bbox': bucket['bbox'],
                'has_contact_card': has_contact_card,
                'active': True,
                'counted': self.bucket_tracker.get(track_id, {}).get('counted', False)
            }
            
            # Check for line crossing
            line_crossed = self.check_line_crossing(track_id, bucket_center, has_contact_card)
            
            # Draw bounding box (green if has contact card, red if empty)
            color = (0, 255, 0) if has_contact_card else (0, 0, 255)  # Green or Red
            thickness = 3 if line_crossed else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"Bucket #{track_id}"
            if has_contact_card:
                label += " (OK)"
            else:
                label += " (EMPTY)"
                
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw contact cards with blue boxes
        for coupon in coupon_detections:
            x1, y1, x2, y2 = coupon['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
            cv2.putText(frame, "Contact Card", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame

    def draw_ui_elements(self, frame):
        """Draw counting line and statistics on frame"""
        # Draw vertical counting line (dotted)
        line_color = (255, 255, 0)  # Yellow
        dot_spacing = 20
        
        for y in range(0, self.height, dot_spacing):
            cv2.circle(frame, (self.counting_line_x, y), 3, line_color, -1)
        
        # Draw statistics panel
        panel_height = 120
        panel_color = (0, 0, 0)  # Black background
        cv2.rectangle(frame, (10, 10), (400, panel_height), panel_color, -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # Statistics text
        stats = [
            f"Total Buckets Passed: {self.total_buckets_passed}",
            f"OK Buckets (with card): {self.ok_buckets_count}",
            f"Empty Buckets: {self.total_buckets_passed - self.ok_buckets_count}",
            f"Active Tracks: {sum(1 for t in self.bucket_tracker.values() if t.get('active', False))}"
        ]
        
        for i, stat in enumerate(stats):
            y_pos = 30 + i * 25
            cv2.putText(frame, stat, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Legend
        legend_y = self.height - 100
        cv2.rectangle(frame, (10, legend_y), (300, self.height - 10), panel_color, -1)
        cv2.rectangle(frame, (10, legend_y), (300, self.height - 10), (255, 255, 255), 2)
        
        legend_items = [
            ("Green Box: Bucket with contact card", (0, 255, 0)),
            ("Red Box: Empty bucket (ALERT)", (0, 0, 255)),
            ("Blue Box: Contact card", (255, 0, 0))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + 20 + i * 20
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run(self):
        """Main processing loop"""
        print("Starting bucket detection system...")
        print("Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video or failed to read frame")
                    break
                
                # Run YOLO inference
                results = self.model(frame, verbose=False)
                
                # Process detections and apply logic
                processed_frame = self.process_detections(frame, results)
                
                # Draw UI elements
                self.draw_ui_elements(processed_frame)
                
                # Save frame if output path specified
                if self.out:
                    self.out.write(processed_frame)
                
                # Display frame
                cv2.imshow('Bucket Detection System', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\nFinal Statistics:")
        print(f"Total Buckets Passed: {self.total_buckets_passed}")
        print(f"OK Buckets: {self.ok_buckets_count}")
        print(f"Empty Buckets (Alerts): {self.total_buckets_passed - self.ok_buckets_count}")
        
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

def main():
    # Configuration
    MODEL_PATH = "model/best.pt"   
    VIDEO_PATH = "assets/input video/test_video1.mp4"    
    OUTPUT_PATH = "assets/output videos/output_video.mp4"               

    # Create and run the detection system
    try:
        detector = BucketDetectionSystem(MODEL_PATH, VIDEO_PATH, OUTPUT_PATH)
        detector.run()
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


# Alternative usage example for different scenarios:

def run_detection_with_webcam():
    """Run detection using webcam instead of video file"""
    MODEL_PATH = "model/best.pt"
    detector = BucketDetectionSystem(MODEL_PATH, 0)  # 0 for webcam
    detector.run()

def run_detection_batch():
    """Process multiple videos"""
    MODEL_PATH = "model/best.pt"
    video_files = ["assets/input video/test_video1.mp4"]

    for i, video_path in enumerate(video_files):
        print(f"\nProcessing video {i+1}/{len(video_files)}: {video_path}")
        output_path = f"assets/output videos/output_{i+1}.mp4"
        
        detector = BucketDetectionSystem(MODEL_PATH, video_path, output_path)
        detector.run()

# Additional utility functions

def analyze_detection_results(model_path, video_path):
    """Analyze detection performance without visualization"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    total_frames = 0
    bucket_detections = 0
    coupon_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        total_frames += 1
        results = model(frame, verbose=False)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if confidence > 0.5:
                    if class_id == 0:  # bucket
                        bucket_detections += 1
                    elif class_id == 1:  # coupon
                        coupon_detections += 1
    
    cap.release()
    
    print(f"Analysis Results:")
    print(f"Total frames: {total_frames}")
    print(f"Bucket detections: {bucket_detections}")
    print(f"Contact card detections: {coupon_detections}")
    print(f"Average buckets per frame: {bucket_detections/total_frames:.2f}")
    print(f"Average contact cards per frame: {coupon_detections/total_frames:.2f}")

# Usage instructions:
"""
SETUP INSTRUCTIONS:

1. Install required packages:
   pip install ultralytics opencv-python numpy

2. Update the paths in main():
   - MODEL_PATH: Point to your trained YOLOv8 model file (.pt)
   - VIDEO_PATH: Point to your input video file
   - OUTPUT_PATH: Where to save processed video (optional)

3. Run the script:
   python bucket_detection.py

FEATURES:
- Detects buckets and contact cards using your trained YOLOv8 model
- Green bounding boxes for buckets containing contact cards
- Red bounding boxes for empty buckets (triggers alerts)
- Blue bounding boxes for contact cards
- Virtual dotted line for counting buckets that pass through
- Real-time statistics display
- Console alerts for empty buckets
- Optional video output saving

CONTROLS:
- Press 'q' to quit
- Press 'p' to pause/resume

CUSTOMIZATION:
- Adjust confidence threshold in process_detections() (currently 0.5)
- Modify distance threshold for tracking in assign_track_id()
- Change colors by modifying the color tuples (BGR format)
- Adjust UI panel sizes and positions
"""