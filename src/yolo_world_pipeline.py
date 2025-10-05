"""
Zero-Shot Bucket and Contact Card Detection using YOLO-World
No training required - just provide text prompts!
"""

import cv2
import numpy as np
from ultralytics import YOLOWorld
import time
from collections import defaultdict
import math

class ZeroShotBucketDetection:
    def __init__(self, video_path, output_path=None):
        """
        Initialize zero-shot detection system using YOLO-World
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video
        """
        # Load YOLO-World model (no training required!)
        self.model = YOLOWorld("yolov8s-world.pt")
        
        # Define classes using natural language - optimized for your specific objects
        # self.model.set_classes([
        #     "paint bucket", 
        #     "bucket", 
        #     "white bucket",
        #     "plastic bucket",
        #     "color card", 
        #     "color sample",
        #     "paint color card",
        #     "colored label",
        #     "product label",
        #     "sample card",
        #     "color chart",
        #     "color swatch"
        # ])

        self.model.set_classes([
            "plastic paint bucket", 
            "paper", 
            "sheet", 
            "color swatch card", 
            "paper card on paint",
            "rectangular paper on brown paint",
            "rectangular paper card on paint"
            ])

        
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
        self.bucket_tracker = {}
        self.total_buckets_passed = 0
        self.ok_buckets_count = 0
        self.track_id_counter = 0
        
        # Video writer setup
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None
            
        print(f"🚀 Zero-Shot Detection System Initialized!")
        print(f"📏 Video: {self.width}x{self.height} @ {self.fps}fps")
        print(f"🎯 Counting line at x={self.counting_line_x}")
        print(f"🔍 Detecting: Buckets and Contact Cards (no training needed!)")

    def calculate_distance(self, center1, center2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def assign_track_id(self, bucket_center):
        """Simple tracking based on proximity"""
        min_distance = float('inf')
        assigned_id = None
        
        for track_id, prev_data in self.bucket_tracker.items():
            if prev_data.get('active', False):
                distance = self.calculate_distance(bucket_center, prev_data['center'])
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    assigned_id = track_id
        
        if assigned_id is None:
            assigned_id = self.track_id_counter
            self.track_id_counter += 1
            
        return assigned_id

    def check_line_crossing(self, track_id, current_center, has_contact_card):
        """Check if bucket crossed the counting line"""
        if track_id in self.bucket_tracker:
            prev_center = self.bucket_tracker[track_id]['center']
            
            # Check crossing from left to right
            if (prev_center[0] < self.counting_line_x and 
                current_center[0] >= self.counting_line_x):
                
                if not self.bucket_tracker[track_id].get('counted', False):
                    self.total_buckets_passed += 1
                    self.bucket_tracker[track_id]['counted'] = True
                    
                    if has_contact_card:
                        self.ok_buckets_count += 1
                        print(f"✅ OK Bucket #{track_id} passed (contains color card)")
                    else:
                        print(f"🚨 ALERT: Empty Bucket #{track_id} passed (no color card)")
                    
                    return True
        return False

    def is_bucket_class(self, class_name):
        """Check if detected class is a bucket-type object"""
        bucket_keywords = ['bucket', 'container']
        return any(keyword in class_name.lower() for keyword in bucket_keywords)
    
    def is_card_class(self, class_name):
        """Check if detected class is a card-type object"""
        card_keywords = ['card', 'label', 'sample', 'swatch', 'chart', 'color']
        return any(keyword in class_name.lower() for keyword in card_keywords)

    def process_detections(self, frame, results):
        """Process YOLO-World detections"""
        bucket_detections = []
        card_detections = []
        
        # Parse detections
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get class name from YOLO-World
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                if confidence > 0.10:  # Lower threshold for small color cards
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'confidence': confidence,
                        'class_name': class_name,
                        'area': area
                    }
                    
                    if self.is_bucket_class(class_name):
                        bucket_detections.append(detection)
                    elif self.is_card_class(class_name):
                        # Only consider small objects as cards (filter out large false positives)
                        if area < (self.width * self.height) * 0.1:  # Less than 10% of image
                            card_detections.append(detection)
        
        # Mark existing buckets as inactive
        for track_id in self.bucket_tracker:
            self.bucket_tracker[track_id]['active'] = False
        
        # Process bucket detections
        for bucket in bucket_detections:
            x1, y1, x2, y2 = bucket['bbox']
            bucket_center = bucket['center']
            
            # Check if bucket contains a color card (improved logic)
            has_contact_card = False
            detected_cards_in_bucket = []
            
            for card in card_detections:
                cx1, cy1, cx2, cy2 = card['bbox']
                card_center = card['center']
                
                # More sophisticated containment check
                # Check if card center is within bucket bounds
                card_in_bucket_x = x1 <= card_center[0] <= x2
                card_in_bucket_y = y1 <= card_center[1] <= y2
                
                # Also check if card overlaps significantly with bucket
                overlap_x = max(0, min(x2, cx2) - max(x1, cx1))
                overlap_y = max(0, min(y2, cy2) - max(y1, cy1))
                overlap_area = overlap_x * overlap_y
                card_area = (cx2 - cx1) * (cy2 - cy1)
                overlap_ratio = overlap_area / card_area if card_area > 0 else 0
                
                if (card_in_bucket_x and card_in_bucket_y) or overlap_ratio > 0.5:
                    has_contact_card = True
                    detected_cards_in_bucket.append(card)
                    break
            
            # Assign track ID and update tracker
            track_id = self.assign_track_id(bucket_center)
            self.bucket_tracker[track_id] = {
                'center': bucket_center,
                'bbox': bucket['bbox'],
                'has_contact_card': has_contact_card,
                'active': True,
                'counted': self.bucket_tracker.get(track_id, {}).get('counted', False),
                'class_name': bucket['class_name']
            }
            
            # Check for line crossing
            line_crossed = self.check_line_crossing(track_id, bucket_center, has_contact_card)
            
            # Draw bounding box - GREEN if has card, RED if empty
            color = (0, 255, 0) if has_contact_card else (0, 0, 255)  # Green/Red
            thickness = 4 if line_crossed else 3
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with detection info
            status = "✓ HAS CARD" if has_contact_card else "✗ EMPTY"
            label = f"Bucket #{track_id} ({status})"
            conf_text = f"{bucket['class_name']} - Conf: {bucket['confidence']:.2f}"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            max_width = max(label_size[0], conf_size[0])
            
            cv2.rectangle(frame, (x1, y1 - 35), 
                         (x1 + max_width + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 5, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw color cards with enhanced visibility
        for card in card_detections:
            x1, y1, x2, y2 = card['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue border
            
            # Add a small colored fill to make cards more visible
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), -1)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            
            label = f"{card['class_name']} ({card['confidence']:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + label_size[0], y1), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

    def draw_ui_elements(self, frame):
        """Draw UI elements and statistics"""
        # Counting line
        for y in range(0, self.height, 20):
            cv2.circle(frame, (self.counting_line_x, y), 3, (255, 255, 0), -1)
        
        # Statistics panel
        panel_height = 140
        cv2.rectangle(frame, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, panel_height), (255, 255, 255), 2)
        
        stats = [
            "🎨 PAINT BUCKET + COLOR CARD DETECTION",
            f"📊 Total Buckets Passed: {self.total_buckets_passed}",
            f"✅ Buckets with Color Cards: {self.ok_buckets_count}",
            f"❌ Empty Buckets: {self.total_buckets_passed - self.ok_buckets_count}",
            f"🎯 Active Tracks: {sum(1 for t in self.bucket_tracker.values() if t.get('active', False))}"
        ]
        
        for i, stat in enumerate(stats):
            y_pos = 30 + i * 22
            cv2.putText(frame, stat, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Legend
        legend_y = self.height - 120
        cv2.rectangle(frame, (10, legend_y), (400, self.height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, legend_y), (400, self.height - 10), (255, 255, 255), 2)
        
        legend_items = [
            ("🟢 Green: Bucket with color card", (0, 255, 0)),
            ("🔴 Red: Empty bucket (ALERT)", (0, 0, 255)),
            ("🔵 Blue: Color card/sample detected", (255, 0, 0)),
            ("💡 Tip: Lower confidence = 0.25 for small cards", (255, 255, 255))
        ]
        
        for i, (text, color) in enumerate(legend_items):
            y_pos = legend_y + 20 + i * 20
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def run(self):
        """Main processing loop"""
        print("🚀 Starting zero-shot bucket detection...")
        print("💡 No training data required - using AI text understanding!")
        print("⌨️  Press 'q' to quit, 'p' to pause/resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                # Run zero-shot detection
                results = self.model(frame, verbose=False)
                
                # Process detections
                processed_frame = self.process_detections(frame, results)
                
                # Draw UI
                self.draw_ui_elements(processed_frame)
                
                # Save frame
                if self.out:
                    self.out.write(processed_frame)
                
                # Display
                cv2.imshow('Zero-Shot Bucket Detection - YOLO-World', processed_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources and print final stats"""
        print("\n" + "="*50)
        print("📊 FINAL DETECTION STATISTICS")
        print("="*50)
        print(f"🔢 Total Buckets Passed: {self.total_buckets_passed}")
        print(f"✅ OK Buckets: {self.ok_buckets_count}")
        print(f"❌ Empty Buckets: {self.total_buckets_passed - self.ok_buckets_count}")
        print(f"📈 Success Rate: {(self.ok_buckets_count/max(1,self.total_buckets_passed))*100:.1f}%")
        print("="*50)
        
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

def main():
    """Main function with easy configuration"""
    print("🤖 YOLO-World Zero-Shot Bucket Detection")
    print("=" * 45)
    print("✨ No training required - just run and detect!")
    print("🎯 Automatically detects buckets and contact cards")
    print("🔧 Customize detection by editing the class list")
    print("=" * 45)
    
    # Configuration
    VIDEO_PATH = "assets/input video/test_video1.mp4"  # Change this!
    OUTPUT_PATH = "assets/output videos/zero_shot_output.mp4"         # Optional
    
    try:
        detector = ZeroShotBucketDetection(VIDEO_PATH, OUTPUT_PATH)
        detector.run()
    except FileNotFoundError:
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        print("💡 Please update VIDEO_PATH with your actual video file path")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_with_webcam():
    """Test with webcam instead of video file"""
    print("📹 Testing with webcam...")
    detector = ZeroShotBucketDetection(0)  # 0 = webcam
    detector.run()

def customize_detection_classes():
    """Example of how to customize the detection classes"""
    print("🎨 Customizing detection classes for your specific use case...")
    
    # You can modify the classes in the __init__ method
    # For example, if your objects look different:
    custom_classes = [
        "metal bucket",
        "plastic bucket", 
        "paint container",
        "industrial bucket",
        "contact card",
        "business card",
        "label sticker",
        "white card",
        "paper label"
    ]
    print("💡 Custom classes example:", custom_classes)
    print("🔧 Edit the model.set_classes() call in __init__ to use these")

if __name__ == "__main__":
    main()
    
    # Uncomment to test other features:
    # test_with_webcam()
    # customize_detection_classes()

"""
🚀 ZERO-SHOT DETECTION ADVANTAGES:

✅ NO TRAINING REQUIRED - Just run immediately!
✅ NO DATA ANNOTATION - No need to label thousands of images
✅ CUSTOMIZABLE - Change detection classes by editing text
✅ FAST SETUP - Install and run in minutes
✅ FLEXIBLE - Works with any similar objects

📝 SETUP INSTRUCTIONS:

1. Install required packages:
   pip install ultralytics

2. Update VIDEO_PATH in main() function

3. Run the script:
   python zero_shot_detection.py

4. Customize classes if needed in the model.set_classes() call

🎯 DETECTION TUNING TIPS:

- Adjust confidence threshold (currently 0.3) in process_detections()
- Modify class names to match your specific objects
- Add more descriptive terms like "industrial bucket", "metal container"
- Test different combinations to find what works best

🔧 ALTERNATIVE APPROACHES:

If zero-shot doesn't work well enough:
1. Use this as a starting point to auto-label your data
2. Collect 50-100 images and fine-tune
3. Try the container detection models from Roboflow
4. Combine with traditional computer vision techniques

💡 This approach can save you weeks of data collection and annotation!
"""