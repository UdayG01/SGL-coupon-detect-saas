"""
Simple OWL-ViT Detection System
Focus: Detect buckets and cards, mark green if card inside bucket, red if empty
"""

import cv2
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import time

class SimpleOwlViTDetection:
    def __init__(self, video_path, output_path=None):
        """
        Initialize simple OWL-ViT detection system
        
        Args:
            video_path: Path to input video file or 0 for webcam
            output_path: Optional path to save output video
        """
        print("🦉 Loading OWL-ViT model...")
        
        # Load OWL-ViT model and processor with optimizations
        print("🔧 Optimizing for RTX 3050...")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        
        # Move to GPU if available with memory optimization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Optimize for RTX 3050
        if self.device == "cuda":
            # Enable mixed precision for faster inference
            self.model.half()  # Use FP16 for speed
            torch.backends.cudnn.benchmark = True
            print(f"🚀 Model loaded on GPU with FP16 optimization")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB available")
        else:
            print(f"⚠️  Running on CPU - consider using GPU for better performance")
        
        # Define text queries for detection
        self.text_queries = [
            # Bucket descriptions
            "a white paint bucket",
            "a plastic bucket with brown rim", 
            "a paint container",
            "a white bucket",
            "a round paint bucket",
            
            # Card descriptions  
            "a color sample card",
            "a colorful rectangular card",
            "a paint color card",
            "a small colored label",
            "a color swatch",
            "a product sample card",
            "a rectangular color sticker",
            "a small colorful card"
        ]
        
        # Video setup
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer setup
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None
            
        print(f"✅ Simple Detection System Ready!")
        print(f"📏 Video: {self.width}x{self.height} @ {self.fps}fps")
        print(f"🎯 Detection queries: {len(self.text_queries)} descriptions")

    def detect_objects(self, image_pil, confidence_threshold=0.1):
        """
        Run OWL-ViT detection on PIL image (optimized for RTX 3050)
        """
        # Resize image if too large (optimization for RTX 3050)
        max_size = 800  # Reduce from default to save memory
        original_size = image_pil.size
        if max(image_pil.size) > max_size:
            ratio = max_size / max(image_pil.size)
            new_size = (int(image_pil.size[0] * ratio), int(image_pil.size[1] * ratio))
            image_pil = image_pil.resize(new_size, Image.Resampling.LANCZOS)
            resize_ratio = ratio
        else:
            resize_ratio = 1.0
        
        # Preprocess image and text
        inputs = self.processor(text=self.text_queries, images=image_pil, return_tensors="pt")
        
        # Move to device and convert to FP16 if using GPU
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if hasattr(inputs['pixel_values'], 'half'):
                inputs['pixel_values'] = inputs['pixel_values'].half()
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference with memory optimization
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                outputs = self.model(**inputs)
        
        # Process outputs
        target_sizes = torch.Tensor([image_pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )
        
        # Parse results and scale back to original size
        detections = []
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.cpu().numpy()
            
            # Scale coordinates back to original image size
            if resize_ratio != 1.0:
                x1, x2 = x1 / resize_ratio, x2 / resize_ratio
                y1, y2 = y1 / resize_ratio, y2 / resize_ratio
            
            score = score.cpu().item()
            text_query = self.text_queries[label]
            
            detection = {
                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                'center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                'confidence': score,
                'text_query': text_query,
                'area': (x2 - x1) * (y2 - y1)
            }
            detections.append(detection)
        
        # Clear GPU cache periodically
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return detections

    def is_bucket_detection(self, text_query):
        """Check if detection is a bucket"""
        bucket_keywords = ['bucket', 'container', 'paint']
        return any(keyword in text_query.lower() for keyword in bucket_keywords)
    
    def is_card_detection(self, text_query):
        """Check if detection is a card"""
        card_keywords = ['card', 'sample', 'swatch', 'label', 'sticker', 'color']
        return any(keyword in text_query.lower() for keyword in card_keywords)

    def is_card_inside_bucket(self, card_bbox, bucket_bbox, tolerance=10):
        """
        Check if card is inside bucket with improved logic
        
        Args:
            card_bbox: (x1, y1, x2, y2) of card
            bucket_bbox: (x1, y1, x2, y2) of bucket
            tolerance: Pixel tolerance for containment
        
        Returns:
            bool: True if card is inside bucket
        """
        cx1, cy1, cx2, cy2 = card_bbox
        bx1, by1, bx2, by2 = bucket_bbox
        
        # Card center
        card_center_x = (cx1 + cx2) / 2
        card_center_y = (cy1 + cy2) / 2
        
        # Check if card center is inside bucket (with tolerance)
        center_inside = (bx1 - tolerance <= card_center_x <= bx2 + tolerance and 
                        by1 - tolerance <= card_center_y <= by2 + tolerance)
        
        # Calculate overlap area
        overlap_x1 = max(cx1, bx1)
        overlap_y1 = max(cy1, by1)
        overlap_x2 = min(cx2, bx2)
        overlap_y2 = min(cy2, by2)
        
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            card_area = (cx2 - cx1) * (cy2 - cy1)
            overlap_ratio = overlap_area / card_area if card_area > 0 else 0
        else:
            overlap_ratio = 0
        
        # Card is inside if center is inside OR significant overlap
        return center_inside or overlap_ratio > 0.3

    def process_frame(self, frame):
        """
        Process single frame for bucket and card detection
        """
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        
        # Run detection
        detections = self.detect_objects(image_pil, confidence_threshold=0.1)
        
        # Separate bucket and card detections
        bucket_detections = []
        card_detections = []
        
        for detection in detections:
            if self.is_bucket_detection(detection['text_query']):
                bucket_detections.append(detection)
            elif self.is_card_detection(detection['text_query']):
                # Filter out very large detections (likely false positives)
                if detection['area'] < (self.width * self.height) * 0.2:  # Less than 20% of image
                    card_detections.append(detection)
        
        # Process each bucket and check for cards
        processed_frame = frame.copy()
        
        for bucket in bucket_detections:
            bx1, by1, bx2, by2 = bucket['bbox']
            
            # Check if any card is inside this bucket
            has_card = False
            cards_in_bucket = []
            
            for card in card_detections:
                if self.is_card_inside_bucket(card['bbox'], bucket['bbox']):
                    has_card = True
                    cards_in_bucket.append(card)
            
            # Draw bucket bounding box - GREEN if has card, RED if empty
            color = (0, 255, 0) if has_card else (0, 0, 255)  # Green/Red
            thickness = 3
            
            cv2.rectangle(processed_frame, (bx1, by1), (bx2, by2), color, thickness)
            
            # Draw bucket label
            status = "✓ HAS CARD" if has_card else "✗ EMPTY"
            bucket_label = f"BUCKET ({status})"
            conf_text = f"Conf: {bucket['confidence']:.3f}"
            query_text = f"Query: {bucket['text_query'][:25]}..."
            
            # Label background
            label_height = 60
            cv2.rectangle(processed_frame, (bx1, by1 - label_height), 
                         (bx1 + 300, by1), color, -1)
            
            # Draw text
            cv2.putText(processed_frame, bucket_label, (bx1 + 5, by1 - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(processed_frame, conf_text, (bx1 + 5, by1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(processed_frame, query_text, (bx1 + 5, by1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw all detected cards with blue boxes
        for card in card_detections:
            cx1, cy1, cx2, cy2 = card['bbox']
            
            # Blue bounding box for cards
            cv2.rectangle(processed_frame, (cx1, cy1), (cx2, cy2), (255, 0, 0), 2)
            
            # Add semi-transparent overlay to make cards more visible
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (255, 100, 0), -1)
            processed_frame = cv2.addWeighted(processed_frame, 0.85, overlay, 0.15, 0)
            
            # Card label
            card_label = f"CARD: {card['confidence']:.3f}"
            query_text = card['text_query'][:20] + "..."
            
            # Label background
            cv2.rectangle(processed_frame, (cx1, cy1 - 35), (cx1 + 200, cy1), (255, 0, 0), -1)
            cv2.putText(processed_frame, card_label, (cx1 + 2, cy1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(processed_frame, query_text, (cx1 + 2, cy1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return processed_frame

    def draw_ui_elements(self, frame):
        """Draw simple UI with detection info"""
        # Simple info panel
        panel_height = 100
        cv2.rectangle(frame, (10, 10), (500, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (500, panel_height), (255, 255, 255), 2)
        
        info_text = [
            "🦉 OWL-ViT Simple Detection",
            "🎯 Green = Bucket with card | Red = Empty bucket",
            "🔵 Blue = Detected color cards"
        ]
        
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main processing loop"""
        print("🚀 Starting simple OWL-ViT detection...")
        print("🎯 Focus: Bucket and card detection only")
        print("⌨️  Press 'q' to quit, 'p' to pause/resume, 's' to save frame")
        
        paused = False
        frame_count = 0
        saved_frames = 0
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("📹 End of video reached")
                    break
                
                frame_count += 1
                
                # Process every 2nd frame for better performance on RTX 3050
                if frame_count % 2 == 0:
                    start_time = time.time()
                    processed_frame = self.process_frame(frame)
                    processing_time = time.time() - start_time
                    
                    if frame_count % 60 == 0:  # Print timing every 60 frames
                        fps = 1.0 / processing_time if processing_time > 0 else 0
                        print(f"⏱️  Processing: {processing_time:.3f}s/frame (~{fps:.1f} FPS)")
                        
                        # Memory usage info
                        if self.device == "cuda":
                            memory_used = torch.cuda.memory_allocated() / 1e9
                            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                            print(f"💾 GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used")
                else:
                    processed_frame = frame  # Skip processing for performance
                
                # Draw UI
                self.draw_ui_elements(processed_frame)
                
                # Save frame
                if self.out:
                    self.out.write(processed_frame)
                
                # Display
                cv2.imshow('Simple OWL-ViT Detection', processed_frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'⏸️  Paused' if paused else '▶️  Resumed'}")
            elif key == ord('s'):
                # Save current frame
                if not paused:
                    save_path = f"saved_frame_{saved_frames:03d}.jpg"
                    cv2.imwrite(save_path, processed_frame)
                    saved_frames += 1
                    print(f"💾 Saved frame: {save_path}")
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\n" + "="*50)
        print("🦉 SIMPLE DETECTION COMPLETE")
        print("="*50)
        print("✅ Detection focused on buckets and cards only")
        print("🎯 Green boxes = Buckets with cards")
        print("🔴 Red boxes = Empty buckets") 
        print("🔵 Blue boxes = Detected cards")
        print("="*50)
        
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("🦉 Simple OWL-ViT Detection System")
    print("=" * 50)
    print("🎯 Simplified: Just detect buckets and cards")
    print("✅ Green = Bucket with card inside")
    print("❌ Red = Empty bucket")
    print("🔵 Blue = Color cards")
    print("=" * 50)
    
    # Configuration
    VIDEO_PATH = "assets/input video/test_video1.mp4"  # Update this!
    OUTPUT_PATH = "assets/output videos/owl_vit_simple_detection_output.mp4"  # Optional
    
    try:
        detector = SimpleOwlViTDetection(VIDEO_PATH, OUTPUT_PATH)
        detector.run()
    except FileNotFoundError:
        print(f"❌ Error: Video file not found at {VIDEO_PATH}")
        print("💡 Please update VIDEO_PATH with your actual video file path")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you have: pip install transformers torch torchvision pillow opencv-python")

def test_single_image():
    """Test on single image"""
    print("🖼️ Testing single image...")
    
    # For single image testing
    image_path = "assets/contact_card1.jpeg"  # Update this!

    # Initialize detector
    detector = SimpleOwlViTDetection(0)  # Dummy video path
    
    # Load and process image
    frame = cv2.imread(image_path)
    if frame is not None:
        processed = detector.process_frame(frame)
        detector.draw_ui_elements(processed)
        
        # Save result
        cv2.imwrite("test_result.jpg", processed)
        print("💾 Result saved as: test_result.jpg")
        
        # Display
        cv2.imshow("Test Result", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ Could not load image: {image_path}")

def test_webcam():
    """Test with webcam"""
    print("📹 Testing with webcam...")
    detector = SimpleOwlViTDetection(0)  # 0 = webcam
    detector.run()

if __name__ == "__main__":
    main()
    
    # Uncomment to test other modes:
    # test_single_image()
    # test_webcam()
