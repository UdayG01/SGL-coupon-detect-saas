import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import cv2
import os
from tqdm import tqdm

class PaintBucketCardDetector:
    def __init__(self):
        """Initialize OWL-ViT model and processor"""
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        
        # Text queries for zero-shot detection
        self.text_queries = [
            "paint bucket",
            "paint container", 
            "bucket",
            "rectangular card",
            "card",
            "label",
            "contact card",
            "business card",
            "rectangular paper",
            "sticker",
        ]
        
    def detect_objects(self, image, confidence_threshold=0.1):
        """
        Detect paint buckets and cards in the image
        
        Args:
            image: PIL Image
            confidence_threshold: minimum confidence for detections
            
        Returns:
            dict with paint_buckets and cards lists containing bounding boxes and scores
        """
        # Process inputs
        inputs = self.processor(text=self.text_queries, images=image, return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs
        target_sizes = torch.Tensor([image.size[::-1]])  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=confidence_threshold
        )
        
        # Separate paint buckets and cards
        paint_buckets = []
        cards = []
        
        boxes = results[0]["boxes"].cpu().numpy()
        scores = results[0]["scores"].cpu().numpy()
        labels = results[0]["labels"].cpu().numpy()
        
        for box, score, label in zip(boxes, scores, labels):
            query_text = self.text_queries[label].lower()
            
            # Classify as paint bucket or card based on query text
            if any(keyword in query_text for keyword in ["paint", "bucket", "container"]):
                paint_buckets.append({
                    "box": box,
                    "score": score,
                    "label": query_text
                })
            elif any(keyword in query_text for keyword in ["card", "label", "contact", "business"]):
                cards.append({
                    "box": box, 
                    "score": score,
                    "label": query_text
                })
        
        return {"paint_buckets": paint_buckets, "cards": cards}
    
    def is_box_inside(self, inner_box, outer_box, overlap_threshold=0.8):
        """
        Check if inner_box is inside outer_box
        
        Args:
            inner_box: [x1, y1, x2, y2] - card bounding box
            outer_box: [x1, y1, x2, y2] - paint bucket bounding box
            overlap_threshold: minimum overlap ratio to consider "inside"
            
        Returns:
            bool: True if inner_box is inside outer_box
        """
        # Calculate intersection
        x1_inter = max(inner_box[0], outer_box[0])
        y1_inter = max(inner_box[1], outer_box[1])
        x2_inter = min(inner_box[2], outer_box[2])
        y2_inter = min(inner_box[3], outer_box[3])
        
        # Check if there's an intersection
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return False
        
        # Calculate areas
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
        
        # Calculate overlap ratio
        overlap_ratio = intersection_area / inner_area if inner_area > 0 else 0
        
        return overlap_ratio >= overlap_threshold
    
    def assign_card_colors(self, detections):
        """
        Assign colors to cards based on whether they're inside paint buckets
        
        Args:
            detections: dict from detect_objects()
            
        Returns:
            list of tuples: (card_dict, color) where color is 'green' or 'red'
        """
        paint_buckets = detections["paint_buckets"]
        cards = detections["cards"]
        
        card_colors = []
        
        for card in cards:
            is_inside_bucket = False
            
            # Check if card is inside any paint bucket
            for bucket in paint_buckets:
                if self.is_box_inside(card["box"], bucket["box"]):
                    is_inside_bucket = True
                    break
            
            color = "green" if is_inside_bucket else "red"
            card_colors.append((card, color))
        
        return card_colors
    
    def process_video(self, video_path, output_path=None, confidence_threshold=0.1, 
                     overlap_threshold=0.8, frame_skip=1, max_frames=None):
        """
        Process video frame by frame and create annotated output video
        
        Args:
            video_path: path to input video file
            output_path: path for output video (if None, uses input_name_processed.mp4)
            confidence_threshold: minimum confidence for detections
            overlap_threshold: minimum overlap to consider card "inside" bucket
            frame_skip: process every nth frame (1 = every frame, 2 = every other frame)
            max_frames: maximum number of frames to process (None = all frames)
            
        Returns:
            list of detection results for each processed frame
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video
        if output_path is None:
            base_name = os.path.splitext(video_path)[0]
            output_path = f"{base_name}_processed.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_count = 0
        processed_count = 0
        
        # Determine how many frames to process
        frames_to_process = min(max_frames or total_frames, total_frames)
        
        print(f"Processing video: {frames_to_process} frames (every {frame_skip} frame(s))")
        
        with tqdm(total=frames_to_process // frame_skip) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret or (max_frames and processed_count >= max_frames):
                    break
                
                # Skip frames if needed
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Convert OpenCV frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Detect objects in this frame
                detections = self.detect_objects(pil_image, confidence_threshold)
                card_colors = self.assign_card_colors(detections)
                
                # Create annotated frame
                annotated_pil = self.draw_annotations(pil_image, detections, card_colors)
                
                # Convert back to OpenCV format
                annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Store results
                frame_results.append({
                    "frame_number": frame_count,
                    "detections": detections,
                    "card_colors": card_colors
                })
                
                processed_count += 1
                frame_count += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"Processed video saved to: {output_path}")
        return frame_results
    
    def draw_annotations(self, image, detections, card_colors):
        """
        Draw bounding boxes and labels on image
        
        Args:
            image: PIL Image
            detections: dict from detect_objects()
            card_colors: list from assign_card_colors()
            
        Returns:
            PIL Image with annotations
        """
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw paint buckets in blue
        for bucket in detections["paint_buckets"]:
            box = bucket["box"]
            draw.rectangle(box, outline="blue", width=3)
            draw.text((box[0], max(0, box[1] - 20)), 
                     f"Paint Bucket ({bucket['score']:.2f})", 
                     fill="blue", font=font)
        
        # Draw cards with assigned colors
        for card, color in card_colors:
            box = card["box"]
            draw.rectangle(box, outline=color, width=3)
            status = "Inside Bucket" if color == "green" else "Outside Bucket"
            draw.text((box[0], max(0, box[1] - 20)), 
                     f"Card - {status} ({card['score']:.2f})", 
                     fill=color, font=font)
        
        return img_draw

    def visualize_frame(self, image, detections, card_colors, save_path=None):
        """
        Visualize detection results for a single frame
        
        Args:
            image: PIL Image
            detections: dict from detect_objects()
            card_colors: list from assign_card_colors()
            save_path: optional path to save the visualization
        """
        img_draw = self.draw_annotations(image, detections, card_colors)
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(img_draw)
        plt.axis('off')
        plt.title("Paint Bucket and Card Detection\nBlue=Paint Buckets, Green=Cards Inside, Red=Cards Outside")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
        
        return img_draw

def process_video(video_path, output_path=None, confidence_threshold=0.1, 
                 overlap_threshold=0.8, frame_skip=1, max_frames=None):
    """
    Main function to process a video and create annotated output
    
    Args:
        video_path: path to input video file
        output_path: path for output video (if None, auto-generated)
        confidence_threshold: minimum confidence for detections
        overlap_threshold: minimum overlap to consider card "inside" bucket
        frame_skip: process every nth frame (1=every frame, 5=every 5th frame)
        max_frames: maximum frames to process (None=all frames)
        
    Returns:
        list of detection results for each frame
    """
    detector = PaintBucketCardDetector()
    return detector.process_video(
        video_path, output_path, confidence_threshold, 
        overlap_threshold, frame_skip, max_frames
    )

def process_image(image_path, confidence_threshold=0.1, overlap_threshold=0.8):
    """
    Analyze detection results across all frames
    
    Args:
        frame_results: list of frame detection results
        
    Returns:
        dict with analysis statistics
    """
    total_frames = len(frame_results)
    frames_with_buckets = 0
    frames_with_cards = 0
    frames_with_cards_inside = 0
    
    total_buckets = 0
    total_cards = 0
    total_cards_inside = 0
    
    for result in frame_results:
        buckets = result["detections"]["paint_buckets"]
        cards_info = result["card_colors"]
        
        if buckets:
            frames_with_buckets += 1
            total_buckets += len(buckets)
        
        if cards_info:
            frames_with_cards += 1
            total_cards += len(cards_info)
            
            cards_inside = sum(1 for _, color in cards_info if color == "green")
            if cards_inside > 0:
                frames_with_cards_inside += 1
            total_cards_inside += cards_inside
    
    analysis = {
        "total_frames_processed": total_frames,
        "frames_with_buckets": frames_with_buckets,
        "frames_with_cards": frames_with_cards,
        "frames_with_cards_inside_buckets": frames_with_cards_inside,
        "total_buckets_detected": total_buckets,
        "total_cards_detected": total_cards,
        "total_cards_inside_buckets": total_cards_inside,
        "avg_buckets_per_frame": total_buckets / total_frames if total_frames > 0 else 0,
        "avg_cards_per_frame": total_cards / total_frames if total_frames > 0 else 0,
        "percentage_cards_inside": (total_cards_inside / total_cards * 100) if total_cards > 0 else 0
    }
    
    return analysis
    """
    Main function to process an image and detect paint buckets and cards
    
    Args:
        image_path: path to image file or PIL Image
        confidence_threshold: minimum confidence for detections
        overlap_threshold: minimum overlap to consider card "inside" bucket
    """
    # Initialize detector
    detector = PaintBucketCardDetector()
    
    # Load image
    if isinstance(image_path, str):
        if image_path.startswith('http'):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
    else:
        image = image_path  # Already a PIL Image
    
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    print(f"Processing image of size: {image.size}")
    
    # Detect objects
    print("Detecting objects...")
    detections = detector.detect_objects(image, confidence_threshold)
    
    print(f"Found {len(detections['paint_buckets'])} paint buckets")
    print(f"Found {len(detections['cards'])} cards")
    
    # Assign colors to cards
    card_colors = detector.assign_card_colors(detections)
    
    # Print results
    print("\nResults:")
    for i, (card, color) in enumerate(card_colors):
        status = "inside a paint bucket" if color == "green" else "outside paint buckets"
        print(f"Card {i+1}: {status} (confidence: {card['score']:.3f})")
    
    # Visualize results
    result_image = detector.visualize_frame(image, detections, card_colors)
    
    return detections, card_colors, result_image

def analyze_video_results(frame_results):
    """
    Analyze detection results across all frames
    
    Args:
        frame_results: list of frame detection results
        
    Returns:
        dict with analysis statistics
    """
    total_frames = len(frame_results)
    frames_with_buckets = 0
    frames_with_cards = 0
    frames_with_cards_inside = 0
    
    total_buckets = 0
    total_cards = 0
    total_cards_inside = 0
    
    for result in frame_results:
        buckets = result["detections"]["paint_buckets"]
        cards_info = result["card_colors"]
        
        if buckets:
            frames_with_buckets += 1
            total_buckets += len(buckets)
        
        if cards_info:
            frames_with_cards += 1
            total_cards += len(cards_info)
            
            cards_inside = sum(1 for _, color in cards_info if color == "green")
            if cards_inside > 0:
                frames_with_cards_inside += 1
            total_cards_inside += cards_inside
    
    analysis = {
        "total_frames_processed": total_frames,
        "frames_with_buckets": frames_with_buckets,
        "frames_with_cards": frames_with_cards,
        "frames_with_cards_inside_buckets": frames_with_cards_inside,
        "total_buckets_detected": total_buckets,
        "total_cards_detected": total_cards,
        "total_cards_inside_buckets": total_cards_inside,
        "avg_buckets_per_frame": total_buckets / total_frames if total_frames > 0 else 0,
        "avg_cards_per_frame": total_cards / total_frames if total_frames > 0 else 0,
        "percentage_cards_inside": (total_cards_inside / total_cards * 100) if total_cards > 0 else 0
    }
    
    return analysis

# Example usage
if __name__ == "__main__":
    # Example 1: Process a single image
    # detections, card_colors, result_img = process_image("your_image.jpg")
    
    # Example 2: Process a video (basic)
    # frame_results = process_video("your_video.mp4")
    
    # Example 3: Process video with custom settings
    frame_results = process_video(
        "assets/input video/input_video.mp4",
        output_path="assets/output videos/annotated_output2.mp4",
        confidence_threshold=0.05,  # Lower for more detections
        overlap_threshold=0.7,      # Lower for looser "inside" criteria
        frame_skip=1
    )
    
    # Example 4: Analyze video results
    # frame_results = process_video("your_video.mp4")
    # analysis = analyze_video_results(frame_results)
    # print("Video Analysis:")
    # for key, value in analysis.items():
    #     print(f"{key}: {value}")
    
    print("Video Processing Usage:")
    print("1. Basic: process_video('video.mp4')")
    print("2. Custom: process_video('video.mp4', frame_skip=5, max_frames=100)")
    print("3. Analysis: analyze_video_results(frame_results)")
    print("\nImage Processing Usage:")
    print("1. Basic: process_image('image.jpg')")
    print("2. Custom: process_image('image.jpg', confidence_threshold=0.05)")
    
    print("\nRequired packages:")
    print("pip install transformers torch pillow opencv-python matplotlib tqdm")