"""
Test YOLO-World detection on single images
Perfect for debugging card detection without video complexity
"""

import cv2
import numpy as np
from ultralytics import YOLOWorld
import matplotlib.pyplot as plt

class ImageDetectionTester:
    def __init__(self):
        """Initialize YOLO-World model for testing"""
        print("🔧 Loading YOLO-World model...")
        self.model = YOLOWorld("yolov8s-world.pt")
        
        # Define classes - optimized for your color cards
        self.model.set_classes([
            "color card", 
            "color sample",
            "paint color card",
            "colored label",
            "product label",
            "sample card",
            "color chart",
            "color swatch",
            "card",
            "label",
            "sticker",
            "rectangular card",
            "rectangular paper"
        ])
        
        print("✅ Model loaded successfully!")
        print(f"🎯 Detection classes: {self.model.names}")

    def test_image(self, image_path, confidence_threshold=0.1, save_result=True):
        """
        Test detection on a single image
        
        Args:
            image_path: Path to your test image
            confidence_threshold: Detection confidence threshold (0.1 = very sensitive)
            save_result: Whether to save the annotated result
        """
        print(f"\n🖼️ Testing image: {image_path}")
        print(f"🎚️ Confidence threshold: {confidence_threshold}")
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Error: Could not load image from {image_path}")
                return False
                
            print(f"📏 Image size: {image.shape[1]}x{image.shape[0]}")
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
        
        # Run detection
        print("🔍 Running detection...")
        results = self.model(image, verbose=False)
        
        # Process results
        detections = []
        annotated_image = image.copy()
        
        if results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                if confidence >= confidence_threshold:
                    # Get bounding box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'center': center,
                        'area': area
                    }
                    detections.append(detection)
                    
                    # Draw detection on image
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {confidence:.3f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Label background
                    cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Label text
                    cv2.putText(annotated_image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    print(f"  ✅ Found: {class_name} (conf: {confidence:.3f}) at {center}")
        
        # Print results summary
        print(f"\n📊 DETECTION RESULTS:")
        print(f"🔢 Total detections: {len(detections)}")
        
        if len(detections) == 0:
            print("❌ No objects detected!")
            print("💡 Try:")
            print("   - Lower confidence threshold (e.g., 0.05)")
            print("   - Different class descriptions")
            print("   - Check if image contains the target objects")
        else:
            print("✅ Detections found:")
            for i, det in enumerate(detections):
                print(f"   {i+1}. {det['class_name']} - Confidence: {det['confidence']:.3f}")
        
        # Save or display result
        if save_result:
            output_path = image_path.replace('.', '_detected.')
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Saved annotated image: {output_path}")
        
        # Display using matplotlib (better for Jupyter/Colab)
        self.display_result(image, annotated_image, detections)
        
        return len(detections) > 0

    def display_result(self, original, annotated, detections):
        """Display original and annotated images side by side"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Convert BGR to RGB for matplotlib
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Display images
        ax1.imshow(original_rgb)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2.imshow(annotated_rgb)
        ax2.set_title(f"Detections ({len(detections)} found)")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

    def batch_test(self, image_paths, confidence_threshold=0.15):
        """Test multiple images at once"""
        print(f"🔄 Testing {len(image_paths)} images...")
        
        results = {}
        for i, path in enumerate(image_paths):
            print(f"\n--- Image {i+1}/{len(image_paths)} ---")
            success = self.test_image(path, confidence_threshold, save_result=True)
            results[path] = success
        
        # Summary
        print(f"\n📈 BATCH RESULTS:")
        successful = sum(results.values())
        print(f"✅ Successful detections: {successful}/{len(image_paths)}")
        
        for path, success in results.items():
            status = "✅" if success else "❌"
            print(f"{status} {path}")

    def experiment_with_thresholds(self, image_path):
        """Test different confidence thresholds to find optimal setting"""
        print(f"🧪 Experimenting with confidence thresholds on: {image_path}")
        
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            print(f"\n--- Threshold: {threshold} ---")
            success = self.test_image(image_path, threshold, save_result=False)
            if not success:
                print(f"❌ No detections at threshold {threshold}")

    def test_custom_classes(self, image_path, custom_classes):
        """Test with custom class descriptions"""
        print(f"🎨 Testing custom classes: {custom_classes}")
        
        # Temporarily change classes
        original_classes = self.model.names.copy()
        self.model.set_classes(custom_classes)
        
        # Test
        success = self.test_image(image_path, confidence_threshold=0.1)
        
        # Restore original classes
        self.model.set_classes(list(original_classes.values()))
        
        return success

def main():
    """Main function with easy testing options"""
    print("🧪 YOLO-World Image Detection Tester")
    print("="*50)
    
    # Initialize tester
    tester = ImageDetectionTester()
    
    # Test your image - CHANGE THIS PATH!
    image_path = "path/to/your/card_image.jpg"  # UPDATE THIS!
    
    print(f"\n🎯 Testing single image...")
    tester.test_image(image_path, confidence_threshold=0.1)

def test_multiple_images():
    """Test multiple images at once"""
    tester = ImageDetectionTester()
    
    image_paths = [
        "assets/contact_card1.jpeg",
        "assets/contact_card12.jpeg"
    ]
    
    tester.batch_test(image_paths)

def find_best_threshold():
    """Find the best confidence threshold for your images"""
    tester = ImageDetectionTester()
    image_path = "path/to/your/card_image.jpg"  # UPDATE THIS!
    
    tester.experiment_with_thresholds(image_path)

def test_different_descriptions():
    """Test different class descriptions"""
    tester = ImageDetectionTester()
    image_path = "path/to/your/card_image.jpg"  # UPDATE THIS!
    
    # Test different description sets
    description_sets = [
        # Set 1: Generic
        ["card", "label", "sticker"],
        
        # Set 2: Color-specific  
        ["color card", "color sample", "paint sample"],
        
        # Set 3: Detailed
        ["rectangular color card", "small colored label", "product color sample"],
        
        # Set 4: Very specific
        ["paint color swatch", "color test card", "product sample card"]
    ]
    
    for i, classes in enumerate(description_sets):
        print(f"\n🧪 Testing description set {i+1}: {classes}")
        tester.test_custom_classes(image_path, classes)

if __name__ == "__main__":
    # main()
    
    # Uncomment to run other tests:
    test_multiple_images()
    # find_best_threshold() 
    # test_different_descriptions()

"""
🚀 QUICK START GUIDE:

1. Install requirements:
   pip install ultralytics matplotlib

2. Update the image_path in main() function:
   image_path = "your_card_image.jpg"

3. Run the script:
   python image_test.py

4. Check the results:
   - Console output shows detection details
   - Annotated image saved as "*_detected.jpg"
   - Visual display shows before/after

🔧 DEBUGGING TIPS:

If no cards detected:
- Try lower confidence (0.05)
- Test different class descriptions
- Use experiment_with_thresholds()
- Use test_different_descriptions()

🎯 TESTING STRATEGY:

1. Start with a single card image
2. Find optimal confidence threshold  
3. Test different class descriptions
4. Then test bucket+card combinations
5. Finally test full video

📝 EXAMPLE USAGE:

# Test single image
tester = ImageDetectionTester()
tester.test_image("my_card.jpg", confidence_threshold=0.1)

# Find best settings
tester.experiment_with_thresholds("my_card.jpg")

# Test custom descriptions
tester.test_custom_classes("my_card.jpg", ["paint sample", "color chip"])
"""