import cv2
from ultralytics import YOLO

# Paths
video_path = "assets/input video/recording_20250903_151656.avi"         
output_path = "assets/output videos/output2.mp4"       
model_path = "model/best_(2).pt"          # your trained YOLO model

# Load model
model = YOLO(model_path)

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Write to output video
    out.write(annotated_frame)

    # (Optional) Show video live
    cv2.imshow("YOLO Inference", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Inference complete. Output saved at {output_path}")
