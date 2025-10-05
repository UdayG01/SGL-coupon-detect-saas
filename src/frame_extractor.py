import cv2
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """
    A class to extract frames from a video at a specified FPS rate.
    
    This is useful for creating image datasets from video for object detection tasks.
    """

    def __init__(self, video_path: str, output_dir: str = "extracted_frames"):
        """
        Initialize the VideoFrameExtractor.
        
        Args:
            video_path (str): Path to the input video file.
            output_dir (str): Directory where extracted frames will be saved.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap: Optional[cv2.VideoCapture] = None

        # Validate video path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")

    def extract_frames(self, target_fps: float = 3.0, max_frames: Optional[int] = None) -> list:
        """
        Extract frames from the video at approximately `target_fps`.
        
        Args:
            target_fps (float): Number of frames to extract per second.
            max_frames (int, optional): Maximum number of frames to extract. If None, extract all.
        
        Returns:
            List[str]: List of saved frame file paths.
        """
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {self.video_path}")

        # Get video properties
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps

        logger.info(f"Video Info: {original_fps:.2f} FPS, {total_frames} frames, {duration:.2f} seconds")

        # Calculate interval between frames to achieve target FPS
        frame_interval = int(round(original_fps / target_fps))
        logger.info(f"Extracting 1 frame every {frame_interval} frames (target: {target_fps} FPS)")

        saved_frames = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Extract frame based on interval
            if frame_count % frame_interval == 0:
                filename = os.path.join(self.output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_frames.append(filename)
                saved_count += 1
                logger.debug(f"Saved: {filename}")

                # Stop if we've reached the maximum number of frames
                if max_frames and saved_count >= max_frames:
                    logger.info(f"Reached maximum frame limit: {max_frames}")
                    break

            frame_count += 1

        self.cap.release()
        logger.info(f"Frame extraction complete. Extracted {saved_count} frames to '{self.output_dir}'")
        return saved_frames

    def __enter__(self):
        """Support for context manager (with statement)."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensure video capture is released on exit."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if exc_type:
            logger.error(f"Exception occurred: {exc_value}")
        return False  # Do not suppress exceptions


# --- Usage Example ---
if __name__ == "__main__":
    VIDEO_PATH = "assets/input video/recording_20250903_151656.avi"

    OUTPUT_DIR_PATH = "dataset/raw_frames"

    try:
        with VideoFrameExtractor(video_path=VIDEO_PATH, output_dir=OUTPUT_DIR_PATH) as extractor:
            frames = extractor.extract_frames(target_fps=3, max_frames=400)
            print(f"Successfully extracted {len(frames)} frames.")
    except Exception as e:
        logger.error(f"Failed to extract frames: {e}")