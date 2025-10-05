import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import sys
import threading

# Initialize GStreamer
Gst.init(None)

# Global frame counter
frame_count = 0

# Callback for cairooverlay to draw on each frame
def draw_callback(overlay, context, timestamp, user_data):
    global frame_count
    frame_count += 1
    ctx = context  # Cairo context
    width = overlay.get_property("video-width")
    height = overlay.get_property("video-height")

    # Draw frame count text
    ctx.set_source_rgb(0, 1, 0)  # green
    ctx.select_font_face("Sans", 0, 0)
    ctx.set_font_size(40)
    ctx.move_to(10, 50)
    ctx.show_text(f"Frame count: {frame_count}")

    # Draw a dummy rectangle to simulate detection
    ctx.set_source_rgb(1, 0, 0)  # red
    ctx.rectangle(100, 100, 200, 150)
    ctx.set_line_width(5)
    ctx.stroke()

# Create GStreamer pipeline
pipeline = Gst.parse_launch(
    "autovideosrc ! videoconvert ! cairooverlay name=overlay ! autovideosink"
)

# Get the cairooverlay element
overlay = pipeline.get_by_name("overlay")
overlay.connect("draw", draw_callback)

# Start the pipeline
pipeline.set_state(Gst.State.PLAYING)

# Run the GLib MainLoop in the main thread
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print("Exiting...")
finally:
    pipeline.set_state(Gst.State.NULL)
