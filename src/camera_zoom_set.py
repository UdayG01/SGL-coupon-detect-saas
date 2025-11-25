import tkinter as tk
from tkinter import messagebox
import subprocess

def set_camera_zoom(zoom_value):
    """Set the zoom of the camera using v4l2-ctl"""
    try:
        cmd = ["v4l2-ctl", "-d", "/dev/video0", "--set-ctrl", f"zoom_absolute={zoom_value}"]
        subprocess.run(cmd, check=True)
        messagebox.showinfo("Success", f"✅ Zoom set to {zoom_value}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"❌ Failed to set zoom.\n\n{e}")
    except FileNotFoundError:
        messagebox.showerror("Error", "v4l2-ctl not found. Please install it using:\n\nsudo apt install v4l-utils")

def get_current_zoom():
    """Get the current zoom value using v4l2-ctl"""
    try:
        cmd = ["v4l2-ctl", "-d", "/dev/video0", "--get-ctrl=zoom_absolute"]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        zoom_value = result.stdout.strip().split(":")[-1].strip()
        zoom_entry.delete(0, tk.END)
        zoom_entry.insert(0, zoom_value)
        messagebox.showinfo("Current Zoom", f"🔍 Current Zoom: {zoom_value}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"❌ Failed to get current zoom.\n\n{e}")
    except FileNotFoundError:
        messagebox.showerror("Error", "v4l2-ctl not found. Please install it using:\n\nsudo apt install v4l-utils")

def on_submit():
    """Handle zoom value submission"""
    value = zoom_entry.get()
    if not value.isdigit():
        messagebox.showwarning("Invalid Input", "Please enter a valid integer zoom value.")
        return
    zoom_value = int(value)
    if zoom_value < 100 or zoom_value > 300:
        messagebox.showwarning("Out of Range", "Zoom value must be between 100 and 300.")
        return
    set_camera_zoom(zoom_value)

# ----------------------------
# Tkinter GUI Setup
# ----------------------------
root = tk.Tk()
root.title("Camera Zoom Controller")
root.geometry("320x220")
root.resizable(False, False)

# Title Label
tk.Label(root, text="Camera Zoom Controller", font=("Arial", 14, "bold")).pack(pady=10)

# Entry Frame
frame = tk.Frame(root)
frame.pack(pady=5)

tk.Label(frame, text="Zoom Value (100–300):", font=("Arial", 11)).grid(row=0, column=0, padx=5)
zoom_entry = tk.Entry(frame, width=10, font=("Arial", 11))
zoom_entry.grid(row=0, column=1)

# Buttons Frame
btn_frame = tk.Frame(root)
btn_frame.pack(pady=15)

apply_btn = tk.Button(btn_frame, text="Apply Zoom", font=("Arial", 11, "bold"), bg="#4CAF50", fg="white",
                      padx=10, pady=5, command=on_submit)
apply_btn.grid(row=0, column=0, padx=5)

get_btn = tk.Button(btn_frame, text="Get Current Zoom", font=("Arial", 11, "bold"), bg="#2196F3", fg="white",
                    padx=10, pady=5, command=get_current_zoom)
get_btn.grid(row=0, column=1, padx=5)

# Run the app
root.mainloop()
