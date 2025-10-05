import customtkinter as ctk
import json
import os
from typing import List
from PIL import Image

# ----------------------------
# Configuration
# ----------------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Professional Color Palette (as per your reference)
COLORS = {
    "primary": "#2A3042",           # Deep navy — main actions, title
    "secondary": "#5D6B82",          # Slate blue — secondary buttons
    "background": "#E0E0E0",         # Light gray — app background
    "card": "#FDFDFF",               # Pure white — cards, inputs
    "text_primary": "#1E293B",       # Near-black — headings/body
    "text_secondary": "#6B7280",     # Medium gray — helper text
    "border": "#CBD5E1",             # Light border
    "hover": "#E2E8F0",              # Very light gray hover
    "accent": "#FA6B3B",             # Warm coral — only for "Start Detection"
    "danger": "#DC2626",             # Muted red — delete
    "success": "#10B981",            # Muted green — update
    "info": "#60A5FA",               # Light blue — view
}

DB_FILE = "sku_database.json"

# ----------------------------
# Helper: Load/Save SKU DB
# ----------------------------
def load_sku_db() -> List[str]:
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_sku_db(skus: List[str]):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(skus, f, indent=2)

# ----------------------------
# Searchable Dropdown Component
# ----------------------------
class SearchableDropdown(ctk.CTkFrame):
    def __init__(self, parent, values, width=300, height=36, **kwargs):
        super().__init__(parent, fg_color="transparent", width=width, height=height)
        self.values = values
        self.selected_value = None
        self.popup = None

        self.button = ctk.CTkButton(
            self,
            text="Select an option...",
            width=width,
            height=height,
            font=("Segoe UI", 13),
            fg_color=COLORS["card"],
            text_color=COLORS["text_primary"],
            border_color=COLORS["border"],
            border_width=1,
            hover_color=COLORS["hover"],
            anchor="w",
            command=self.open_popup
        )
        self.button.pack(fill="both", expand=True)

    def open_popup(self):
        if self.popup and self.popup.winfo_exists():
            self.popup.destroy()

        self.popup = ctk.CTkToplevel(self)
        self.popup.title("")
        self.popup.geometry(f"{self.button.winfo_width()}x300+{self.winfo_rootx()}+{self.winfo_rooty() + self.winfo_height()}")
        self.popup.overrideredirect(True)
        self.popup.configure(fg_color=COLORS["card"])
        self.popup.focus()

        self.search_var = ctk.StringVar()
        self.search_var.trace("w", self.filter_options)
        search_entry = ctk.CTkEntry(
            self.popup,
            textvariable=self.search_var,
            placeholder_text="Search...",
            height=30,
            font=("Segoe UI", 12)
        )
        search_entry.pack(fill="x", padx=5, pady=5)
        search_entry.focus()

        list_frame = ctk.CTkFrame(self.popup, fg_color="transparent")
        list_frame.pack(fill="both", expand=True, padx=5, pady=(0, 5))

        self.list_canvas = ctk.CTkCanvas(list_frame, bg=COLORS["card"], highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(list_frame, command=self.list_canvas.yview)
        self.list_canvas.configure(yscrollcommand=scrollbar.set)

        self.list_inner = ctk.CTkFrame(self.list_canvas, fg_color="transparent")
        self.list_canvas.create_window((0, 0), window=self.list_inner, anchor="nw")

        self.list_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.update_list(self.values)
        self.popup.bind("<FocusOut>", lambda e: self.close_popup())

    def filter_options(self, *args):
        query = self.search_var.get().lower()
        filtered = [v for v in self.values if query in v.lower()]
        self.update_list(filtered)

    def update_list(self, items):
        for widget in self.list_inner.winfo_children():
            widget.destroy()

        if not items:
            label = ctk.CTkLabel(self.list_inner, text="No matches found", text_color=COLORS["text_secondary"])
            label.pack(pady=10)
        else:
            for item in items:
                btn = ctk.CTkButton(
                    self.list_inner,
                    text=item,
                    height=28,
                    fg_color="transparent",
                    text_color=COLORS["text_primary"],
                    hover_color=COLORS["hover"],
                    anchor="w",
                    command=lambda x=item: self.select_item(x)
                )
                btn.pack(fill="x", padx=2, pady=1)

        self.list_inner.update_idletasks()
        self.list_canvas.config(scrollregion=self.list_canvas.bbox("all"))

    def select_item(self, value):
        self.selected_value = value
        self.button.configure(text=value)
        self.close_popup()

    def close_popup(self):
        if self.popup:
            self.popup.destroy()
            self.popup = None

    def get(self):
        return self.selected_value

    def set(self, value):
        if value in self.values:
            self.selected_value = value
            self.button.configure(text=value)
        else:
            self.selected_value = None
            self.button.configure(text="Select an option...")

# ----------------------------
# Management Windows
# ----------------------------
class AddSKUWindow(ctk.CTkToplevel):
    def __init__(self, parent, sku_list, callback):
        super().__init__(parent)
        self.parent = parent
        self.sku_list = sku_list
        self.callback = callback
        self.title("Add New SKU")
        self.geometry("400x300")
        self.configure(fg_color=COLORS["background"])
        self.grab_set()

        ctk.CTkLabel(self, text="Add New SKU", font=("Segoe UI", 18, "bold"), text_color=COLORS["primary"]).pack(pady=15)

        self.name_entry = self._create_input("Name")
        self.grade_entry = self._create_input("Grade")
        self.pack_entry = self._create_input("Pack Size (kg)")

        ctk.CTkButton(
            self, text="Add SKU",
            command=self.add_sku,
            fg_color=COLORS["secondary"],
            hover_color="#6B7C93",
            text_color="white"
        ).pack(pady=20)

    def _create_input(self, label_text):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(frame, text=label_text, width=100, anchor="w", text_color=COLORS["text_primary"]).pack(side="left")
        entry = ctk.CTkEntry(frame, width=200)
        entry.pack(side="right")
        return entry

    def add_sku(self):
        name = self.name_entry.get().strip()
        grade = self.grade_entry.get().strip()
        pack = self.pack_entry.get().strip()
        if not all([name, grade, pack]):
            ctk.CTkLabel(self, text="All fields required!", text_color="red").pack()
            return
        sku_code = f"{name}_{grade}_{pack}kg"
        if sku_code in self.sku_list:
            ctk.CTkLabel(self, text="SKU already exists!", text_color="red").pack()
            return
        self.sku_list.append(sku_code)
        self.callback()
        self.destroy()

class DeleteSKUWindow(ctk.CTkToplevel):
    def __init__(self, parent, sku_list, callback):
        super().__init__(parent)
        self.parent = parent
        self.sku_list = sku_list
        self.callback = callback
        self.title("Delete SKU")
        self.geometry("400x200")
        self.configure(fg_color=COLORS["background"])
        self.grab_set()

        ctk.CTkLabel(self, text="Select SKU to Delete", font=("Segoe UI", 16), text_color=COLORS["primary"]).pack(pady=15)

        if not sku_list:
            ctk.CTkLabel(self, text="No SKUs to delete!").pack()
            return

        self.delete_dropdown = SearchableDropdown(
            self,
            values=sku_list,
            width=300
        )
        self.delete_dropdown.pack(pady=10)
        self.delete_dropdown.set(sku_list[0])

        ctk.CTkButton(
            self, text="Delete",
            command=self.delete_sku,
            fg_color=COLORS["danger"],
            hover_color="#EF4444",
            text_color="white"
        ).pack(pady=15)

    def delete_sku(self):
        sku = self.delete_dropdown.get()
        if sku in self.sku_list:
            self.sku_list.remove(sku)
            self.callback()
        self.destroy()

class UpdateSKUWindow(ctk.CTkToplevel):
    def __init__(self, parent, sku_list, callback):
        super().__init__(parent)
        self.parent = parent
        self.sku_list = sku_list
        self.callback = callback
        self.title("Update SKU")
        self.geometry("400x350")
        self.configure(fg_color=COLORS["background"])
        self.grab_set()

        ctk.CTkLabel(self, text="Update Existing SKU", font=("Segoe UI", 18, "bold"), text_color=COLORS["primary"]).pack(pady=15)

        if not sku_list:
            ctk.CTkLabel(self, text="No SKUs to update!").pack()
            return

        self.update_dropdown = SearchableDropdown(
            self,
            values=sku_list,
            width=300
        )
        self.update_dropdown.pack(pady=10)
        self.update_dropdown.set(sku_list[0])

        self.name_entry = self._create_input("New Name")
        self.grade_entry = self._create_input("New Grade")
        self.pack_entry = self._create_input("New Pack Size (kg)")

        ctk.CTkButton(
            self, text="Update SKU",
            command=self.update_sku,
            fg_color=COLORS["success"],
            hover_color="#34D399",
            text_color="white"
        ).pack(pady=20)

    def _create_input(self, label_text):
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(frame, text=label_text, width=120, anchor="w", text_color=COLORS["text_primary"]).pack(side="left")
        entry = ctk.CTkEntry(frame, width=180)
        entry.pack(side="right")
        return entry

    def update_sku(self):
        old_sku = self.update_dropdown.get()
        name = self.name_entry.get().strip()
        grade = self.grade_entry.get().strip()
        pack = self.pack_entry.get().strip()

        if not all([name, grade, pack]):
            ctk.CTkLabel(self, text="All fields required!", text_color="red").pack()
            return

        new_sku = f"{name}_{grade}_{pack}"
        if new_sku in self.sku_list and new_sku != old_sku:
            ctk.CTkLabel(self, text="New SKU already exists!", text_color="red").pack()
            return

        idx = self.sku_list.index(old_sku)
        self.sku_list[idx] = new_sku
        self.callback()
        self.destroy()

class ViewAllWindow(ctk.CTkToplevel):
    def __init__(self, parent, sku_list):
        super().__init__(parent)
        self.title("All SKUs")
        self.geometry("600x450")
        self.configure(fg_color=COLORS["background"])
        self.grab_set()

        ctk.CTkLabel(
            self,
            text="All SKUs",
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS["primary"]
        ).pack(pady=(15, 5))

        table_frame = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=8)
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)

        header_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(5, 0))

        headers = ["Name", "Grade", "Pack Size (Kg)"]
        for i, header in enumerate(headers):
            ctk.CTkLabel(
                header_frame,
                text=header,
                font=("Segoe UI", 13, "bold"),
                text_color=COLORS["primary"],
                width=150 if i == 0 else 100,
                anchor="w"
            ).grid(row=0, column=i, padx=5, sticky="w")

        canvas = ctk.CTkCanvas(table_frame, bg=COLORS["card"], highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ctk.CTkScrollbar(table_frame, command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        inner_frame = ctk.CTkFrame(canvas, fg_color="transparent")
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")

        for idx, sku in enumerate(sku_list):
            parts = sku.split("_", 2)
            if len(parts) == 3:
                name, grade, pack_size = parts[0], parts[1], parts[2]
            else:
                name, grade, pack_size = sku, "N/A", "N/A"

            row_frame = ctk.CTkFrame(inner_frame, fg_color="transparent")
            row_frame.grid(row=idx, column=0, sticky="ew", padx=10, pady=2)

            ctk.CTkLabel(row_frame, text=name, font=("Segoe UI", 12), text_color=COLORS["text_primary"], width=150, anchor="w").grid(row=0, column=0, padx=5, sticky="w")
            ctk.CTkLabel(row_frame, text=grade, font=("Segoe UI", 12), text_color=COLORS["text_primary"], width=100, anchor="w").grid(row=0, column=1, padx=5, sticky="w")
            ctk.CTkLabel(row_frame, text=pack_size, font=("Segoe UI", 12), text_color=COLORS["text_primary"], width=100, anchor="w").grid(row=0, column=2, padx=5, sticky="w")

        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

# ----------------------------
# Main App Class
# ----------------------------
class CouponDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Coupon Detector")
        self.geometry("800x600")
        self.configure(fg_color=COLORS["background"])

        self.sku_list = load_sku_db()
        self.selected_sku = None

        self.create_widgets()
        self.update_dropdowns()

    def create_widgets(self):
        # Header with logos and title
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(30, 15), ipady=10)

        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        header.grid_columnconfigure(2, weight=1)

        try:
            from PIL import Image

            img_left = ctk.CTkImage(
                light_image=Image.open("src/logo_left.png"),
                dark_image=Image.open("src/logo_left.png"),
                size=(150, 60)
            )
            self.logo_left = ctk.CTkLabel(header, image=img_left, text="")
            self.logo_left.image = img_left
            self.logo_left.grid(row=0, column=0, sticky="w", padx=(0, 20))

            title = ctk.CTkLabel(
                header,
                text="Coupon Detector",
                font=("Segoe UI", 28, "bold"),
                text_color=COLORS["primary"]  # Deep navy
            )
            title.grid(row=0, column=1, padx=(20, 20), pady=5)

            img_right = ctk.CTkImage(
                light_image=Image.open("src/logo_right.png"),
                dark_image=Image.open("src/logo_right.png"),
                size=(150, 50)
            )
            self.logo_right = ctk.CTkLabel(header, image=img_right, text="")
            self.logo_right.image = img_right
            self.logo_right.grid(row=0, column=2, sticky="e", padx=(20, 0))

        except Exception as e:
            print(f"[WARNING] Logo loading failed: {e}")
            self.logo_left = ctk.CTkLabel(header, text="LOGO LEFT", font=("Segoe UI", 12), text_color=COLORS["text_secondary"])
            self.logo_left.grid(row=0, column=0, sticky="w", padx=(0, 20))

            title = ctk.CTkLabel(header, text="Coupon Detector", font=("Segoe UI", 28, "bold"), text_color=COLORS["primary"])
            title.grid(row=0, column=1, padx=(20, 20), pady=5)

            self.logo_right = ctk.CTkLabel(header, text="LOGO RIGHT", font=("Segoe UI", 12), text_color=COLORS["text_secondary"])
            self.logo_right.grid(row=0, column=2, sticky="e", padx=(20, 0))

        # Main content
        self.main_frame = ctk.CTkFrame(
            self,
            fg_color=COLORS["card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        ctk.CTkLabel(
            self.main_frame,
            text="Select SKU Code:",
            font=("Segoe UI", 14),
            text_color=COLORS["text_primary"]
        ).pack(pady=(20, 5))

        self.sku_dropdown = SearchableDropdown(
            self.main_frame,
            values=self.sku_list,
            width=300,
            height=36
        )
        self.sku_dropdown.pack(pady=10)

        # Only one vibrant button: Start Detection
        self.detect_btn = ctk.CTkButton(
            self.main_frame,
            text="Start Detection",
            command=self.submit,
            width=200,
            height=40,
            font=("Segoe UI", 14, "bold"),
            fg_color=COLORS["accent"],      # Warm coral
            hover_color="#F99A76",          # Lighter coral
            text_color="white"
        )
        self.detect_btn.pack(pady=20)

        # Management buttons — all muted
        mgmt_frame = ctk.CTkFrame(self, fg_color="transparent")
        mgmt_frame.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(
            mgmt_frame,
            text="Add SKU",
            command=self.open_add_sku,
            width=120,
            height=32,
            fg_color=COLORS["secondary"],
            hover_color="#6B7C93",
            text_color="white"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            mgmt_frame,
            text="Delete SKU",
            command=self.open_delete_sku,
            width=120,
            height=32,
            fg_color=COLORS["danger"],
            hover_color="#EF4444",
            text_color="white"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            mgmt_frame,
            text="Update SKU",
            command=self.open_update_sku,
            width=120,
            height=32,
            fg_color=COLORS["success"],
            hover_color="#34D399",
            text_color="white"
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            mgmt_frame,
            text="View All SKUs",
            command=self.view_all_skus,
            width=120,
            height=32,
            fg_color=COLORS["info"],
            hover_color="#93C5FD",
            text_color="white"
        ).pack(side="right", padx=5)

    def update_dropdowns(self):
        self.sku_list = load_sku_db()
        self.sku_dropdown.values = self.sku_list
        if not self.sku_list:
            self.sku_dropdown.set(None)
        else:
            self.sku_dropdown.set(self.sku_list[0])

    def start_detection(self):
        sku = self.sku_dropdown.get()
        if sku:
            self.selected_sku = sku
            print(f"[INFO] Detection started for SKU: {self.selected_sku}")
        else:
            print("[WARN] No valid SKU selected.")

    def open_add_sku(self):
        AddSKUWindow(self, self.sku_list, self.on_sku_change)

    def open_delete_sku(self):
        DeleteSKUWindow(self, self.sku_list, self.on_sku_change)

    def open_update_sku(self):
        UpdateSKUWindow(self, self.sku_list, self.on_sku_change)

    def view_all_skus(self):
        ViewAllWindow(self, self.sku_list)

    def on_sku_change(self):
        save_sku_db(self.sku_list)
        self.update_dropdowns()

    def submit(self):
        dropdown_sku = self.sku_dropdown.get()

        sku = dropdown_sku

        if sku and sku != "Select an option...":
            self.selected_sku = sku
            self.destroy()
        else:
            # Show warning using CTkMessagebox or fallback
            try:
                from CTkMessagebox import CTkMessagebox
                CTkMessagebox(title="Invalid Input", message="Please select or enter a valid SKU code.", icon="warning")
            except ImportError:
                # Fallback to standard messagebox
                from tkinter import messagebox
                messagebox.showwarning("Invalid Input", "Please select or enter a valid SKU code.")
    
    def run(self):
        """Run the app modally and return the selected SKU code after window is closed."""
        self.mainloop()
        return self.selected_sku

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    print("Starting SKU Input Window...")
    app = CouponDetectorApp()
    sku_code = app.run()

    if not sku_code:
        print("No SKU code selected. Exiting...")
        exit(0)

    print(f"Selected SKU Code: {sku_code}")
    # ✅ Now you can use `sku_code` in your detection logic!
    # Example:
    # run_coupon_detection(sku_code)