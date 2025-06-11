import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import opening, disk
import cv2
import threading
import os


class DistanceWatershedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Distance Transform Watershed")
        self.root.geometry("1400x900")

        # Variables
        self.image_rgb = None
        self.image_gray = None
        self.binary_mask = None
        self.distance = None
        self.labels = None
        self.processing = False

        # Default parameters
        self.params = {
            "binary_threshold": 0.5,
            "min_distance": 20,
            "threshold_abs": 0.3,
            "opening_size": 3,
        }

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # File selection
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(
            pady=5
        )

        # Parameters
        ttk.Label(control_frame, text="Parameters:", font=("Arial", 12, "bold")).pack(
            pady=(20, 10)
        )

        # Binary threshold
        ttk.Label(control_frame, text="Binary Threshold:").pack(anchor=tk.W)
        self.binary_var = tk.DoubleVar(value=self.params["binary_threshold"])
        binary_scale = tk.Scale(
            control_frame,
            from_=0.1,
            to=1,
            orient=tk.HORIZONTAL,
            variable=self.binary_var,
            command=self.on_param_change,
            resolution=0.01,
        )
        binary_scale.pack(fill=tk.X, pady=(0, 10))

        # Min distance for local maxima
        ttk.Label(control_frame, text="Min Distance (Local Maxima):").pack(anchor=tk.W)
        self.distance_var = tk.IntVar(value=self.params["min_distance"])
        distance_scale = tk.Scale(
            control_frame,
            from_=1,
            to=50,
            orient=tk.HORIZONTAL,
            variable=self.distance_var,
            command=self.on_param_change,
            resolution=1,
        )
        distance_scale.pack(fill=tk.X, pady=(0, 10))

        # Threshold absolute for local maxima
        ttk.Label(control_frame, text="Peak Threshold (Absolute):").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=self.params["threshold_abs"])
        threshold_scale = tk.Scale(
            control_frame,
            from_=0.1,
            to=1,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.on_param_change,
            resolution=0.01,
        )
        threshold_scale.pack(fill=tk.X, pady=(0, 10))

        # Opening size
        ttk.Label(control_frame, text="Opening Size (Noise Removal):").pack(anchor=tk.W)
        self.opening_var = tk.IntVar(value=self.params["opening_size"])
        opening_scale = tk.Scale(
            control_frame,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.opening_var,
            command=self.on_param_change,
            resolution=1,
        )
        opening_scale.pack(fill=tk.X, pady=(0, 10))

        # Reset button
        ttk.Button(
            control_frame, text="Reset Parameters", command=self.reset_parameters
        ).pack(pady=20)

        # Status
        self.status_var = tk.StringVar(value="Load an image to start")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=280).pack(
            pady=10
        )

        # Right panel for plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
        )

        if file_path:
            try:
                self.image_rgb = cv2.imread(file_path)
                self.image_rgb = cv2.cvtColor(self.image_rgb, cv2.COLOR_BGR2RGB)
                self.image_gray = rgb2gray(self.image_rgb)
                self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
                self.process_watershed()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def on_param_change(self, event=None):
        if self.image_gray is not None and not self.processing:
            # Update parameters
            self.params["binary_threshold"] = self.binary_var.get()
            self.params["min_distance"] = self.distance_var.get()
            self.params["threshold_abs"] = self.threshold_var.get()
            self.params["opening_size"] = self.opening_var.get()

            # Process in thread to avoid blocking UI
            threading.Thread(target=self.process_watershed, daemon=True).start()

    def reset_parameters(self):
        self.params = {
            "binary_threshold": 0.5,
            "min_distance": 20,
            "threshold_abs": 0.3,
            "opening_size": 3,
        }

        self.binary_var.set(self.params["binary_threshold"])
        self.distance_var.set(self.params["min_distance"])
        self.threshold_var.set(self.params["threshold_abs"])
        self.opening_var.set(self.params["opening_size"])

        if self.image_gray is not None:
            self.process_watershed()

    def set_error_status(self, error_msg):
        """Helper method to set error status from thread"""
        self.status_var.set(f"Error: {error_msg}")

    def process_watershed(self):
        if self.image_gray is None:
            return

        self.processing = True
        self.status_var.set("Processing...")

        try:
            # Step 1: Create binary mask using Otsu threshold adjusted by parameter
            otsu_thresh = threshold_otsu(self.image_gray)
            adjusted_thresh = otsu_thresh * self.params["binary_threshold"]
            self.binary_mask = self.image_gray > adjusted_thresh

            # Step 2: Clean up binary mask with morphological opening
            if self.params["opening_size"] > 0:
                self.binary_mask = opening(
                    self.binary_mask, disk(self.params["opening_size"])
                )

            # Step 3: Distance transform
            self.distance = ndi.distance_transform_edt(self.binary_mask)

            # Step 4: Find local maxima coordinates (peak_local_max returns coordinates by default)
            coords = peak_local_max(
                self.distance,
                min_distance=self.params["min_distance"],
                threshold_abs=self.params["threshold_abs"] * np.max(self.distance),
            )

            # Step 5: Create markers from local maxima coordinates
            markers = np.zeros_like(self.distance, dtype=int)
            for i, coord in enumerate(coords):
                markers[coord[0], coord[1]] = i + 1

            # Step 6: Apply watershed
            self.labels = watershed(-self.distance, markers, mask=self.binary_mask)

            # Update display
            self.root.after(0, self.update_display)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self.set_error_status(msg))
        finally:
            self.processing = False

    def update_display(self):
        self.fig.clear()

        if self.image_rgb is None:
            return

        # Create subplots
        ax1 = self.fig.add_subplot(2, 3, 1)
        ax2 = self.fig.add_subplot(2, 3, 2)
        ax3 = self.fig.add_subplot(2, 3, 3)
        ax4 = self.fig.add_subplot(2, 3, 4)
        ax5 = self.fig.add_subplot(2, 3, 5)
        ax6 = self.fig.add_subplot(2, 3, 6)

        # Original image
        ax1.imshow(self.image_rgb)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Grayscale
        ax2.imshow(self.image_gray, cmap="gray")
        ax2.set_title("Grayscale")
        ax2.axis("off")

        # Binary mask
        if self.binary_mask is not None:
            ax3.imshow(self.binary_mask, cmap="gray")
            ax3.set_title("Binary Mask")
            ax3.axis("off")

        # Distance transform
        if self.distance is not None:
            ax4.imshow(self.distance, cmap="viridis")
            ax4.set_title("Distance Transform")
            ax4.axis("off")

            # Local maxima overlay
            coords = peak_local_max(
                self.distance,
                min_distance=self.params["min_distance"],
                threshold_abs=self.params["threshold_abs"] * np.max(self.distance),
            )
            if len(coords) > 0:
                ax4.plot(coords[:, 1], coords[:, 0], "r*", markersize=8)

        # Watershed labels
        if self.labels is not None:
            ax5.imshow(self.labels, cmap="nipy_spectral", alpha=0.8)
            ax5.set_title(f"Watershed Labels ({len(np.unique(self.labels))-1} objects)")
            ax5.axis("off")

            # Overlay on original
            ax6.imshow(self.image_rgb)
            ax6.imshow(self.labels, cmap="nipy_spectral", alpha=0.3)
            ax6.set_title("Segmentation Overlay")
            ax6.axis("off")

        self.fig.tight_layout()
        self.canvas.draw()

        # Update status
        if self.labels is not None:
            num_objects = len(np.unique(self.labels)) - 1
            self.status_var.set(f"Segmentation complete: {num_objects} objects found")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DistanceWatershedGUI(root)
    root.mainloop()
