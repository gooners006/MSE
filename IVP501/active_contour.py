import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2
import threading
import os


class ActiveContourGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Active Contour Parameter Tuner")
        self.root.geometry("1400x900")

        # Initialize variables
        self.img = None
        self.init_contour = None
        self.current_snake = None
        self.processing = False
        self.auto_update_timer = None
        self.update_delay = 200  # shorter delay since we're only updating on release
        self.param_defaults = {}  # Store default values for reset

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding="5")
        file_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        ttk.Button(file_frame, text="Select Image", command=self.load_image).grid(
            row=0, column=0, padx=(0, 10)
        )
        self.file_label = ttk.Label(file_frame, text="No image selected")
        self.file_label.grid(row=0, column=1, sticky=tk.W)

        # Parameter control frame
        control_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        control_frame.grid(
            row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10)
        )

        # Parameters
        self.params = {}
        param_configs = [
            ("Alpha (Contraction)", "alpha", 0.001, 0.1, 0.015, 0.005),
            (
                "Beta (Smoothness)",
                "beta",
                0.1,
                20,
                10,
                0.5,
            ),
            ("Sigma (Blur)", "sigma", 0.1, 5, 1, 0.1),
            ("W_Edge (Edge Attraction)", "w_edge", -2, 2, 1, 0.1),
            (
                "W_Line (Brightness)",
                "w_line",
                -5,
                5,
                0,
                0.2,
            ),
            (
                "Gamma (Time Step)",
                "gamma",
                0.0001,
                0.1,
                0.001,
                0.001,
            ),
            ("Max Iterations", "max_iter", 100, 5000, 2500, 100),
        ]

        for i, (label, key, min_val, max_val, default, step) in enumerate(
            param_configs
        ):
            ttk.Label(control_frame, text=label).grid(
                row=i, column=0, sticky=tk.W, pady=2
            )

            if key == "max_iter":
                scale = tk.Scale(
                    control_frame,
                    from_=min_val,
                    to=max_val,
                    resolution=step,
                    orient=tk.HORIZONTAL,
                    length=200,
                )
            else:
                scale = tk.Scale(
                    control_frame,
                    from_=min_val,
                    to=max_val,
                    resolution=step,
                    orient=tk.HORIZONTAL,
                    length=200,
                    digits=4,
                )

            scale.set(default)
            scale.grid(row=i, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)

            # Bind mouse release event instead of command
            scale.bind("<ButtonRelease-1>", self.on_parameter_release)
            scale.bind(
                "<Button-1>", self.on_parameter_click
            )  # Track when dragging starts

            self.params[key] = scale
            self.param_defaults[key] = default  # Store default value

        control_frame.columnconfigure(1, weight=1)

        # Initial contour parameters
        ttk.Label(control_frame, text="").grid(
            row=len(param_configs), column=0
        )  # Spacer

        contour_configs = [
            ("Center Row", "center_row", 0, 1080, 200, 10),
            ("Center Col", "center_col", 0, 1080, 200, 10),
            ("Radius Row", "radius_row", 20, 600, 20, 10),
            ("Radius Col", "radius_col", 20, 600, 20, 10),
        ]

        for i, (label, key, min_val, max_val, default, step) in enumerate(
            contour_configs
        ):
            row_idx = len(param_configs) + 1 + i
            ttk.Label(control_frame, text=label).grid(
                row=row_idx, column=0, sticky=tk.W, pady=2
            )

            scale = tk.Scale(
                control_frame,
                from_=min_val,
                to=max_val,
                resolution=step,
                orient=tk.HORIZONTAL,
                length=200,
            )
            scale.set(default)
            scale.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=2)

            # Bind mouse release events for contour parameters
            scale.bind("<ButtonRelease-1>", self.on_contour_release)
            scale.bind("<Button-1>", self.on_contour_click)

            self.params[key] = scale
            self.param_defaults[key] = default  # Store default value

        # Control buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(
            row=len(param_configs) + len(contour_configs) + 2,
            column=0,
            columnspan=2,
            pady=10,
        )

        ttk.Button(
            button_frame, text="Reset All to Default", command=self.reset_all_parameters
        ).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(
            button_frame, text="Stop Processing", command=self.stop_processing
        ).pack(side=tk.LEFT)

        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.grid(
            row=len(param_configs) + len(contour_configs) + 3,
            column=0,
            columnspan=2,
            sticky=(tk.W, tk.E),
            pady=5,
        )

        # Status label
        self.status_label = ttk.Label(
            control_frame, text="Ready - Select an image to begin"
        )
        self.status_label.grid(
            row=len(param_configs) + len(contour_configs) + 4,
            column=0,
            columnspan=2,
            pady=5,
        )

        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Result", padding="5")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        # Matplotlib figure
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)
        )

        # Initial empty plot
        self.ax.text(
            0.5,
            0.5,
            "Please select an image to begin",
            ha="center",
            va="center",
            transform=self.ax.transAxes,
            fontsize=16,
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def on_parameter_click(self, event):
        """Called when user starts dragging a parameter slider"""
        self.status_label.config(text="Adjusting parameter...")

    def on_parameter_release(self, event):
        """Called when user releases a parameter slider"""
        self.schedule_auto_update()

    def on_contour_click(self, event):
        """Called when user starts dragging a contour slider"""
        self.status_label.config(text="Adjusting contour...")

    def on_contour_release(self, event):
        """Called when user releases a contour slider"""
        if self.img is not None:
            # Immediately update the red contour line
            self.init_contour = self.create_initial_contour()
            self.plot_initial_contour_only()
        self.schedule_auto_update()

    def schedule_auto_update(self):
        """Schedule automatic snake update with delay"""
        # Cancel previous timer if it exists
        if self.auto_update_timer:
            self.root.after_cancel(self.auto_update_timer)

        # Schedule new update
        if self.img is not None and not self.processing:
            self.auto_update_timer = self.root.after(
                self.update_delay, self.auto_run_snake
            )
            self.status_label.config(text="Will update snake in 0.2s...")

    def auto_run_snake(self):
        """Automatically run snake algorithm"""
        if self.img is None or self.processing:
            return

        # Start processing in separate thread
        thread = threading.Thread(target=self.process_snake)
        thread.daemon = True
        thread.start()

    def stop_processing(self):
        """Stop current processing"""
        if self.auto_update_timer:
            self.root.after_cancel(self.auto_update_timer)
            self.auto_update_timer = None

        self.status_label.config(text="Processing stopped")

    def reset_all_parameters(self):
        """Reset all parameters and contour to default values"""
        # Reset all sliders to their default values
        for param_key, slider in self.params.items():
            default_value = self.param_defaults[param_key]
            slider.set(default_value)

        # Reset contour if image is loaded
        if self.img is not None:
            self.reset_contour()
            # Automatically update with default parameters
            self.schedule_auto_update()

        self.status_label.config(text="All parameters reset to default values")

    def load_image(self):
        """Load image using file dialog"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            try:
                # Load and process image
                img = cv2.imread(file_path)
                if img is None:
                    messagebox.showerror("Error", "Could not load image file")
                    return

                if img.ndim == 3:
                    self.img = rgb2gray(img)
                else:
                    self.img = img

                # Update file label
                filename = os.path.basename(file_path)
                self.file_label.config(text=f"Loaded: {filename}")

                # Reset and initialize contour
                self.reset_contour()
                self.status_label.config(
                    text=f"Image loaded: {self.img.shape} - Drag sliders and release to update"
                )

                # Automatically run snake with current parameters
                self.schedule_auto_update()

            except Exception as e:
                messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def create_initial_contour(self):
        """Create initial elliptical contour"""
        if self.img is None:
            return None

        angles = np.linspace(0, 2 * np.pi, 400)
        center_row = self.params["center_row"].get()
        center_col = self.params["center_col"].get()
        radius_row = self.params["radius_row"].get()
        radius_col = self.params["radius_col"].get()

        row_coords = center_row + radius_row * np.sin(angles)
        col_coords = center_col + radius_col * np.cos(angles)

        return np.array([row_coords, col_coords]).T

    def reset_contour(self):
        """Reset to initial contour"""
        if self.img is None:
            return

        self.init_contour = self.create_initial_contour()
        self.current_snake = None
        self.plot_results()

    def plot_initial_contour_only(self):
        """Quick update of just the initial contour (red line)"""
        if self.img is None or self.init_contour is None:
            return

        self.ax.clear()
        self.ax.imshow(self.img, cmap="gray")

        # Plot initial contour
        self.ax.plot(
            self.init_contour[:, 1],
            self.init_contour[:, 0],
            "r--",
            linewidth=2,
            label="Initial contour",
        )

        # Plot previous snake if it exists
        if self.current_snake is not None:
            self.ax.plot(
                self.current_snake[:, 1],
                self.current_snake[:, 0],
                "b-",
                linewidth=2,
                label="Active contour",
            )
            self.ax.legend()

        self.ax.set_title("Active Contour Result")
        self.ax.axis("off")
        self.canvas.draw()

    def process_snake(self):
        """Process active contour algorithm"""
        self.root.after(0, self.start_processing)

        try:
            # Get parameters
            alpha = self.params["alpha"].get()
            beta = self.params["beta"].get()
            sigma = self.params["sigma"].get()
            w_edge = self.params["w_edge"].get()
            w_line = self.params["w_line"].get()
            gamma = self.params["gamma"].get()
            max_iter = int(self.params["max_iter"].get())

            # Create initial contour
            init_contour = self.create_initial_contour()

            # Run active contour
            self.current_snake = active_contour(
                gaussian(self.img, sigma=sigma, preserve_range=False),
                init_contour,
                alpha=alpha,
                beta=beta,
                w_line=w_line,
                w_edge=w_edge,
                gamma=gamma,
                max_num_iter=max_iter,
            )

            # Update UI
            self.root.after(0, self.finish_processing)

        except Exception as e:
            self.root.after(0, lambda: self.error_processing(str(e)))

    def start_processing(self):
        """Update UI for processing start"""
        self.processing = True
        self.progress.start()
        self.status_label.config(text="Running active contour algorithm...")

    def finish_processing(self):
        """Update UI for processing completion"""
        self.processing = False
        self.progress.stop()
        self.status_label.config(
            text="Snake updated - Drag and release sliders to adjust"
        )
        self.plot_results()

    def error_processing(self, error_msg):
        """Handle processing error"""
        self.processing = False
        self.progress.stop()
        self.status_label.config(text=f"Error: {error_msg}")
        messagebox.showerror(
            "Processing Error", f"Error running active contour: {error_msg}"
        )

    def plot_results(self):
        """Plot image with contours"""
        if self.img is None:
            return

        self.ax.clear()
        self.ax.imshow(self.img, cmap="gray")

        # Plot initial contour
        if self.init_contour is not None:
            self.ax.plot(
                self.init_contour[:, 1],
                self.init_contour[:, 0],
                "r--",
                linewidth=2,
                label="Initial contour",
            )

        # Plot final snake
        if self.current_snake is not None:
            self.ax.plot(
                self.current_snake[:, 1],
                self.current_snake[:, 0],
                "b-",
                linewidth=2,
                label="Active contour",
            )

        self.ax.set_title("Active Contour Result")
        self.ax.legend()
        self.ax.axis("off")
        self.canvas.draw()


def main():
    root = tk.Tk()
    app = ActiveContourGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
