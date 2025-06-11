import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import threading
import os

class MeanShiftSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mean Shift Image Segmentation")
        self.root.geometry("1300x800")
        
        # Variables
        self.image = None
        self.lab_image = None
        self.labels = None
        self.colored_segmented_image = None
        self.processing = False
        self.estimated_bandwidth = None
        
        # Parameters
        self.auto_bandwidth = tk.BooleanVar(value=True)
        self.use_spatial = tk.BooleanVar(value=True)
        self.bandwidth_var = tk.DoubleVar(value=50.0)
        self.quantile_var = tk.DoubleVar(value=0.2)
        self.n_samples_var = tk.IntVar(value=500)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # Load button
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(pady=10)
        
        # Parameters section
        ttk.Label(control_frame, text="Mean Shift Parameters:", font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        
        # Auto bandwidth checkbox
        ttk.Checkbutton(control_frame, text="Auto Estimate Bandwidth", 
                       variable=self.auto_bandwidth,
                       command=self.on_auto_change).pack(anchor=tk.W, pady=5)
        
        # Manual bandwidth
        ttk.Label(control_frame, text="Manual Bandwidth:").pack(anchor=tk.W)
        self.bandwidth_scale = tk.Scale(
            control_frame, 
            from_=10, 
            to=200, 
            orient=tk.HORIZONTAL,
            variable=self.bandwidth_var, 
            command=self.on_param_change,
            resolution=5
        )
        self.bandwidth_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Auto bandwidth parameters
        ttk.Label(control_frame, text="Auto Bandwidth Parameters:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Quantile
        ttk.Label(control_frame, text="Quantile (sensitivity):").pack(anchor=tk.W)
        quantile_scale = tk.Scale(
            control_frame, 
            from_=0.05, 
            to=0.5, 
            orient=tk.HORIZONTAL,
            variable=self.quantile_var, 
            command=self.on_param_change,
            resolution=0.01
        )
        quantile_scale.pack(fill=tk.X, pady=(0, 10))
        
        # N samples
        ttk.Label(control_frame, text="N Samples (for estimation):").pack(anchor=tk.W)
        n_samples_scale = tk.Scale(
            control_frame, 
            from_=100, 
            to=2000, 
            orient=tk.HORIZONTAL,
            variable=self.n_samples_var, 
            command=self.on_param_change,
            resolution=50
        )
        n_samples_scale.pack(fill=tk.X, pady=(0, 10))
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Feature space parameters
        ttk.Label(control_frame, text="Feature Space:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        # Spatial coordinates
        ttk.Checkbutton(control_frame, text="Include Spatial Coordinates (X,Y)", 
                       variable=self.use_spatial,
                       command=self.on_param_change).pack(anchor=tk.W, pady=5)
        
        # Process button
        ttk.Button(control_frame, text="Process Mean Shift", command=self.process_meanshift).pack(pady=20)
        
        # Bandwidth info display
        self.bandwidth_info_frame = ttk.Frame(control_frame)
        self.bandwidth_info_frame.pack(fill=tk.X, pady=10)
        
        self.bandwidth_label = ttk.Label(self.bandwidth_info_frame, text="", font=('Arial', 12, 'bold'), foreground='green')
        self.bandwidth_label.pack(anchor=tk.W)
        
        # Results display
        ttk.Label(control_frame, text="Results:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(20, 5))
        self.info_text = tk.Text(control_frame, height=10, width=35)
        self.info_text.pack(fill=tk.X, pady=(0, 10))
        
        # Status
        self.status_var = tk.StringVar(value="Load an image to start")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=280).pack(pady=10)
        
        # Right panel for plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update initial state
        self.update_bandwidth_state()
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load image exactly like reference
                self.image = cv2.imread(file_path)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                
                # Resize for performance
                if self.image.shape[0] > 300 or self.image.shape[1] > 300:
                    scale = min(300 / self.image.shape[0], 300 / self.image.shape[1])
                    new_height = int(self.image.shape[0] * scale)
                    new_width = int(self.image.shape[1] * scale)
                    self.image = cv2.resize(self.image, (new_width, new_height))
                
                # Convert to LAB exactly like reference
                self.lab_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2LAB)
                
                # Reset results
                self.labels = None
                self.colored_segmented_image = None
                self.estimated_bandwidth = None
                self.bandwidth_label.config(text="")
                
                self.status_var.set(f"Loaded: {os.path.basename(file_path)} ({self.image.shape[1]}x{self.image.shape[0]})")
                self.update_display()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def on_auto_change(self):
        self.update_bandwidth_state()
        if self.image is not None and not self.processing:
            threading.Thread(target=self.process_meanshift, daemon=True).start()
        
    def update_bandwidth_state(self):
        """Enable/disable manual bandwidth based on auto setting"""
        if self.auto_bandwidth.get():
            self.bandwidth_scale.config(state='disabled')
        else:
            self.bandwidth_scale.config(state='normal')
            
    def on_param_change(self, event=None):
        """Auto process when parameters change"""
        if self.image is not None and not self.processing:
            threading.Thread(target=self.process_meanshift, daemon=True).start()
    
    def process_meanshift(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
            
        self.processing = True
        self.status_var.set("Processing Mean Shift clustering...")
        
        try:
            # Prepare feature space exactly like reference
            flat_image = self.lab_image.reshape((-1, 3))
            
            if self.use_spatial.get():
                # Create feature space [L, a, b, x, y] exactly like reference
                height, width, _ = self.image.shape
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                features = np.column_stack([flat_image, x.flatten(), y.flatten()])
                feature_description = "LAB + Spatial (X,Y)"
            else:
                # Use only LAB color [L, a, b]
                features = flat_image
                feature_description = "LAB Color Only"
            
            # Get bandwidth
            if self.auto_bandwidth.get():
                # Estimate bandwidth with user parameters
                self.status_var.set("Estimating bandwidth...")
                bandwidth = estimate_bandwidth(
                    features, 
                    quantile=self.quantile_var.get(), 
                    n_samples=self.n_samples_var.get()
                )
                self.estimated_bandwidth = bandwidth
                bandwidth_source = f"Auto (quantile={self.quantile_var.get():.2f}, n_samples={self.n_samples_var.get()})"
            else:
                # Use manual bandwidth
                bandwidth = self.bandwidth_var.get()
                bandwidth_source = "Manual"
            
            # Update bandwidth display
            self.root.after(0, lambda: self.bandwidth_label.config(text=f"Bandwidth: {bandwidth:.2f} ({bandwidth_source})"))
            
            # Perform Mean Shift clustering exactly like reference
            self.status_var.set("Running Mean Shift clustering...")
            mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            mean_shift.fit(features)
            self.labels = mean_shift.labels_
            
            # Create colored segmented image exactly like reference
            height, width = self.image.shape[:2]
            segmented_image = self.labels.reshape((height, width))
            
            unique_labels = np.unique(self.labels)
            np.random.seed(42)  # Consistent colors
            segmented_colors = np.random.randint(0, 255, size=(len(unique_labels), 3))
            self.colored_segmented_image = segmented_colors[segmented_image]
            
            # Update info display
            self.update_info(len(unique_labels), bandwidth, bandwidth_source, feature_description)
            
            # Update display
            self.root.after(0, self.update_display)
            self.root.after(0, lambda: self.status_var.set(f"Complete: {len(unique_labels)} clusters found"))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self.status_var.set(f"Error: {error_msg}"))
        finally:
            self.processing = False
    
    def update_info(self, num_clusters, bandwidth, bandwidth_source, feature_description):
        """Update the results information display"""
        info = f"Mean Shift Results:\n\n"
        info += f"Clusters found: {num_clusters}\n"
        info += f"Bandwidth: {bandwidth:.2f} ({bandwidth_source})\n"
        info += f"Features: {feature_description}\n"
        
        if self.auto_bandwidth.get():
            info += f"Quantile: {self.quantile_var.get():.2f}\n"
            info += f"N Samples: {self.n_samples_var.get()}\n"
        
        info += f"\nCluster Distribution:\n"
        
        # Calculate cluster sizes and percentages
        for i in range(num_clusters):
            size = np.sum(self.labels == i)
            percent = (size / len(self.labels)) * 100
            info += f"  Cluster {i}: {size:,} pixels ({percent:.1f}%)\n"
        
        # Add parameter recommendations
        info += f"\nParameter Notes:\n"
        if num_clusters > 20:
            info += "• Too many clusters? Try higher bandwidth/quantile\n"
        elif num_clusters < 3:
            info += "• Too few clusters? Try lower bandwidth/quantile\n"
        else:
            info += "• Good cluster count for segmentation\n"
        
        def update_text():
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
        
        self.root.after(0, update_text)
            
    def update_display(self):
        self.fig.clear()
        
        if self.image is None:
            return
            
        if self.colored_segmented_image is not None:
            # Show comprehensive results (2x2 grid)
            ax1 = self.fig.add_subplot(2, 2, 1)
            ax2 = self.fig.add_subplot(2, 2, 2)
            ax3 = self.fig.add_subplot(2, 2, 3)
            ax4 = self.fig.add_subplot(2, 2, 4)
            
            # Original image
            ax1.imshow(self.image)
            ax1.set_title('Original Image (RGB)')
            ax1.axis('off')
            
            # LAB L channel
            ax2.imshow(self.lab_image[:, :, 0], cmap='gray')
            ax2.set_title('LAB - L Channel (Lightness)')
            ax2.axis('off')
            
            # Mean Shift segmentation
            ax3.imshow(self.colored_segmented_image)
            unique_labels = np.unique(self.labels)
            ax3.set_title(f'Mean Shift Segmentation\n({len(unique_labels)} clusters)')
            ax3.axis('off')
            
            # Overlay on original
            ax4.imshow(self.image)
            ax4.imshow(self.colored_segmented_image, alpha=0.6)
            ax4.set_title('Segmentation Overlay')
            ax4.axis('off')
            
        else:
            # Show only original image
            ax = self.fig.add_subplot(1, 1, 1)
            ax.imshow(self.image)
            ax.set_title('Original Image - Load and Process')
            ax.axis('off')
        
        self.fig.tight_layout()
        self.canvas.draw()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = MeanShiftSegmentationGUI(root)
    root.mainloop()