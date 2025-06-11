import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import cv2
import threading
import os


class KMeansSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Image Segmentation (RGB)")
        self.root.geometry("1600x1000")  # Larger window for more plots

        # Variables
        self.img = None
        self.X = None  # Reshaped image data
        self.labels = None
        self.clusters = None
        self.seg_img = None
        self.processing = False
        self.elbow_data = None
        self.optimal_k = None
        self.all_method_data = None  # Store all method results

        # Default parameters
        self.params = {
            "n_clusters": 3,
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
        ttk.Label(
            control_frame, text="K-Means Parameters:", font=("Arial", 12, "bold")
        ).pack(pady=(20, 10))

        # Number of clusters
        ttk.Label(control_frame, text="Number of Clusters (K):").pack(anchor=tk.W)
        self.clusters_var = tk.IntVar(value=self.params["n_clusters"])
        clusters_scale = tk.Scale(
            control_frame,
            from_=2,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.clusters_var,
            command=self.on_param_change,
            resolution=1,
        )
        clusters_scale.pack(fill=tk.X, pady=(0, 20))

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Reset", command=self.reset_parameters).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        self.elbow_button = ttk.Button(
            button_frame, text="Analyze Methods", command=self.calculate_elbow
        )
        self.elbow_button.pack(side=tk.RIGHT)

        # Optimal K suggestion
        self.optimal_frame = ttk.Frame(control_frame)
        self.optimal_frame.pack(fill=tk.X, pady=10)

        self.optimal_label = ttk.Label(
            self.optimal_frame, text="", font=("Arial", 12, "bold"), foreground="green"
        )
        self.optimal_label.pack(anchor=tk.W)

        self.use_optimal_button = ttk.Button(
            self.optimal_frame,
            text="Use Optimal K",
            command=self.use_optimal_k,
            state="disabled",
        )
        self.use_optimal_button.pack(anchor=tk.W, pady=(5, 0))

        # Method breakdown
        ttk.Label(
            control_frame, text="Method Results:", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(10, 5))
        self.method_info = tk.Text(control_frame, height=6, width=35)
        self.method_info.pack(fill=tk.X, pady=(0, 10))

        # Cluster information
        ttk.Label(
            control_frame, text="Cluster Centers (RGB):", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W, pady=(10, 5))
        self.cluster_info = tk.Text(control_frame, height=8, width=35)
        self.cluster_info.pack(fill=tk.X, pady=(0, 10))

        # Status
        self.status_var = tk.StringVar(value="Load an image to start")
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=280).pack(
            pady=10
        )

        # Right panel for plots
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create matplotlib figure - larger for more subplots
        self.fig = Figure(figsize=(16, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
        )

        if file_path:
            try:
                # Load image using matplotlib (like in your notebook)
                self.img = plt.imread(file_path)

                # Handle different image formats
                if self.img.dtype == np.float32 or self.img.dtype == np.float64:
                    self.img = (self.img * 255).astype(np.uint8)

                # Resize if too large for performance
                if self.img.shape[0] > 400 or self.img.shape[1] > 400:
                    scale = min(400 / self.img.shape[0], 400 / self.img.shape[1])
                    new_height = int(self.img.shape[0] * scale)
                    new_width = int(self.img.shape[1] * scale)
                    self.img = cv2.resize(self.img, (new_width, new_height))

                # Reshape image to (-1, 3) like in your notebook
                self.X = np.reshape(self.img, (-1, 3))

                # Reset all data
                self.elbow_data = None
                self.all_method_data = None
                self.optimal_k = None
                self.optimal_label.config(text="")
                self.use_optimal_button.config(state="disabled")

                self.status_var.set(
                    f"Loaded: {os.path.basename(file_path)} ({self.img.shape[1]}x{self.img.shape[0]})"
                )

                self.process_kmeans()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def on_param_change(self, event=None):
        if self.img is not None and not self.processing:
            # Update parameters
            self.params["n_clusters"] = self.clusters_var.get()

            # Process in thread to avoid blocking UI
            threading.Thread(target=self.process_kmeans, daemon=True).start()

    def calculate_elbow(self):
        if self.img is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        # Disable button and start calculation
        self.elbow_button.config(state="disabled")
        threading.Thread(target=self._calculate_all_methods, daemon=True).start()

    def _calculate_all_methods(self):
        """Calculate all optimization methods"""
        try:
            self.status_var.set("Analyzing optimal K using multiple methods...")

            # Use sample of data if too large
            X_sample = self.X
            if len(self.X) > 10000:  # Smaller sample for multiple methods
                np.random.seed(42)
                indices = np.random.choice(len(self.X), 10000, replace=False)
                X_sample = self.X[indices]
                self.status_var.set("Using 10,000 samples for analysis...")

            # Initialize results storage
            results = {
                'K': list(range(1, 9)),  # K values 1-8
                'inertias': [],
                'silhouette_scores': [],
                'calinski_harabasz_scores': [],
                'davies_bouldin_scores': [],
                'percentage_improvements': []
            }

            # Calculate metrics for each K
            for i, k in enumerate(results['K']):
                self.status_var.set(f"Calculating metrics for K={k}...")
                
                # Fit KMeans
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X_sample)
                labels = kmeans.labels_
                
                # Inertia (for elbow method)
                results['inertias'].append(kmeans.inertia_)
                
                # Silhouette score (only for k >= 2)
                if k >= 2:
                    try:
                        sil_score = silhouette_score(X_sample, labels)
                        results['silhouette_scores'].append(sil_score)
                    except:
                        results['silhouette_scores'].append(0)
                else:
                    results['silhouette_scores'].append(0)  # Undefined for k=1
                
                # Calinski-Harabasz score (only for k >= 2)
                if k >= 2:
                    try:
                        ch_score = calinski_harabasz_score(X_sample, labels)
                        results['calinski_harabasz_scores'].append(ch_score)
                    except:
                        results['calinski_harabasz_scores'].append(0)
                else:
                    results['calinski_harabasz_scores'].append(0)
                
                # Davies-Bouldin score (only for k >= 2)
                if k >= 2:
                    try:
                        db_score = davies_bouldin_score(X_sample, labels)
                        results['davies_bouldin_scores'].append(db_score)
                    except:
                        results['davies_bouldin_scores'].append(float('inf'))
                else:
                    results['davies_bouldin_scores'].append(float('inf'))

            # Calculate percentage improvements
            for i in range(1, len(results['inertias'])):
                if results['inertias'][i-1] > 0:
                    improvement = (results['inertias'][i-1] - results['inertias'][i]) / results['inertias'][i-1] * 100
                    results['percentage_improvements'].append(improvement)
                else:
                    results['percentage_improvements'].append(0)

            # Find optimal K using each method
            optimal_ks = {}
            
            # Elbow method
            if len(results['inertias']) >= 3:
                first_deriv = np.diff(results['inertias'])
                second_deriv = np.diff(first_deriv)
                elbow_k = np.argmax(np.abs(second_deriv)) + 2  # +2 because of double diff
                optimal_ks['Elbow'] = min(max(elbow_k, 2), 8)
            
            # Silhouette method (highest score)
            valid_sil_scores = [score for score in results['silhouette_scores'] if score > 0]
            if valid_sil_scores:
                sil_k = results['silhouette_scores'].index(max(valid_sil_scores)) + 1
                optimal_ks['Silhouette'] = sil_k
            
            # Calinski-Harabasz method (highest score)
            valid_ch_scores = [score for score in results['calinski_harabasz_scores'] if score > 0]
            if valid_ch_scores:
                ch_k = results['calinski_harabasz_scores'].index(max(valid_ch_scores)) + 1
                optimal_ks['Calinski-Harabasz'] = ch_k
            
            # Davies-Bouldin method (lowest score)
            valid_db_scores = [score for score in results['davies_bouldin_scores'] if score != float('inf')]
            if valid_db_scores:
                db_k = results['davies_bouldin_scores'].index(min(valid_db_scores)) + 1
                optimal_ks['Davies-Bouldin'] = db_k
            
            # Percentage improvement method
            for i, imp in enumerate(results['percentage_improvements']):
                if imp < 15 and i >= 1:  # At least K=3
                    optimal_ks['Percentage'] = i + 2
                    break
            else:
                optimal_ks['Percentage'] = 3

            # Consensus optimal K
            if optimal_ks:
                from collections import Counter
                k_counts = Counter(optimal_ks.values())
                self.optimal_k = k_counts.most_common(1)[0][0]
            else:
                self.optimal_k = 3

            # Store all results
            self.all_method_data = {
                'results': results,
                'optimal_ks': optimal_ks,
                'sample_size': len(X_sample)
            }

            # Also store elbow data for backward compatibility
            self.elbow_data = {
                "K": results['K'],
                "inertias": results['inertias'],
                "sample_size": len(X_sample),
                "optimal_k": self.optimal_k
            }

            # Update UI
            self.root.after(0, self.update_optimal_k_display)
            self.root.after(0, self.update_method_info)
            self.root.after(0, self.update_display)
            self.root.after(0, lambda: self.status_var.set(f"Analysis complete! Consensus optimal K: {self.optimal_k}"))
            self.root.after(0, lambda: self.elbow_button.config(state="normal"))

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.root.after(0, lambda: self.status_var.set(error_msg))
            self.root.after(0, lambda: self.elbow_button.config(state="normal"))

    def update_optimal_k_display(self):
        """Update the optimal K suggestion display"""
        if self.optimal_k:
            self.optimal_label.config(text=f"📊 Consensus Optimal K: {self.optimal_k}")
            self.use_optimal_button.config(state="normal")

    def update_method_info(self):
        """Update method breakdown information"""
        if not self.all_method_data:
            return
            
        info_text = "Method Results:\n\n"
        
        for method, k_val in self.all_method_data['optimal_ks'].items():
            info_text += f"{method}: K={k_val}\n"
        
        info_text += f"\nConsensus: K={self.optimal_k}\n"
        info_text += f"Sample size: {self.all_method_data['sample_size']}"
        
        def update_text():
            self.method_info.delete(1.0, tk.END)
            self.method_info.insert(1.0, info_text)
        
        self.root.after(0, update_text)

    def use_optimal_k(self):
        """Set the K value to the optimal K found by methods"""
        if self.optimal_k:
            self.clusters_var.set(self.optimal_k)
            self.params["n_clusters"] = self.optimal_k
            threading.Thread(target=self.process_kmeans, daemon=True).start()

    def reset_parameters(self):
        self.params = {"n_clusters": 3}
        self.clusters_var.set(self.params["n_clusters"])

        # Reset all data
        self.elbow_data = None
        self.all_method_data = None
        self.optimal_k = None
        self.optimal_label.config(text="")
        self.use_optimal_button.config(state="disabled")

        if self.img is not None:
            self.process_kmeans()

    def process_kmeans(self):
        if self.img is None:
            return

        self.processing = True
        self.status_var.set("Processing K-means clustering...")

        try:
            # Apply K-means
            Kmeans = KMeans(n_clusters=self.params["n_clusters"])
            Kmeans.fit(self.X)

            # Get clusters and labels
            self.clusters = Kmeans.cluster_centers_.astype(np.uint8)
            self.labels = Kmeans.labels_

            # Create segmented image
            self.seg_img = self.clusters[self.labels].reshape(self.img.shape)

            # Update cluster information
            self.update_cluster_info()

            # Update display
            self.root.after(0, self.update_display)

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: self.set_error_status(msg))
        finally:
            self.processing = False

    def set_error_status(self, error_msg):
        self.status_var.set(f"Error: {error_msg}")

    def update_cluster_info(self):
        """Update cluster information display"""
        if self.clusters is None:
            return

        info_text = f"K = {self.params['n_clusters']} clusters\n"

        # Show optimal K info
        if self.optimal_k:
            if self.params["n_clusters"] == self.optimal_k:
                info_text += "✅ Using optimal K!\n\n"
            else:
                info_text += f"💡 Optimal K suggested: {self.optimal_k}\n\n"
        else:
            info_text += "\n"

        for i, cluster in enumerate(self.clusters):
            cluster_size = np.sum(self.labels == i)
            cluster_percentage = (cluster_size / len(self.labels)) * 100

            info_text += f"Cluster {i}:\n"
            info_text += f"  RGB: ({cluster[0]}, {cluster[1]}, {cluster[2]})\n"
            info_text += (
                f"  Size: {cluster_size} pixels ({cluster_percentage:.1f}%)\n\n"
            )

        # Update text widget
        def update_text():
            self.cluster_info.delete(1.0, tk.END)
            self.cluster_info.insert(1.0, info_text)

        self.root.after(0, update_text)

    def update_display(self):
        self.fig.clear()

        if self.img is None:
            return

        # Determine layout based on available data
        if self.all_method_data is not None:
            # Show all method plots (3x3 grid)
            ax1 = self.fig.add_subplot(3, 3, 1)  # Original image
            ax2 = self.fig.add_subplot(3, 3, 2)  # Segmented image
            ax3 = self.fig.add_subplot(3, 3, 3)  # Elbow method
            ax4 = self.fig.add_subplot(3, 3, 4)  # Silhouette
            ax5 = self.fig.add_subplot(3, 3, 5)  # Calinski-Harabasz
            ax6 = self.fig.add_subplot(3, 3, 6)  # Davies-Bouldin
            ax7 = self.fig.add_subplot(3, 3, 7)  # Percentage improvement
            ax8 = self.fig.add_subplot(3, 3, 8)  # Method comparison
            ax9 = self.fig.add_subplot(3, 3, 9)  # Consensus

            results = self.all_method_data['results']
            optimal_ks = self.all_method_data['optimal_ks']
            current_k = self.params["n_clusters"]

            # Original image
            ax1.imshow(self.img)
            ax1.set_title("Original Image")
            ax1.axis("off")

            # Segmented image
            if self.seg_img is not None:
                ax2.imshow(self.seg_img)
                title = f"K-Means (K={current_k})"
                if self.optimal_k and current_k == self.optimal_k:
                    title += " ✅"
                ax2.set_title(title)
                ax2.axis("off")

            # Elbow method
            ax3.plot(results['K'], results['inertias'], 'bo-', linewidth=2, markersize=6)
            if 'Elbow' in optimal_ks:
                elbow_k = optimal_ks['Elbow']
                elbow_inertia = results['inertias'][elbow_k - 1]
                ax3.plot(elbow_k, elbow_inertia, 'go', markersize=12, label=f'Optimal K={elbow_k}')
            ax3.plot(current_k, results['inertias'][current_k - 1], 'ro', markersize=10, label=f'Current K={current_k}')
            ax3.set_title('Elbow Method')
            ax3.set_xlabel('K')
            ax3.set_ylabel('Inertia')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

            # Silhouette analysis
            K_sil = [k for k, score in zip(results['K'], results['silhouette_scores']) if score > 0]
            sil_scores = [score for score in results['silhouette_scores'] if score > 0]
            if sil_scores:
                ax4.plot(K_sil, sil_scores, 'go-', linewidth=2, markersize=6)
                if 'Silhouette' in optimal_ks:
                    sil_k = optimal_ks['Silhouette']
                    if sil_k in K_sil:
                        sil_score = results['silhouette_scores'][sil_k - 1]
                        ax4.plot(sil_k, sil_score, 'go', markersize=12, label=f'Optimal K={sil_k}')
                if current_k in K_sil:
                    current_sil = results['silhouette_scores'][current_k - 1]
                    ax4.plot(current_k, current_sil, 'ro', markersize=10, label=f'Current K={current_k}')
                ax4.legend()
            ax4.set_title('Silhouette Analysis')
            ax4.set_xlabel('K')
            ax4.set_ylabel('Silhouette Score')
            ax4.grid(True, alpha=0.3)

            # Calinski-Harabasz
            K_ch = [k for k, score in zip(results['K'], results['calinski_harabasz_scores']) if score > 0]
            ch_scores = [score for score in results['calinski_harabasz_scores'] if score > 0]
            if ch_scores:
                ax5.plot(K_ch, ch_scores, 'mo-', linewidth=2, markersize=6)
                if 'Calinski-Harabasz' in optimal_ks:
                    ch_k = optimal_ks['Calinski-Harabasz']
                    if ch_k in K_ch:
                        ch_score = results['calinski_harabasz_scores'][ch_k - 1]
                        ax5.plot(ch_k, ch_score, 'go', markersize=12, label=f'Optimal K={ch_k}')
                if current_k in K_ch:
                    current_ch = results['calinski_harabasz_scores'][current_k - 1]
                    ax5.plot(current_k, current_ch, 'ro', markersize=10, label=f'Current K={current_k}')
                ax5.legend()
            ax5.set_title('Calinski-Harabasz Index')
            ax5.set_xlabel('K')
            ax5.set_ylabel('CH Score')
            ax5.grid(True, alpha=0.3)

            # Davies-Bouldin
            K_db = [k for k, score in zip(results['K'], results['davies_bouldin_scores']) if score != float('inf')]
            db_scores = [score for score in results['davies_bouldin_scores'] if score != float('inf')]
            if db_scores:
                ax6.plot(K_db, db_scores, 'co-', linewidth=2, markersize=6)
                if 'Davies-Bouldin' in optimal_ks:
                    db_k = optimal_ks['Davies-Bouldin']
                    if db_k in K_db:
                        db_score = results['davies_bouldin_scores'][db_k - 1]
                        ax6.plot(db_k, db_score, 'go', markersize=12, label=f'Optimal K={db_k}')
                if current_k in K_db:
                    current_db = results['davies_bouldin_scores'][current_k - 1]
                    ax6.plot(current_k, current_db, 'ro', markersize=10, label=f'Current K={current_k}')
                ax6.legend()
            ax6.set_title('Davies-Bouldin Index')
            ax6.set_xlabel('K')
            ax6.set_ylabel('DB Score (lower is better)')
            ax6.grid(True, alpha=0.3)

            # Percentage improvement
            if results['percentage_improvements']:
                K_pct = list(range(2, len(results['percentage_improvements']) + 2))
                ax7.plot(K_pct, results['percentage_improvements'], 'yo-', linewidth=2, markersize=6)
                ax7.axhline(y=15, color='r', linestyle='--', alpha=0.7, label='15% threshold')
                if 'Percentage' in optimal_ks:
                    pct_k = optimal_ks['Percentage']
                    if pct_k in K_pct:
                        pct_idx = pct_k - 2
                        if pct_idx < len(results['percentage_improvements']):
                            pct_val = results['percentage_improvements'][pct_idx]
                            ax7.plot(pct_k, pct_val, 'go', markersize=12, label=f'Optimal K={pct_k}')
                ax7.legend()
            ax7.set_title('Percentage Improvement')
            ax7.set_xlabel('K')
            ax7.set_ylabel('% Improvement')
            ax7.grid(True, alpha=0.3)

            # Method comparison bar chart
            methods = list(optimal_ks.keys())
            k_values = list(optimal_ks.values())
            colors = ['blue', 'green', 'purple', 'cyan', 'orange']
            bars = ax8.bar(methods, k_values, color=colors[:len(methods)])
            ax8.set_title('Method Comparison')
            ax8.set_xlabel('Method')
            ax8.set_ylabel('Optimal K')
            ax8.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, val in zip(bars, k_values):
                ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(val), ha='center', va='bottom')

            # Consensus result
            ax9.text(0.5, 0.6, f'🎯 CONSENSUS\nOptimal K: {self.optimal_k}', 
                    ha='center', va='center', transform=ax9.transAxes,
                    fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            
            methods_agree = len(set(optimal_ks.values()))
            confidence = "High" if methods_agree <= 2 else "Medium" if methods_agree <= 3 else "Low"
            ax9.text(0.5, 0.3, f'Confidence: {confidence}', 
                    ha='center', va='center', transform=ax9.transAxes, fontsize=12)
            ax9.set_title('Final Recommendation')
            ax9.axis('off')

        elif self.elbow_data is not None:
            # Show simple elbow plot (original layout)
            ax1 = self.fig.add_subplot(2, 3, 1)
            ax2 = self.fig.add_subplot(2, 3, 2)
            ax3 = self.fig.add_subplot(2, 3, 3)

            # Original image
            ax1.imshow(self.img)
            ax1.set_title("Original Image")
            ax1.axis("off")

            # Segmented image
            if self.seg_img is not None:
                ax2.imshow(self.seg_img)
                current_k = self.params["n_clusters"]
                title = f"K-Means Segmentation (K={current_k})"
                if self.optimal_k and current_k == self.optimal_k:
                    title += " ✅"
                ax2.set_title(title)
                ax2.axis("off")

                # Elbow plot
                ax3.plot(self.elbow_data["K"], self.elbow_data["inertias"], "bx-", linewidth=2, markersize=8)
                ax3.set_xlabel("k")
                ax3.set_ylabel("Inertia")
                ax3.set_title("Elbow Method For Optimal k")
                ax3.grid(True, alpha=0.3)

                # Mark optimal K
                if self.optimal_k and self.optimal_k <= len(self.elbow_data["inertias"]):
                    optimal_inertia = self.elbow_data["inertias"][self.optimal_k - 1]
                    ax3.plot(self.optimal_k, optimal_inertia, "go", markersize=15,
                            label=f"Optimal K={self.optimal_k}", markerfacecolor="lightgreen",
                            markeredgecolor="green", markeredgewidth=2)

                # Mark current K
                current_k = self.params["n_clusters"]
                if current_k <= len(self.elbow_data["inertias"]):
                    current_inertia = self.elbow_data["inertias"][current_k - 1]
                    color = "green" if current_k == self.optimal_k else "red"
                    ax3.plot(current_k, current_inertia, "o", color=color, markersize=12,
                            label=f"Current K={current_k}")

                ax3.legend()
        else:
            # No analysis data, show basic layout
            ax1 = self.fig.add_subplot(1, 2, 1)
            ax2 = self.fig.add_subplot(1, 2, 2)

            ax1.imshow(self.img)
            ax1.set_title("Original Image")
            ax1.axis("off")

            if self.seg_img is not None:
                ax2.imshow(self.seg_img)
                ax2.set_title(f'K-Means Segmentation (K={self.params["n_clusters"]})')
                ax2.axis("off")

        self.fig.tight_layout()
        self.canvas.draw()

        # Update status
        if self.seg_img is not None:
            status_text = f"K-means complete: {self.params['n_clusters']} clusters"
            if self.optimal_k:
                if self.params["n_clusters"] == self.optimal_k:
                    status_text += " ✅ (Optimal)"
                else:
                    status_text += f" (Optimal K: {self.optimal_k})"
            self.status_var.set(status_text)


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = KMeansSegmentationGUI(root)
    root.mainloop()