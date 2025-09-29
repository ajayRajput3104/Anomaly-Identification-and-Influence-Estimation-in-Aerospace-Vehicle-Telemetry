"""
For Python 3:
bash
sudo apt update
sudo apt install python3-tk
For Python 2 (if still needed):
bash
sudo apt install python-tk
Method 2: Verify Installation
After installing, check if tkinter works by running:

bash
python3 -m tkinter
"""
import os
import io
import sys
import copy
import base64
import traceback
import tempfile
import datetime
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fpdf import FPDF
import plotly.graph_objs as go
import plotly.io as pio
import webbrowser
from tkhtmlview import HTMLLabel



class AeroSpaceVehicleAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AeroSpace Vehicle Data Analysis with Gradient Boosting")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.train_files = []
        self.test_file = ""
        self.train_data = None
        self.test_data = None
        self.original_train_data = None
        self.original_test_data = None
        self.features = []
        self.target = None
        self.scalers = {}
        
        # Create UI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Reset Button Frame (Top-Right) ===
        top_right_frame = ttk.Frame(self.root)
        top_right_frame.place(relx=1.0, y=5, anchor="ne")  # place in top-right corner
        ttk.Button(top_right_frame, text="🔁 Reset All", command=self.reset_all).pack(padx=5, pady=5)

        
        # Left panel (controls)
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Right panel (output/plots)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File selection
        file_frame = ttk.LabelFrame(left_panel, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Training Files:").pack(anchor=tk.W)
        self.train_files_entry = ttk.Entry(file_frame, width=30)
        self.train_files_entry.pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_train_files).pack(pady=2)
        
        ttk.Label(file_frame, text="Test File:").pack(anchor=tk.W, pady=(10, 0))
        self.test_file_entry = ttk.Entry(file_frame, width=30)
        self.test_file_entry.pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Browse...", command=self.browse_test_file).pack(pady=2)
        
        # Parameters
        param_frame = ttk.LabelFrame(left_panel, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Threshold:").pack(anchor=tk.W)
        self.threshold_entry = ttk.Entry(param_frame)
        self.threshold_entry.insert(0, "0.3")
        self.threshold_entry.pack(fill=tk.X, pady=2)

        # Feature column input (index + name)
        feature_info_frame = ttk.LabelFrame(param_frame, text="Feature Info", padding="10")
        feature_info_frame.pack(fill=tk.X, pady=5)

        ttk.Label(feature_info_frame, text="Column Index:").pack(anchor=tk.W)
        self.feature_index_entry = ttk.Entry(feature_info_frame)
        self.feature_index_entry.pack(fill=tk.X, pady=2)

        ttk.Label(feature_info_frame, text="Column Name:").pack(anchor=tk.W)
        self.feature_name_entry = ttk.Entry(feature_info_frame)
        self.feature_name_entry.pack(fill=tk.X, pady=2)

        ttk.Button(feature_info_frame, text="+ Add Feature", command=self.add_feature_info).pack(pady=2)
        
        # Target column input (index + name)
        target_frame = ttk.LabelFrame(param_frame, text="Target Column Info", padding="10")
        target_frame.pack(fill=tk.X, pady=5)

        ttk.Label(target_frame, text="Target Column Index:").pack(anchor=tk.W)
        self.target_index_entry = ttk.Entry(target_frame)
        self.target_index_entry.pack(fill=tk.X, pady=2)

        ttk.Label(target_frame, text="Target Column Name:").pack(anchor=tk.W)
        self.target_name_entry = ttk.Entry(target_frame)
        self.target_name_entry.pack(fill=tk.X, pady=2)

        # Summary output
        self.feature_summary = tk.Text(feature_info_frame, height=6, state=tk.DISABLED, wrap=tk.WORD)
        self.feature_summary.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        ttk.Button(feature_info_frame, text="✅ Final Submit", command=self.final_submit_feature_info).pack(pady=5)

        # Buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Train Model", command=self.train_model).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Analyze Outliers", command=self.analyze_outliers).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Feature Contribution", command=self.show_feature_contribution_pie).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Show Plots", command=self.show_plots).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Generate Report", command=self.generate_report).pack(fill=tk.X, pady=2)

        
        # Output area
        self.output_frame = ttk.LabelFrame(right_panel, text="Output", padding="10")
        self.output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Use a PanedWindow to split output frame
        paned = ttk.PanedWindow(self.output_frame, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # Text area (top)
        text_frame = ttk.Frame(paned)
        self.output_text = tk.Text(text_frame, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        paned.add(text_frame, weight=1)

        # Plot area (bottom)
        plot_frame = ttk.Frame(paned)
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        paned.add(plot_frame, weight=3)

        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
    
    def reset_all(self):
        """Reset all fields, logs, plots, and stored data."""
        try:
            # Clear input fields
            self.train_files_entry.delete(0, tk.END)
            self.test_file_entry.delete(0, tk.END)
            self.threshold_entry.delete(0, tk.END)
            self.threshold_entry.insert(0, "0.3")

            self.feature_index_entry.delete(0, tk.END)
            self.feature_name_entry.delete(0, tk.END)
            self.target_index_entry.delete(0, tk.END)
            self.target_name_entry.delete(0, tk.END)

            # Reset internal variables
            self.features = []
            self.feature_info = {}
            self.model = None
            self.scalers = {}
            self.contributions = {}
            self.avg_contributions = {}
            self.train_data = None
            self.test_data = None
            self.train_data_scaled = None
            self.test_data_scaled = None
            self.original_train_data = None
            self.original_test_data = None
            self.X_test = None
            self.y_test = None
            self.outliers = None
            self.outlier_groups = None
            self.target = None

            # Clear text outputs
            self.output_text.delete(1.0, tk.END)
            self.feature_summary.config(state=tk.NORMAL)
            self.feature_summary.delete(1.0, tk.END)
            self.feature_summary.config(state=tk.DISABLED)

            # Clear plot
            self.figure.clear()
            self.canvas.draw()

            # Update status and log
            self.status_var.set("🔄 Ready for new analysis.")
            self.log_message("🔁 All fields reset. Ready for new input.")

        except Exception as e:
            messagebox.showerror("Error", f"Reset failed: {str(e)}")


        
    def browse_train_files(self):
        files = filedialog.askopenfilenames(title="Select Training Files")
        if files:
            self.train_files = list(files)
            self.train_files_entry.delete(0, tk.END)
            self.train_files_entry.insert(0, ", ".join(files))
    
    def browse_test_file(self):
        file = filedialog.askopenfilename(title="Select Test File")
        if file:
            self.test_file = file
            self.test_file_entry.delete(0, tk.END)
            self.test_file_entry.insert(0, file)
    
    def load_data(self):
        if not self.train_files:
            messagebox.showerror("Error", "Please select training files")
            return
        
        if not self.test_file:
            messagebox.showerror("Error", "Please select a test file")
            return
        
        try:
            self.status_var.set("Loading training data...")
            self.root.update()
            
            # Load and combine data from multiple files
            dfs = []
            for file in self.train_files:
                try:
                    df = pd.read_csv(file.strip(), sep=r'\s+', header=None)
                    dfs.append(df)
                    self.log_message(f"Successfully loaded {file}")
                except Exception as e:
                    self.log_message(f"Error loading {file}: {str(e)}")
                    return
            
            self.train_data = pd.concat(dfs, ignore_index=True)
            
            self.status_var.set("Loading test data...")
            self.root.update()
            
            try:
                self.test_data = pd.read_csv(self.test_file, sep=r'\s+', header=None)
                self.log_message(f"Successfully loaded {self.test_file}")
            except Exception as e:
                self.log_message(f"Error loading {self.test_file}: {str(e)}")
                return
            
            # Keep original data for plotting
            self.original_train_data = self.train_data.copy()
            self.original_test_data = self.test_data.copy()
            
            self.log_message(f"\nTraining data shape: {self.train_data.shape}")
            self.log_message(f"Test data shape: {self.test_data.shape}")
            
            self.status_var.set("Data loaded successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.status_var.set("Error loading data")
    
    def train_model(self,display=True):
        if self.train_data is None or self.test_data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        
        try:
            self.log_message("⚙️ Training model using GradientBoostingRegressor...")
            # Use the final feature_info dict
            if not hasattr(self, 'feature_info') or not self.feature_info:
                messagebox.showerror("Error", "Please add feature columns and submit")
                return
            self.features = list(sorted(self.feature_info.keys()))

            # Target column
            try:
                self.target = int(self.target_index_entry.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Target column index must be an integer")
                return

            
            # Validate feature columns
            max_col = max(self.train_data.shape[1], self.test_data.shape[1]) - 1
            self.features = [f for f in self.features if f <= max_col]
            if self.target > max_col:
                messagebox.showerror("Error", f"Target column {self.target} exceeds data dimensions")
                return
            
            self.status_var.set("Standardizing features...")
            self.root.update()
            
            # Standardize features (except time)
            self.train_data_scaled, self.test_data_scaled, self.scalers = self.standardize_data(
                self.train_data, self.test_data, [f for f in self.features if f != 0]
            )
            
            # Prepare data
            X_train = self.train_data_scaled.iloc[:, self.features].values
            y_train = self.train_data_scaled.iloc[:, self.target].values
            X_test = self.test_data_scaled.iloc[:, self.features].values
            y_test = self.test_data_scaled.iloc[:, self.target].values
            
            self.status_var.set("Training Gradient Boosting model...")
            self.root.update()

            self.model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            self.model.fit(X_train, y_train)

            y_train_pred = self.model.predict(X_train)
            mse_train = mean_squared_error(y_train, y_train_pred)

            # Plot actual vs predicted
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(y_train, label='Actual', linewidth=2)
            ax.plot(y_train_pred, label='Predicted', linestyle='--')
            ax.set_xlabel('Time(s)')
            ax.set_ylabel('Target (Scaled)')
            ax.set_title(f'Actual vs Predicted\nTraining Prediction - MSE: {mse_train:.4f}')
            ax.legend()
            ax.grid(True)
            if display:
                self.canvas.draw()

            self.model_plot_fig = self.figure
            if not display:
                return self.embed_png_image(self.model_plot_fig)
    
            self.figure.savefig("model_plot.png", bbox_inches="tight", dpi=150)

            self.log_message("\nModel trained successfully. Training predictions plotted.")
            self.status_var.set("Model training complete")

            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Error in model training")
    
    def analyze_outliers(self,display=True):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first")
            return

        try:
            self.log_message("🔍 Analyzing outliers using prediction residuals...")
            threshold = float(self.threshold_entry.get())

            X_test = self.test_data_scaled.iloc[:, self.features].values
            y_test = self.test_data_scaled.iloc[:, self.target].values

            y_pred = self.model.predict(X_test)

            # Compute MSE per sample
            mse_values = np.array([
                mean_squared_error([y_test[i]], [y_pred[i]])
                for i in range(len(y_test))
            ])
            outliers = mse_values > threshold
            outlier_indices = np.where(outliers)[0]
            outlier_groups = self.group_consecutive_outliers(outlier_indices)

            if display:
                self.log_message(f"\nFound {sum(outliers)} outliers:")
                for i, idx in enumerate(outlier_indices):
                    self.log_message(f"{i+1}. Time: {X_test[idx, 0]:.1f}s, MSE: {mse_values[idx]:.4f}")
            

            # === Plot Actual vs Predicted with Outliers ===
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # Inverse transform if scaling exists
            if self.target in self.scalers:
                y_test_orig = self.scalers[self.target].inverse_transform(y_test.reshape(-1, 1)).flatten()
                y_pred_orig = self.scalers[self.target].inverse_transform(y_pred.reshape(-1, 1)).flatten()
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred

            ax.plot(X_test[:, 0], y_test_orig, 'b-', linewidth=2, label='Actual')
            ax.plot(X_test[:, 0], y_pred_orig, 'r--', linewidth=2, label='Predicted')

            if sum(outliers) > 0:
                outlier_times = X_test[outlier_indices, 0]
                outlier_values = y_test_orig[outlier_indices]
                ax.scatter(outlier_times, outlier_values, c='red', s=150, edgecolors='black',
                        linewidth=1.5, zorder=5, label='Outliers')

            ax.set_xlabel('Time (s)', fontsize=12)
            target_name = self.target_name_entry.get().strip() or f'Target {self.target}'
            ax.set_ylabel(f'{target_name} (Original Units)', fontsize=12)
            ax.set_title('Outlier Plot', fontsize=14)
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(True, linestyle=':', alpha=0.7)

            if display:
                self.canvas.draw()

            self.figure.savefig("outlier_plot.png", bbox_inches="tight", dpi=150)
            self.outlier_plot_fig = self.figure
            if not display:
                return self.embed_png_image(self.outlier_plot_fig)

            # Save for use in other methods
            self.outliers = outliers
            self.outlier_groups = outlier_groups
            self.X_test = X_test
            self.y_test = y_test

            self.status_var.set("Outlier analysis complete")
            self.log_message("✅ Outlier analysis complete. Residual plot generated.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze outliers: {str(e)}")
            self.status_var.set("Error in outlier analysis")


    
    def show_plots(self, save=True):
        if not hasattr(self, 'contributions') or not self.contributions:
            messagebox.showerror("Error", "No analysis results to plot")
            return

        # Start HTML
        html = """
        <html>
        <head><title>Interactive AeroSpace Vehicle Plots</title></head>
        <body style="background:#f4f4f4; font-family:Arial;">
        <h1 style="text-align:center;">Interactive Plots - AeroSpace Vehicle Data</h1>
        """

        # Use feature analysis data
        for feat in [f for f in self.features if f != 0 and f < self.original_train_data.shape[1]]:
            feat_name = self.feature_info.get(feat, f'Feature {feat}')
            train_x = self.original_train_data.iloc[:, 0].values
            train_y = self.original_train_data.iloc[:, feat].values
            test_x = self.original_test_data.iloc[:, 0].values
            test_y = self.original_test_data.iloc[:, feat].values

            # Points
            outlier_x = []
            outlier_y = []
            tooltips = []

            for idx, data in self.contributions.items():
                if feat in data['contributions']:
                    x = data['time']
                    y = self.original_test_data.iloc[self.test_data.index[idx], feat]
                    mse = data['contributions'][feat]['mse']
                    actual = data['contributions'][feat]['actual']
                    predicted = data['contributions'][feat]['expected']
                    outlier_x.append(x)
                    outlier_y.append(y)
                    tooltips.append(
                        f"<b>Outlier</b><br>Time: {x:.2f}<br>Actual: {actual:.2f}<br>Predicted: {predicted:.2f}<br>MSE: {mse:.4f}"
                    )

            # Plotly plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_x, y=train_y, mode='lines', name='Training', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=test_x, y=test_y, mode='lines', name='Test', line=dict(dash='solid')))
            fig.add_trace(go.Scatter(x=outlier_x, y=outlier_y, mode='markers', name='Outliers',
                                    marker=dict(size=12, color='red', line=dict(width=2, color='black')),
                                    text=tooltips, hoverinfo='text'))

            fig.update_layout(
                title=f"{feat_name} vs Time (Interactive)",
                xaxis_title="Time (s)",
                yaxis_title=f"{feat_name} (Original Units)",
                hovermode='closest',
                height=500,
            )

            fig.update_xaxes(rangeslider_visible=True)
            html += pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        # Add pie chart of average contributions
        if hasattr(self, 'avg_contributions') and self.avg_contributions:
            features = sorted(self.avg_contributions.keys())
            values = [self.avg_contributions[f] for f in features]
            labels = [self.feature_info.get(f, f'Feature {f}') for f in features]

            pie = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent')])
            pie.update_layout(title='📊 Average Feature Contributions to Outliers')
            html += pio.to_html(pie, include_plotlyjs='cdn', full_html=False)

        # Finalize HTML
        html += "</body></html>"

        # Save and open in browser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html)
            path = f.name

        if save:
           webbrowser.open(f'file://{path}')

        # ✅ ALSO SAVE PERSISTENT VERSION
        if save:
            default_name = "AeroSpace_Vehicle_Plots.html"
            file_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html")],
                initialfile=default_name,
                title="Save Interactive AeroSpace Vehicle Plot As..."
            )
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f_out:
                    f_out.write(html)
                self.log_message(f"✅ Interactive AeroSpace Vehicle plot saved to: {file_path}")
            else:
                self.log_message("⚠️ Save cancelled. No file was saved.")


        return html  # 👈 Return the HTML to use in the report

        

    def log_message(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update()
    
    def standardize_data(self, train_data, test_data, features):
        """Apply Min-Max scaling to features"""
        scalers = {}
        train_scaled = train_data.copy()
        test_scaled = test_data.copy()
        
        for feat in features:
            scaler = MinMaxScaler()
            train_scaled[feat] = scaler.fit_transform(train_data[[feat]])
            test_scaled[feat] = scaler.transform(test_data[[feat]])
            scalers[feat] = scaler
        
        return train_scaled, test_scaled, scalers
    
    def analyze_feature_contributions(self, train_data, test_data, features, target, outliers, outlier_groups, X_test, y_test):
        """Analyze contribution of each feature to grouped outliers"""
        contributions = {}
        time_col = 0  # Assuming first column is time

        # Ensure feature_info includes fallback names
        for f in features:
            if f not in self.feature_info:
                self.feature_info[f] = f"Feature {f}"

        feature_contributions = {f: {'total_percentage': 0, 'count': 0} for f in features if f != time_col}

        for group_id, group in enumerate(outlier_groups):
            if not group:
                continue

            group_size = len(group)
            group_indices = group
            idx = group[group_size // 2]  # representative index (middle)

            point_time = X_test[idx, time_col]
            start_time = X_test[group[0], time_col]
            end_time = X_test[group[-1], time_col]

            point_contributions = {}
            feature_mses = []
            total_mse = 0

            valid_features = [f for f in features if f != time_col and f < X_test.shape[1]]

            for feat in valid_features:
                X_train_feat = train_data.iloc[:, [time_col, feat]].values
                y_train_feat = train_data.iloc[:, target].values

                model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                model.fit(X_train_feat, y_train_feat)

                X_point = X_test[idx, [time_col, feat]].reshape(1, -1)
                y_pred = model.predict(X_point)
                mse = mean_squared_error([y_test[idx]], y_pred)

                feature_mses.append((feat, mse, y_pred[0]))
                total_mse += mse

            for feat, mse, pred in feature_mses:
                percent_contribution = (mse / total_mse) * 100 if total_mse > 0 else 0
                point_contributions[feat] = {
                    'percentage': percent_contribution,
                    'mse': mse,
                    'expected': pred,
                    'actual': y_test[idx]
                }

                if feat in feature_contributions:
                    feature_contributions[feat]['total_percentage'] += percent_contribution
                    feature_contributions[feat]['count'] += 1

            contributions[idx] = {
                'time': point_time,
                'start_time': start_time,
                'end_time': end_time,
                'group_size': group_size,
                'group_indices': group_indices,
                'total_mse': total_mse,
                'contributions': point_contributions
            }

        avg_contributions = {}
        for feat, data in feature_contributions.items():
            if data['count'] > 0:
                avg_contributions[feat] = data['total_percentage'] / data['count']

        return contributions, avg_contributions

    
    def plot_feature_analysis(self, train_data, test_data, features, target, contributions, original_train_data, original_test_data):
        """Generate plots using original (unscaled) values"""
        time_col = 0
        
        # Plot each feature's behavior with original values
        for feat in [f for f in features if f != time_col and f < original_train_data.shape[1]]:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Plot training data (using original values)
            ax.plot(original_train_data.iloc[:, time_col].values, 
                   original_train_data.iloc[:, feat].values, 
                   'b--', alpha=0.7, linewidth=1.5, label='Training Data')
            
            # Plot test data (using original values)
            ax.plot(original_test_data.iloc[:, time_col].values, 
                   original_test_data.iloc[:, feat].values, 
                   'g:', alpha=0.7, linewidth=2, label='Test Data')
            
            # Mark and annotate outliers
            for idx, data in contributions.items():
                if feat in data['contributions']:
                    contrib = data['contributions'][feat]
                    # Ensure we're using the correct index for original data
                    original_idx = idx
                    ax.scatter(data['time'], original_test_data.iloc[original_idx, feat], 
                             c='red', s=200, edgecolors='black', zorder=5,
                             label='Outlier' if idx == list(contributions.keys())[0] else "")
                    
                    ax.annotate(f"{contrib['percentage']:.1f}%", 
                              xy=(data['time'], original_test_data.iloc[original_idx, feat]),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round', fc='white', ec='black', alpha=0.8),
                              fontsize=10)
            
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel(f'Feature {feat} Value (Original Units)', fontsize=12)
            ax.set_title(f'Feature {feat} vs Time (Original Values)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.7)
            self.canvas.draw()
    
    def show_feature_contribution_pie(self):
        """Run contribution analysis and show pie chart"""

        # --- Check required data ---
        if self.model is None or self.test_data_scaled is None:
            messagebox.showerror("Error", "Model or test data not found. Train the model and run outlier detection first.")
            return
        
        if self.outliers is None or self.outlier_groups is None:
            messagebox.showerror("Error", "No outlier information found. Please run 'Analyze Outliers' first.")
            return

        try:
            # --- Run contribution analysis NOW ---
            self.contributions, self.avg_contributions = self.analyze_feature_contributions(
                self.train_data_scaled,
                self.test_data_scaled,
                self.features,
                self.target,
                self.outliers,
                self.outlier_groups,
                self.test_data_scaled.iloc[:, self.features].values,
                self.test_data_scaled.iloc[:, self.target].values
            )

            # --- Check if avg_contributions has useful data ---
            if not self.avg_contributions or sum(self.avg_contributions.values()) == 0:
                messagebox.showinfo("Info", "Feature contributions could not be calculated.")
                return
            

            self.log_message("📊 Generating feature contribution pie chart...")
            # --- Plot pie chart ---
            self.figure.clear()
            self.plot_feature_contributions_pie_fig(self.figure, self.avg_contributions)
            self.canvas.draw()
            self.log_message("✅ Feature contribution pie chart generated.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate contribution pie chart: {str(e)}")
            self.status_var.set("❌ Error in feature contribution analysis.")




    
    def plot_feature_contributions_pie_fig(self, fig, avg_contributions):
        """Plot pie chart on given figure"""
        # Prepare data for pie chart
        features = sorted(avg_contributions.keys())
        contributions = [avg_contributions[f] for f in features]
        labels = [self.feature_info.get(f, f'Feature {f}') for f in features]

        
        # Normalize to 100% if needed
        total = sum(contributions)
        if total > 100:
            contributions = [c/total*100 for c in contributions]
        
        # Create pie chart
        ax = fig.add_subplot(111)
        patches, texts, autotexts = ax.pie(
            contributions, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.85,
            labeldistance=1.05
        )
        
        # Equal aspect ratio ensures pie is drawn as a circle
        ax.axis('equal')
        
        # Make labels more readable
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
        
        ax.set_title('Average Feature Contributions to Outliers', fontsize=14, pad=20)

    def add_feature_info(self):
        try:
            index = int(self.feature_index_entry.get())
            name = self.feature_name_entry.get().strip()

            if not name:
                messagebox.showerror("Error", "Column name cannot be empty")
                return

            if not hasattr(self, 'feature_info') or not isinstance(self.feature_info, dict):
                self.feature_info = {}

            self.feature_info[index] = name

            self.feature_index_entry.delete(0, tk.END)
            self.feature_name_entry.delete(0, tk.END)

            self.update_feature_summary()
            self.log_message(f"➕ Added feature: {name} (index {index})")

        except ValueError:
            messagebox.showerror("Error", "Invalid column index. Must be an integer.")



    def update_feature_summary(self):
        self.feature_summary.config(state=tk.NORMAL)
        self.feature_summary.delete("1.0", tk.END)
        for idx, name in sorted(self.feature_info.items()):
            self.feature_summary.insert(tk.END, f"Index {idx} → {name}\n")
        self.feature_summary.config(state=tk.DISABLED)

    def final_submit_feature_info(self):
        if not hasattr(self, 'feature_info') or not self.feature_info:
            messagebox.showinfo("Info", "No features added.")
            return
        
        summary = "Final Feature Info:\n" + "\n".join(
            [f"Index {idx} → {name}" for idx, name in sorted(self.feature_info.items())]
        )
        messagebox.showinfo("Feature Info Submitted", summary)


    def embed_png_image(self, fig):
        buf = io.BytesIO()
        fig.set_size_inches(12, 6)
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return f'<img src="data:image/png;base64,{encoded}" style="width:100%; height:auto;" />'

    def generate_report(self):
        self.status_var.set("Generating report...")
        self.root.update()

        if not self.contributions or not self.avg_contributions:
            self.contributions, self.avg_contributions = self.analyze_feature_contributions(
                self.train_data_scaled,
                self.test_data_scaled,
                self.features,
                self.target,
                self.outliers,
                self.outlier_groups,
                self.test_data_scaled.iloc[:, self.features].values,
                self.test_data_scaled.iloc[:, self.target].values
            )

        try:
            plt.close('all')
            interactive_html = self.show_plots(save=False) or ""

            report = """
            <html>
            <head>
            <meta charset="UTF-8">
            <title>AeroSpace Vehicle Data Analysis Report</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f7f8; color: #2c3e50; }
                h1, h2, h3 { color: #1f3c88; }
                h1 { border-bottom: 3px solid #ccc; padding-bottom: 10px; }
                table { border-collapse: collapse; width: 100%; margin-top: 10px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                th { background-color: #dfe6e9; }
                img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ccc; border-radius: 5px; }
                .container { max-width: 1000px; margin: auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                .save-button { margin-top: 20px; padding: 10px 20px; background: #0984e3; color: white; border: none; border-radius: 5px; cursor: pointer; }
            </style>
            <script>
                function saveAsPDF() {
                    window.print();
                }
            </script>
            </head><body>
            <div class="container" contenteditable="true">
            <h1>🚀 AeroSpace Vehicle Data Analysis Report</h1>
            """

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report += f"<p><strong>Generated:</strong> {timestamp}</p>"

            report += "<h2>📁 Files</h2>"
            report += f"<p><strong>Training Files:</strong><br>{'<br>'.join(self.train_files)}</p>"
            report += f"<p><strong>Test File:</strong><br>{self.test_file}</p>"

            report += "<h2>📊 Data Shapes</h2>"
            report += f"<p>Train Data Shape: {self.train_data.shape}</p>"
            report += f"<p>Test Data Shape: {self.test_data.shape}</p>"

            report += "<h2>⚙️ Configuration</h2><ul>"
            report += f"<li>Threshold: {self.threshold_entry.get()}</li>"
            for idx, name in sorted(self.feature_info.items()):
                report += f"<li>Feature {idx}: {name}</li>"
            report += "</ul>"

            target_idx = self.target if self.target is not None else "-"
            target_name = self.target_name_entry.get().strip() or f"Target {target_idx}"
            report += f"<p><strong>Target:</strong> Index {target_idx} - {target_name}</p>"

            report += "<h2>📈 Model Performance</h2>"
            report += f"<p><strong>Model:</strong> Gradient Boosting Regressor</p>"
            report += f"<p><strong>Total Outliers Detected:</strong> {len(self.contributions)}</p>"

            report += "<h2>🚨 Outlier Contributions</h2><table><tr><th>Group</th><th>Time Range</th><th>Group Size</th><th>Total MSE</th><th>Feature</th><th>%</th><th>MSE</th><th>Expected</th><th>Actual</th></tr>"
            for i, (idx, entry) in enumerate(self.contributions.items(), 1):
                try:
                    first_row = True
                    for feat, contrib in entry['contributions'].items():
                        report += "<tr>"
                        if first_row:
                            report += f"<td rowspan='{len(entry['contributions'])}'>{i}</td>"
                            report += f"<td rowspan='{len(entry['contributions'])}'>{entry['start_time']:.1f}–{entry['end_time']:.1f}</td>"
                            report += f"<td rowspan='{len(entry['contributions'])}'>{entry['group_size']}</td>"
                            report += f"<td rowspan='{len(entry['contributions'])}'>{entry['total_mse']:.4f}</td>"
                            first_row = False
                        report += (
                            f"<td>{self.feature_info.get(feat, f'Feature {feat}')}</td>"
                            f"<td>{contrib['percentage']:.1f}%</td><td>{contrib['mse']:.4f}</td>"
                            f"<td>{contrib['expected']:.2f}</td><td>{contrib['actual']:.2f}</td>"
                        )
                        report += "</tr>"
                except Exception as e:
                    report += f"<tr><td colspan='9'>❌ Error in group {i}: {str(e)}</td></tr>"
            report += "</table>"

            report += "<h2>📊 Average Feature Contributions</h2><table><tr><th>Feature</th><th>Avg %</th></tr>"
            for feat, percent in sorted(self.avg_contributions.items(), key=lambda x: -x[1]):
                name = self.feature_info.get(feat, f"Feature {feat}")
                report += f"<tr><td>{name}</td><td>{percent:.2f}</td></tr>"
            report += "</table>"

            report += "<h2>📷 Static Plots</h2>"
            report += "<h3>📉 Model Prediction (Train)</h3>"
            report += self.train_model(display=False)
            report += "<h3>🚨 Outlier Detection</h3>"
            report += self.analyze_outliers(display=False)

            report += "<h2>📍 Interactive Plots</h2>"
            report += interactive_html

            report += '<button class="save-button" onclick="saveAsPDF()">Save as PDF</button>'
            report += "</div></body></html>"

            default_name = "Interactive_Report.html"
            save_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML files", "*.html")],
                initialfile=default_name,
                title="Save HTML Report As..."
            )
            if save_path:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(report)
                self.log_message(f"✅ Report saved to {save_path}")
            else:
                self.log_message("⚠️ Report save cancelled.")
                return

            webbrowser.open(f'file://{save_path}')  # ✅ Open in default browser


            self.status_var.set("✅ Report generated and opened in browser.")
            self.log_message("📄 ✅ Report generated and opened in browser.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set("❌ Error generating report.")
            messagebox.showerror("Error", f"An error occurred while generating report:\n{e}")


    def group_consecutive_outliers(self, outlier_indices):
        """Group consecutive indices into clusters"""
        if len(outlier_indices) == 0:
            return []
        
        groups = []
        current = [outlier_indices[0]]
        
        for idx in outlier_indices[1:]:
            if idx == current[-1] + 1:
                current.append(idx)
            else:
                groups.append(current)
                current = [idx]
        
        if current:
            groups.append(current)
        
        return groups


if __name__ == "__main__":
    root = tk.Tk()
    app = AeroSpaceVehicleAnalysisApp(root)
    root.mainloop()