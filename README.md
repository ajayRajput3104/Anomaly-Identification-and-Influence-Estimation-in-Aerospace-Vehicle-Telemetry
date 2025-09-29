# 🚀 Aerospace Telemetry Anomaly Detection Tool

A desktop-based Python application designed to detect anomalies in aerospace telemetry data and analyze feature contributions to these anomalies.
The tool combines machine learning, data visualization, and automated reporting into a user-friendly interface for engineers and analysts.

## 🔎 Overview
-Anomaly Detection: Identifies outliers in telemetry data using a Gradient Boosting Regressor and Mean Squared Error (MSE).
Feature Contribution Analysis: Determines which features contributed most to each anomaly.
-Visualization:
Static plots with Matplotlib.
Interactive dashboards with Plotly (zoom, tooltips, timeline slider).
-Automated Reporting: Generates structured HTML/PDF reports with plots and tables.
-GUI Application: Built with Tkinter for easy dataset selection, parameter tuning, and one-click analysis.

# 🛠 Features
-Load multiple training files and one test file.
-Train predictive models with Gradient Boosting.
-Detect and group anomalies with threshold-based outlier analysis.
-Visualize results with actual vs predicted plots, outlier plots, and pie charts.
-Explore results interactively in a browser-based dashboard.
-Export final analysis into comprehensive HTML/PDF reports.
-Reset and re-run analysis seamlessly within the GUI.

## 📂 Project Structure
```
.
├── Batch2.py              # Main application (Tkinter GUI + ML logic)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── (generated outputs)
    ├── reports/           # HTML / PDF reports
    └── plots/             # Saved static plots
```
## 📦 Installation

-Clone the repository:
```bash
git clone https://github.com/<your-username>/aerospace-anomaly-detection.git
cd aerospace-anomaly-detection
```

-Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
python Batch2.py
```

## 📊 Example Workflow
-Launch the tool (python Batch2.py).
-Select training and test datasets via the GUI.
-Configure anomaly threshold and feature/target columns.
-Train the model and detect anomalies.
-View:
Static plots (Actual vs Predicted, Outlier plots).
Interactive dashboard with zoom and hover insights.
Generate the final HTML/PDF report with all results.

## 📚 Tech Stack
-Python 3.7+
-Tkinter → GUI framework
-Pandas, NumPy → Data preprocessing
-Scikit-learn → Gradient Boosting Regressor, scaling
-Matplotlib → Static visualization
-Plotly → Interactive dashboard
-FPDF → PDF report generation

## 📸 Screenshots (to add)
GUI layout after launch
Example anomaly detection plot
Feature contribution pie chart
Interactive dashboard snapshot

## 🚀 Future Enhancements
-Real-time anomaly detection from live telemetry streams.
-Advanced models (LSTMs, Transformers) for sequential data.
-Cloud-based deployment with multi-user access.
-Integration with explainable AI (e.g., SHAP values).

## 📝 License

This project is open-source and available under the MIT License.
