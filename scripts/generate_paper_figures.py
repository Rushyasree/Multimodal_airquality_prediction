"""
scripts/generate_paper_figures.py
Generate publication-quality experimental figures for IEEE research manuscript.
All plots are derived strictly from real test evaluation artifacts:
- data/processed/model_sequences/y_test.npy
- models/lstm_baseline/test_predictions.npy
- reports/classification_metrics_report.json
- reports/confusion_matrix_pm25.csv
- reports/confusion_matrix_pm10.csv
- models/lstm_baseline/test_metrics.json
- models/lstm_baseline/persistence_comparison.json
- data/processed/sequences/air_quality_24h_sequence_index.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator

# Ensure temp mpl directory
os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get('TEMP', '.'), 'matplotlib_config')

# Output directories
FIGURES_DIR = os.path.abspath('paper/figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling configuration for IEEE publication
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def validate_and_load_data():
    """Validate all source files and load numerical arrays and metrics."""
    print("Validating source artifacts...")
    
    y_test_path = 'data/processed/model_sequences/y_test.npy'
    preds_path = 'models/lstm_baseline/test_predictions.npy'
    report_path = 'reports/classification_metrics_report.json'
    cm_pm25_path = 'reports/confusion_matrix_pm25.csv'
    cm_pm10_path = 'reports/confusion_matrix_pm10.csv'
    persist_path = 'models/lstm_baseline/persistence_comparison.json'
    seq_index_path = 'data/processed/sequences/air_quality_24h_sequence_index.csv'
    
    for path in [y_test_path, preds_path, report_path, cm_pm25_path, cm_pm10_path, persist_path, seq_index_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required artifact: {path}")

    y_test = np.load(y_test_path)
    preds = np.load(preds_path)
    
    if y_test.shape != (404431, 2) or preds.shape != (404431, 2):
        raise ValueError(f"Unexpected shape: y_test {y_test.shape}, preds {preds.shape}")
        
    if not np.isfinite(y_test).all() or not np.isfinite(preds).all():
        raise ValueError("Non-finite values detected in test arrays.")
        
    with open(report_path, 'r', encoding='utf-8') as f:
        metrics_report = json.load(f)
        
    with open(persist_path, 'r', encoding='utf-8') as f:
        persist_report = json.load(f)
        
    cm_pm25 = pd.read_csv(cm_pm25_path, index_col=0)
    cm_pm10 = pd.read_csv(cm_pm10_path, index_col=0)
    
    print("All artifacts successfully verified and loaded.")
    return {
        'y_test': y_test,
        'preds': preds,
        'metrics_report': metrics_report,
        'persist_report': persist_report,
        'cm_pm25': cm_pm25,
        'cm_pm10': cm_pm10,
        'seq_index_path': seq_index_path
    }


def plot_actual_vs_predicted(data):
    """Figure 1: Two panels (PM2.5 and PM10) actual vs predicted with y=x line."""
    print("Generating Figure 1: actual_vs_predicted...")
    y_test = data['y_test']
    preds = data['preds']
    metrics = data['metrics_report']['regression_metrics']
    
    # Subsampling strategy: reproducible random subsample of 12,000 points
    rng = np.random.RandomState(42)
    sample_size = 12000
    indices = rng.choice(len(y_test), size=sample_size, replace=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    
    pollutants = [
        ('PM2.5', 0, axes[0], '#1f77b4', '#0d47a1', 350),
        ('PM10', 1, axes[1], '#2ca02c', '#1b5e20', 600)
    ]
    
    for name, col_idx, ax, dot_color, text_color, max_val in pollutants:
        y_true_sub = y_test[indices, col_idx]
        y_pred_sub = preds[indices, col_idx]
        
        # Hexbin density background for the subsample to clearly show density
        hb = ax.hexbin(y_true_sub, y_pred_sub, gridsize=45, cmap='Blues' if col_idx == 0 else 'YlGn', 
                       mincnt=1, bins='log', alpha=0.85, edgecolors='none')
        
        # Reference y = x line
        ax.plot([0, max_val], [0, max_val], color='#d62728', linestyle='--', linewidth=1.5, label='Ideal 1:1 Reference ($y = x$)')
        
        # Colorbar for hexbin
        cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.03)
        cb.set_label('Point Density (log scale)', fontsize=8.5)
        cb.ax.tick_params(labelsize=8)
        
        # Annotation of FULL test set metrics vs visualization sample
        m = metrics[name]
        info_text = (
            f"Full Offline Test Cohort ($N = 404,431$)\n"
            f"  $R^2$ Score: {m['R2']:.4f}\n"
            f"  MAE: {m['MAE']:.2f} $\\mu\\mathrm{{g/m}}^3$\n"
            f"  RMSE: {m['RMSE']:.2f} $\\mu\\mathrm{{g/m}}^3$\n"
            f"---------------------------------\n"
            f"Visualized: Sample of $n = {sample_size:,}$ points"
        )
        ax.text(0.04, 0.96, info_text, transform=ax.transAxes, verticalalignment='top',
                fontsize=8.5, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.92))
        
        ax.set_title(f"({chr(97 + col_idx)}) $\\mathrm{{{name}}}$ Offline Forecasting Evaluation", fontsize=11.5, pad=10, weight='bold')
        ax.set_xlabel(f"Observed $\\mathrm{{{name}}}$ Concentration ($\\mu\\mathrm{{g/m}}^3$)", fontsize=10.5)
        ax.set_ylabel(f"Predicted $\\mathrm{{{name}}}$ Concentration ($\\mu\\mathrm{{g/m}}^3$)", fontsize=10.5)
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=9)

    plt.suptitle("Offline Test Set: Observed vs. LSTM Next-Hour Predicted Concentrations", fontsize=12.5, y=0.98, weight='bold')
    plt.tight_layout()
    
    png_path = os.path.join(FIGURES_DIR, 'actual_vs_predicted.png')
    pdf_path = os.path.join(FIGURES_DIR, 'actual_vs_predicted.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


def plot_confusion_matrix(cm_df, pollutant_name, metrics, filename_stem, cmap_name):
    """Figures 2 & 3: Publication-ready confusion matrix heatmap with counts and row percentages."""
    print(f"Generating Figure: {filename_stem}...")
    
    categories = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']
    
    # Thresholds for labelling
    if pollutant_name == 'PM2.5':
        thresholds = ['[0-30]', '(30-60]', '(60-90]', '(90-120]', '(120-250]', '(>250)']
    else:
        thresholds = ['[0-50]', '(50-100]', '(100-250]', '(250-350]', '(350-430]', '(>430)']
        
    axis_labels = [f"{cat}\n{thresh}" for cat, thresh in zip(categories, thresholds)]
    
    matrix = cm_df.values.astype(int)
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_matrix = matrix / row_sums.astype(float)
    
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    im = ax.imshow(norm_matrix, cmap=cmap_name, vmin=0, vmax=1.0)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Row-Normalized Fraction (Recall per Class)', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(axis_labels, fontsize=9.5)
    ax.set_yticklabels(axis_labels, fontsize=9.5)
    
    ax.set_xlabel('Predicted CPCB Category (Hourly)', fontsize=11, labelpad=8, weight='bold')
    ax.set_ylabel('Ground Truth CPCB Category (Hourly)', fontsize=11, labelpad=8, weight='bold')
    
    # Overlay exact counts and row percentages
    threshold_color = 0.5
    for i in range(len(categories)):
        for j in range(len(categories)):
            count = matrix[i, j]
            pct = norm_matrix[i, j] * 100
            val_text = f"{count:,}\n({pct:.1f}%)"
            text_color = "white" if norm_matrix[i, j] > threshold_color else "#111111"
            ax.text(j, i, val_text, ha="center", va="center", color=text_color,
                    fontsize=8.5, weight='bold' if i == j else 'normal')
            
    # Add summary banner box
    acc = metrics['exact_accuracy'] * 100
    adj_acc = metrics['adjacent_category_accuracy'] * 100
    macro_f1 = metrics['macro_avg']['f1_score']
    
    summary_text = (
        f"Offline Test Set: $N = 404,431$ sequences | Exact Accuracy: {acc:.2f}% | "
        f"Macro F1: {macro_f1:.4f} | Adjacent ($\\pm 1$ bin) Accuracy: {adj_acc:.2f}%"
    )
    plt.title(f"CPCB Category Classification: $\\mathrm{{{pollutant_name}}}$ Forecast Confusion Matrix\n"
              f"{summary_text}",
              fontsize=10.5, pad=12, weight='bold')
    
    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, f"{filename_stem}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{filename_stem}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


def plot_prediction_timeseries(data):
    """Figure 4: Continuous 168-hour (1 week) test sequence from Station DL007."""
    print("Generating Figure 4: prediction_timeseries...")
    seq_index_path = data['seq_index_path']
    y_test = data['y_test']
    preds = data['preds']
    
    df = pd.read_csv(seq_index_path)
    test_df = df[df['split'] == 'test'].copy()
    test_df['target_time'] = pd.to_datetime(test_df['target_time'])
    
    # Station DL007: CRRI Mathura Road, Delhi - IMD
    # Select continuous 168-hour window (7 days) in November 2019
    st_mask = (test_df['StationId'] == 'DL007')
    st_indices = np.where(st_mask)[0]
    st_sub = test_df.iloc[st_indices].copy()
    
    start_time = pd.Timestamp('2019-11-15 00:00:00')
    end_time = pd.Timestamp('2019-11-21 23:00:00')
    
    window_mask = (st_sub['target_time'] >= start_time) & (st_sub['target_time'] <= end_time)
    window_indices = st_indices[window_mask]
    window_df = st_sub[window_mask]
    
    timestamps = window_df['target_time'].values
    pm25_true = y_test[window_indices, 0]
    pm25_pred = preds[window_indices, 0]
    pm10_true = y_test[window_indices, 1]
    pm10_pred = preds[window_indices, 1]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 6.8), sharex=True)
    
    # PM2.5 Panel
    ax1.plot(timestamps, pm25_true, color='#0d47a1', linewidth=1.8, label='Ground Truth Observed ($y_t$)', alpha=0.95)
    ax1.plot(timestamps, pm25_pred, color='#d84315', linewidth=1.5, linestyle='--', label='LSTM 1-Hour Forecast ($\\hat{y}_t$)', alpha=0.9)
    ax1.fill_between(timestamps, pm25_true, pm25_pred, color='#ffccbc', alpha=0.4, label='Prediction Residual')
    ax1.set_ylabel('$\\mathrm{PM}_{2.5}$ Concentration ($\\mu\\mathrm{g/m}^3$)', fontsize=10.5, weight='bold')
    ax1.set_title('(a) $\\mathrm{PM}_{2.5}$ Hourly Ground Truth vs. LSTM Forecast Sequence', fontsize=11, loc='left', weight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
    
    # Compute window-specific metrics
    pm25_mae = np.mean(np.abs(pm25_true - pm25_pred))
    pm25_rmse = np.sqrt(np.mean((pm25_true - pm25_pred)**2))
    ax1.text(0.015, 0.88, f"Window MAE: {pm25_mae:.2f} $\\mu\\mathrm{{g/m}}^3$ | RMSE: {pm25_rmse:.2f} $\\mu\\mathrm{{g/m}}^3$",
             transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    
    # PM10 Panel
    ax2.plot(timestamps, pm10_true, color='#1b5e20', linewidth=1.8, label='Ground Truth Observed ($y_t$)', alpha=0.95)
    ax2.plot(timestamps, pm10_pred, color='#e65100', linewidth=1.5, linestyle='--', label='LSTM 1-Hour Forecast ($\\hat{y}_t$)', alpha=0.9)
    ax2.fill_between(timestamps, pm10_true, pm10_pred, color='#c8e6c9', alpha=0.4, label='Prediction Residual')
    ax2.set_ylabel('$\\mathrm{PM}_{10}$ Concentration ($\\mu\\mathrm{g/m}^3$)', fontsize=10.5, weight='bold')
    ax2.set_title('(b) $\\mathrm{PM}_{10}$ Hourly Ground Truth vs. LSTM Forecast Sequence', fontsize=11, loc='left', weight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.92, fontsize=9.5)
    
    pm10_mae = np.mean(np.abs(pm10_true - pm10_pred))
    pm10_rmse = np.sqrt(np.mean((pm10_true - pm10_pred)**2))
    ax2.text(0.015, 0.88, f"Window MAE: {pm10_mae:.2f} $\\mu\\mathrm{{g/m}}^3$ | RMSE: {pm10_rmse:.2f} $\\mu\\mathrm{{g/m}}^3$",
             transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
    
    # Format x-axis with date and time
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n(%a)'))
    ax2.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    ax2.set_xlabel('Timestamp (Hourly Resolution - Chronological Held-out Test Split)', fontsize=10.5, weight='bold')
    
    plt.suptitle("Station DL007 (CRRI Mathura Road, Delhi - IMD): 1-Week Offline Forecast Trajectory (Nov 15–21, 2019)",
                 fontsize=12, y=0.98, weight='bold')
    plt.tight_layout()
    
    png_path = os.path.join(FIGURES_DIR, 'prediction_timeseries.png')
    pdf_path = os.path.join(FIGURES_DIR, 'prediction_timeseries.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


def plot_regression_metrics(data):
    """Figure 5: Clean scientific comparison of MAE and RMSE on physical scale, with R2 on separate panel."""
    print("Generating Figure 5: regression_metrics...")
    p_comp = data['persist_report']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={'width_ratios': [1.3, 1.0]})
    
    # Panel 1: Physical Error Metrics (MAE and RMSE in µg/m³)
    pollutants = ['PM2.5', 'PM10']
    x = np.arange(len(pollutants))
    width = 0.20
    
    # 4 bars per pollutant: Persistence MAE, LSTM MAE, Persistence RMSE, LSTM RMSE
    pers_mae = [p_comp['persistence']['PM2.5']['MAE'], p_comp['persistence']['PM10']['MAE']]
    lstm_mae = [p_comp['lstm']['PM2.5']['MAE'], p_comp['lstm']['PM10']['MAE']]
    pers_rmse = [p_comp['persistence']['PM2.5']['RMSE'], p_comp['persistence']['PM10']['RMSE']]
    lstm_rmse = [p_comp['lstm']['PM2.5']['RMSE'], p_comp['lstm']['PM10']['RMSE']]
    
    rects1 = ax1.bar(x - 1.5*width, pers_mae, width, label='Persistence MAE', color='#90caf9', edgecolor='#1565c0', hatch='//')
    rects2 = ax1.bar(x - 0.5*width, lstm_mae, width, label='LSTM MAE', color='#1565c0', edgecolor='#0d47a1')
    rects3 = ax1.bar(x + 0.5*width, pers_rmse, width, label='Persistence RMSE', color='#ffab91', edgecolor='#d84315', hatch='\\\\')
    rects4 = ax1.bar(x + 1.5*width, lstm_rmse, width, label='LSTM RMSE', color='#d84315', edgecolor='#bf360c')
    
    # Bar value callout annotations
    for rects in [rects1, rects2, rects3, rects4]:
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f"{height:.1f}",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, weight='bold')
            
    ax1.set_ylabel('Concentration Error ($\\mu\\mathrm{g/m}^3$)', fontsize=10.5, weight='bold')
    ax1.set_title('(a) Error Metrics on Physical Scale ($\\mu\\mathrm{g/m}^3$)', fontsize=11, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['$\\mathrm{PM}_{2.5}$', '$\\mathrm{PM}_{10}$'], fontsize=11, weight='bold')
    ax1.set_ylim(0, 42)
    ax1.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.92, ncol=2)
    
    # Panel 2: Dimensionless Coefficient of Determination (R²)
    r2_pers = [p_comp['persistence']['PM2.5']['R2'], p_comp['persistence']['PM10']['R2']]
    r2_lstm = [p_comp['lstm']['PM2.5']['R2'], p_comp['lstm']['PM10']['R2']]
    
    width_r2 = 0.28
    rects_r1 = ax2.bar(x - 0.5*width_r2, r2_pers, width_r2, label='1-Hour Persistence', color='#b0bec5', edgecolor='#37474f', hatch='//')
    rects_r2 = ax2.bar(x + 0.5*width_r2, r2_lstm, width_r2, label='2-Layer LSTM Forecaster', color='#2e7d32', edgecolor='#1b5e20')
    
    for rects in [rects_r1, rects_r2]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f"{height:.4f}",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=8.5, weight='bold')
            
    ax2.set_ylabel('Coefficient of Determination ($R^2$)', fontsize=10.5, weight='bold')
    ax2.set_title('(b) Explained Variance Ratio ($R^2$, Dimensionless)', fontsize=11, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['$\\mathrm{PM}_{2.5}$', '$\\mathrm{PM}_{10}$'], fontsize=11, weight='bold')
    ax2.set_ylim(0.88, 0.98)
    ax2.grid(True, linestyle=':', alpha=0.5, axis='y')
    ax2.legend(loc='lower right', fontsize=8.5, framealpha=0.92)
    
    plt.suptitle("Offline Benchmark Performance: LSTM Forecaster vs. 1-Hour Persistence (N = 404,431)",
                 fontsize=11.5, y=0.98, weight='bold')
    plt.tight_layout()
    
    png_path = os.path.join(FIGURES_DIR, 'regression_metrics.png')
    pdf_path = os.path.join(FIGURES_DIR, 'regression_metrics.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved: {png_path} and {pdf_path}")


def main():
    print("="*70)
    print("Starting generation of IEEE publication figures...")
    print("="*70)
    data = validate_and_load_data()
    
    # 1. actual_vs_predicted
    plot_actual_vs_predicted(data)
    
    # 2. confusion_matrix_pm25
    plot_confusion_matrix(data['cm_pm25'], 'PM2.5',
                          data['metrics_report']['classification_metrics']['PM2.5'],
                          'confusion_matrix_pm25', 'Blues')
    
    # 3. confusion_matrix_pm10
    plot_confusion_matrix(data['cm_pm10'], 'PM10',
                          data['metrics_report']['classification_metrics']['PM10'],
                          'confusion_matrix_pm10', 'YlGn')
    
    # 4. prediction_timeseries
    plot_prediction_timeseries(data)
    
    # 5. regression_metrics
    plot_regression_metrics(data)
    
    print("="*70)
    print("All figures successfully created in:", FIGURES_DIR)
    print("="*70)


if __name__ == '__main__':
    main()
