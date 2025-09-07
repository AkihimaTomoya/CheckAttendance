# generate_all_figures.py

import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# PHẦN 1: KHU VỰC CẤU HÌNH (CHỈ CẦN CHỈNH SỬA Ở ĐÂY)
# ==============================================================================
CONFIG = {
    # 1. Đường dẫn đến các file log và tên model tương ứng
    "log_files": [
        'test/res50.txt',
        'test/res50_custom1.txt',
        'test/res50_custom2.txt',
        'test/res50_fan.txt'
    ],
    "model_names": [
        'ResNet-50 (Baseline)',
        'Our Method (Full)',
        'Our Method (Ablation)',
        'ResNet-50 + FFN'
    ],

    # 2. Cài đặt cho thư mục output và định dạng file
    "output_dir": "paper_figures",
    "formats": ["png", "pdf"],  # Lưu ở cả định dạng vector (pdf) và raster (png)
    "dpi": 300,

    # 3. Cài đặt thẩm mỹ cho biểu đồ
    "palette": "colorblind",  # Bảng màu đẹp và khoa học (khác: 'viridis', 'Set2', 'Paired')
    "font_family": "serif",  # Font chữ (serif ~ Times New Roman)
    "font_size": 12,
    "annotate_bars": True  # True để hiện giá trị trên các cột, False để ẩn
}


# ==============================================================================
# PHẦN 2: CÁC HÀM LÕI (Không cần chỉnh sửa)
# ==============================================================================

# --- Nhóm hàm xử lý dữ liệu ---
def parse_log_file(filepath):
    """Đọc và trích xuất dữ liệu từ một file log."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{filepath}'")
        return None

    pattern_metrics = (
        r"\[(\w+)\]Accuracy: (\d+\.\d+).*?\n"
        r"\[\1\]Accuracy-Flip: (\d+\.\d+).*?\n"
        r"\[\1\]VAL @ FAR=1e-3: (\d+\.\d+).*?\n"
        r"\[\1\]EER: (\d+\.\d+)"
    )
    pattern_infer_time = r"infer time ([\d\.]+)"

    matches_metrics = re.findall(pattern_metrics, content, re.DOTALL)
    matches_time = re.findall(pattern_infer_time, content)

    if not matches_metrics:
        print(f"Cảnh báo: Không tìm thấy dữ liệu metrics đầy đủ trong '{filepath}'.")
        return None

    data = {m[0]: {'Accuracy': float(m[1]), 'Accuracy-Flip': float(m[2]),
                   'VAL @ FAR=1e-3': float(m[3]), 'EER (%)': float(m[4])} for m in matches_metrics}

    avg_infer_time = np.mean([float(t) for t in matches_time]) if matches_time else 0
    return {'metrics': data, 'avg_infer_time': avg_infer_time}


# --- Nhóm hàm vẽ biểu đồ ---
def setup_matplotlib_style():
    """Thiết lập style chung cho tất cả các biểu đồ."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': CONFIG['font_family'],
        'font.size': CONFIG['font_size'],
        'axes.labelsize': CONFIG['font_size'],
        'axes.titlesize': CONFIG['font_size'] + 2,
        'xtick.labelsize': CONFIG['font_size'] - 1,
        'ytick.labelsize': CONFIG['font_size'] - 1,
        'legend.fontsize': CONFIG['font_size'] - 1,
        'figure.titlesize': CONFIG['font_size'] + 4,
    })


def save_figure(fig, filename):
    """Lưu figure với các định dạng đã cấu hình."""
    for fmt in CONFIG['formats']:
        full_path = os.path.join(CONFIG['output_dir'], f"{filename}.{fmt}")
        fig.savefig(full_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    print(f"Đã lưu: {filename}.[{', '.join(CONFIG['formats'])}]")


def plot_main_performance(df, all_data):
    """Vẽ figure 2x2 tổng hợp các kết quả hiệu năng chính."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Main Performance Comparison Across Datasets', fontsize=18, weight='bold')

    # (a) Accuracy-Flip
    ax = sns.barplot(data=df, x='Dataset', y='Accuracy-Flip', hue='Model', ax=axes[0, 0], palette=CONFIG['palette'])
    ax.set_title('(a) Verification Accuracy (with flip)', weight='bold')
    ax.set_ylabel('Accuracy');
    ax.set_xlabel(None)
    ax.set_ylim(bottom=max(0, df['Accuracy-Flip'].min() * 0.98), top=1.005)
    if CONFIG['annotate_bars']:
        for container in ax.containers: ax.bar_label(container, fmt='%.3f', size=7, rotation=90, padding=3)
    ax.get_legend().remove()

    # (b) EER (%)
    ax = sns.barplot(data=df, x='Dataset', y='EER (%)', hue='Model', ax=axes[0, 1], palette=CONFIG['palette'])
    ax.set_title('(b) Equal Error Rate (EER)', weight='bold')
    ax.set_ylabel('EER (%) (Lower is better)');
    ax.set_xlabel(None)
    ax.get_legend().remove()

    # (c) VAL @ FAR=1e-3
    ax = sns.barplot(data=df, x='Dataset', y='VAL @ FAR=1e-3', hue='Model', ax=axes[1, 0], palette=CONFIG['palette'])
    ax.set_title('(c) Verification Rate at FAR=1e-3', weight='bold')
    ax.set_ylabel('Verification Rate');
    ax.set_xlabel(None)
    ax.get_legend().remove()

    # (d) Speed vs. Accuracy Trade-off
    ax = axes[1, 1]
    infer_times = [d['avg_infer_time'] for d in all_data.values() if d]
    avg_accuracies = df.groupby('Model')['Accuracy-Flip'].mean().reindex(CONFIG['model_names'])
    df_tradeoff = pd.DataFrame(
        {'Model': CONFIG['model_names'], 'Avg Infer Time (s)': infer_times, 'Avg Accuracy': avg_accuracies})
    sns.scatterplot(data=df_tradeoff, x='Avg Infer Time (s)', y='Avg Accuracy', hue='Model', s=250, ax=ax,
                    palette=CONFIG['palette'], legend=False, style='Model', markers=True)
    for i in df_tradeoff.index: ax.text(df_tradeoff.loc[i, 'Avg Infer Time (s)'],
                                        df_tradeoff.loc[i, 'Avg Accuracy'] + 0.0005, df_tradeoff.loc[i, 'Model'],
                                        ha='center', size=9)
    ax.set_title('(d) Speed-Accuracy Trade-off', weight='bold')
    ax.set_xlabel('Average Inference Time (s) (Lower is better)')
    ax.set_ylabel('Average Accuracy (Higher is better)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(CONFIG['model_names']), bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_figure(fig, 'figure_1_main_performance')


def plot_stability_analysis(df):
    """Vẽ figure 1x2 phân tích độ ổn định và augmentation."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Performance Stability and Augmentation Analysis', fontsize=16, weight='bold')

    # (a) Performance Distribution
    ax = sns.boxplot(data=df, x='Model', y='Accuracy-Flip', ax=axes[0], palette=CONFIG['palette'])
    sns.stripplot(data=df, x='Model', y='Accuracy-Flip', color=".25", size=5, ax=axes[0])
    ax.set_title('(a) Performance Distribution Across Datasets', weight='bold')
    ax.set_ylabel('Accuracy-Flip');
    ax.set_xlabel(None)
    ax.tick_params(axis='x', rotation=10)

    # (b) Flip Improvement
    df['Flip Improvement (%)'] = (df['Accuracy-Flip'] - df['Accuracy']) * 100
    ax = sns.barplot(data=df, x='Dataset', y='Flip Improvement (%)', hue='Model', ax=axes[1], palette=CONFIG['palette'])
    ax.set_title('(b) Accuracy Gain from Flip Augmentation', weight='bold')
    ax.set_ylabel('Improvement (%)');
    ax.set_xlabel(None)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, 'figure_2_stability_analysis')


def plot_summary_heatmap(df, all_data):
    """Vẽ figure heatmap tổng hợp thông minh."""
    df_summary = df.groupby('Model').mean(numeric_only=True).reindex(CONFIG['model_names'])
    df_summary['Avg Infer Time (s)'] = [d['avg_infer_time'] for d in all_data.values() if d]

    # Tách heatmap thành 2 phần để trực quan hơn
    high_is_better = df_summary[['Accuracy-Flip', 'VAL @ FAR=1e-3']]
    low_is_better = df_summary[['EER (%)', 'Avg Infer Time (s)']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Metrics Summary', fontsize=16, weight='bold')

    sns.heatmap(high_is_better, annot=True, fmt=".4f", cmap="Greens", ax=axes[0], linewidths=.5,
                cbar_kws={'label': 'Higher is better'})
    axes[0].set_title('Positive Metrics', weight='bold')

    sns.heatmap(low_is_better, annot=True, fmt=".3f", cmap="Reds_r", ax=axes[1], linewidths=.5,
                cbar_kws={'label': 'Lower is better'})
    axes[1].set_title('Negative Metrics', weight='bold');
    axes[1].set_ylabel('')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, 'figure_3_summary_heatmap')


# ==============================================================================
# PHẦN 3: HÀM THỰC THI CHÍNH
# ==============================================================================
def main():
    """Hàm chính điều khiển toàn bộ quy trình."""
    # 1. Thiết lập môi trường
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        print(f"Đã tạo thư mục: '{CONFIG['output_dir']}'")

    # 2. Đọc và xử lý dữ liệu từ các file log
    all_data = {name: parse_log_file(fp) for name, fp in zip(CONFIG['model_names'], CONFIG['log_files'])}

    valid_data = {k: v for k, v in all_data.items() if v}
    if not valid_data:
        print("Không có dữ liệu hợp lệ nào được tìm thấy. Dừng chương trình.")
        return

    # 3. Chuyển đổi dữ liệu sang Pandas DataFrame để dễ dàng thao tác
    records = []
    for model_name, data in valid_data.items():
        if 'metrics' in data:
            for dataset, metrics in data['metrics'].items():
                records.append({'Model': model_name, 'Dataset': dataset, **metrics})
    df = pd.DataFrame(records)

    # 4. Thiết lập style và vẽ các biểu đồ
    setup_matplotlib_style()

    print("\n--- Bắt đầu tạo các figure ---")
    plot_main_performance(df, valid_data)
    plot_stability_analysis(df)
    plot_summary_heatmap(df, valid_data)

    plt.close('all')  # Đóng tất cả các cửa sổ plot đang mở
    print("\nHoàn thành! Các figure chất lượng cao đã được lưu trong thư mục '{}'.".format(CONFIG['output_dir']))


if __name__ == '__main__':
    main()