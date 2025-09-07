import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# KHU VỰC CẤU HÌNH (CHỈ CẦN CHỈNH SỬA Ở ĐÂY)
# ==============================================================================
CONFIG = {
    # 1. Đường dẫn đến các file log và tên model
    "log_files": [
        'test/res50.txt',
        'test/res50_custom1.txt',
        'test/res50_custom2.txt',
        'test/res50_fan.txt'
    ],
    "model_names": [
        'ResNet-50 (Baseline)',
        'Ours (Method A)',
        'Ours (Method B)',
        'ResNet-50 + FFN'
    ],

    # 2. Cài đặt cho biểu đồ
    "output_dir": "paper_figures",  # Thư mục để lưu các figure
    "formats": ["png", "pdf"],  # Lưu ở cả 2 định dạng
    "dpi": 300,  # Độ phân giải cho file PNG
    "palette": "viridis",  # Bảng màu (các lựa chọn khác: 'colorblind', 'Set2', 'Paired')
    "font_family": "serif",  # Font chữ cho bài báo (e.g., 'Times New Roman')
    "font_size": 12  # Cỡ chữ cơ bản
}


# ==============================================================================

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

    # ***** ĐÃ SỬA LỖI Ở ĐÂY *****
    # Đổi tên khóa từ 'EER' thành 'EER (%)' để nhất quán
    data = {m[0]: {'Accuracy': float(m[1]), 'Accuracy-Flip': float(m[2]),
                   'VAL @ FAR=1e-3': float(m[3]), 'EER (%)': float(m[4])} for m in matches_metrics}

    avg_infer_time = np.mean([float(t) for t in matches_time]) if matches_time else 0
    return {'metrics': data, 'avg_infer_time': avg_infer_time}


def setup_matplotlib_style():
    """Thiết lập style chung cho tất cả các biểu đồ."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': CONFIG['font_family'],
        'font.size': CONFIG['font_size'],
        'axes.labelsize': CONFIG['font_size'] + 2,
        'axes.titlesize': CONFIG['font_size'] + 4,
        'xtick.labelsize': CONFIG['font_size'],
        'ytick.labelsize': CONFIG['font_size'],
        'legend.fontsize': CONFIG['font_size'],
        'figure.titlesize': CONFIG['font_size'] + 6,
    })


def save_figure(fig, filename):
    """Lưu figure với các định dạng đã cấu hình."""
    for fmt in CONFIG['formats']:
        full_path = os.path.join(CONFIG['output_dir'], f"{filename}.{fmt}")
        fig.savefig(full_path, dpi=CONFIG['dpi'], bbox_inches='tight')
    print(f"Đã lưu: {filename}.[{', '.join(CONFIG['formats'])}]")


def create_paper_figures(all_data):
    """Tạo các figure tổng hợp, chất lượng cao cho bài báo."""
    # 1. Chuẩn bị DataFrame
    records = []
    for model_name, data in all_data.items():
        if data and 'metrics' in data:
            for dataset, metrics in data['metrics'].items():
                records.append({'Model': model_name, 'Dataset': dataset, **metrics})
    df = pd.DataFrame(records)
    if df.empty: return
    dataset_order = df['Dataset'].unique()

    # 2. Figure 1: Hiệu năng chính (2x2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 14))
    fig1.suptitle('Main Performance Comparison Across Datasets', weight='bold')

    # (a) Accuracy-Flip
    sns.barplot(data=df, x='Dataset', y='Accuracy-Flip', hue='Model', ax=axes1[0, 0], order=dataset_order,
                palette=CONFIG['palette'])
    axes1[0, 0].set_title('(a) Verification Accuracy (with flip)', weight='bold')
    axes1[0, 0].set_ylabel('Accuracy')
    axes1[0, 0].set_ylim(bottom=max(0, df['Accuracy-Flip'].min() - 0.05), top=1.01)
    for container in axes1[0, 0].containers: axes1[0, 0].bar_label(container, fmt='%.3f', size=8, rotation=90,
                                                                   padding=5)
    axes1[0, 0].get_legend().remove()

    # (b) EER (%)
    sns.barplot(data=df, x='Dataset', y='EER (%)', hue='Model', ax=axes1[0, 1], order=dataset_order,
                palette=CONFIG['palette'])
    axes1[0, 1].set_title('(b) Equal Error Rate (EER)', weight='bold')
    axes1[0, 1].set_ylabel('EER (%) (Lower is better)')
    axes1[0, 1].get_legend().remove()

    # (c) VAL @ FAR=1e-3
    sns.barplot(data=df, x='Dataset', y='VAL @ FAR=1e-3', hue='Model', ax=axes1[1, 0], order=dataset_order,
                palette=CONFIG['palette'])
    axes1[1, 0].set_title('(c) Verification Rate at FAR=1e-3', weight='bold')
    axes1[1, 0].set_ylabel('Verification Rate')
    axes1[1, 0].get_legend().remove()

    # (d) Speed vs. Accuracy Trade-off
    infer_times = [d['avg_infer_time'] for d in all_data.values() if d]
    avg_accuracies = df.groupby('Model')['Accuracy-Flip'].mean().reindex(CONFIG['model_names'])
    df_tradeoff = pd.DataFrame(
        {'Model': CONFIG['model_names'], 'Avg Infer Time (s)': infer_times, 'Avg Accuracy': avg_accuracies})
    sns.scatterplot(data=df_tradeoff, x='Avg Infer Time (s)', y='Avg Accuracy', hue='Model', s=200, ax=axes1[1, 1],
                    palette=CONFIG['palette'], legend=False)
    for i in df_tradeoff.index: axes1[1, 1].text(df_tradeoff.loc[i, 'Avg Infer Time (s)'],
                                                 df_tradeoff.loc[i, 'Avg Accuracy'] + 0.001,
                                                 df_tradeoff.loc[i, 'Model'], ha='center', size=10)
    axes1[1, 1].set_title('(d) Speed-Accuracy Trade-off', weight='bold')
    axes1[1, 1].set_xlabel('Average Inference Time (s)')
    axes1[1, 1].set_ylabel('Average Accuracy')

    handles, labels = axes1[0, 0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower center', ncol=len(CONFIG['model_names']), bbox_to_anchor=(0.5, 0.0))
    fig1.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_figure(fig1, 'figure_1_main_performance')

    # 3. Figure 2: Phân tích độ ổn định (1x2)
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
    fig2.suptitle('Performance Stability and Augmentation Analysis', weight='bold')

    # (a) Performance Distribution
    sns.boxplot(data=df, x='Model', y='Accuracy-Flip', ax=axes2[0], palette=CONFIG['palette'])
    sns.stripplot(data=df, x='Model', y='Accuracy-Flip', color=".25", size=5, ax=axes2[0])
    axes2[0].set_title('(a) Performance Distribution Across Datasets', weight='bold')
    axes2[0].set_ylabel('Accuracy-Flip')
    axes2[0].tick_params(axis='x', rotation=15)

    # (b) Flip Improvement
    df['Flip Improvement (%)'] = (df['Accuracy-Flip'] - df['Accuracy']) * 100
    sns.barplot(data=df, x='Dataset', y='Flip Improvement (%)', hue='Model', ax=axes2[1], order=dataset_order,
                palette=CONFIG['palette'])
    axes2[1].set_title('(b) Accuracy Gain from Flip Augmentation', weight='bold')
    axes2[1].set_ylabel('Improvement (%)')

    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig2, 'figure_2_stability_analysis')

    # 4. Figure 3: Bảng tổng hợp (Heatmap chia 2 nhóm)
    df_summary = df.groupby('Model').mean(numeric_only=True).reindex(CONFIG['model_names'])
    df_summary['Avg Infer Time (s)'] = [d['avg_infer_time'] for d in all_data.values() if d]

    # Chia 2 nhóm metrics
    higher_is_better = ['Accuracy-Flip', 'VAL @ FAR=1e-3']
    lower_is_better = ['EER (%)', 'Avg Infer Time (s)']

    fig3, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap cho Higher is Better
    sns.heatmap(df_summary[higher_is_better],
                annot=True, fmt=".4f", cmap="Greens", linewidths=.5, ax=ax_left, cbar=True)
    ax_left.set_title("Metrics where Higher is Better", weight='bold')
    ax_left.set_ylabel("Model")

    # Heatmap cho Lower is Better
    sns.heatmap(df_summary[lower_is_better],
                annot=True, fmt=".2f", cmap="Reds_r", linewidths=.5, ax=ax_right, cbar=True)
    ax_right.set_title("Metrics where Lower is Better", weight='bold')
    ax_right.set_ylabel("")

    fig3.suptitle("Average Performance Summary", fontsize=16, weight='bold')
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig3, "figure_3_summary_heatmap")

    plt.close('all')


def main():
    """Hàm chính điều khiển toàn bộ quy trình."""
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
        print(f"Đã tạo thư mục: '{CONFIG['output_dir']}'")

    all_data = {name: parse_log_file(fp) for name, fp in zip(CONFIG['model_names'], CONFIG['log_files'])}

    valid_data = {k: v for k, v in all_data.items() if v}
    if not valid_data:
        print("Không có dữ liệu hợp lệ nào được tìm thấy. Dừng chương trình.")
        return

    setup_matplotlib_style()
    create_paper_figures(valid_data)

    print("\nHoàn thành! Các figure chất lượng cao đã được lưu trong thư mục '{}'.".format(CONFIG['output_dir']))


if __name__ == '__main__':
    main()