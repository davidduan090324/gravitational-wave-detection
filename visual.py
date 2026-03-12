import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings
from scipy import signal
from nnAudio.Spectrogram import CQT1992v2
import torch

warnings.filterwarnings("ignore")
plt.style.use('dark_background')

train_labels = pd.read_csv(r"D:\Code\g2net-gravitational-wave-detection\training_labels.csv")
sample_submission = pd.read_csv(r"D:\Code\g2net-gravitational-wave-detection\sample_submission.csv")

COLORS = {
    'raw': '#FFFFFF',
    'hanford': '#00D4FF',
    'livingston': '#FF6B6B', 
    'virgo': '#50FA7B',
    'accent': '#BD93F9',
    'grid': '#44475A',
    'bandpass': '#00D4FF',
    'highpass': '#FF6B6B',
    'lowpass': '#50FA7B',
    'narrow_band': '#FFB86C'
}

FILTER_CONFIGS = {
    'raw': {
        'name': 'Raw Signal (No Filter)',
        'params': None,
        'color': '#FFFFFF'
    },
    'bandpass': {
        'name': 'Bandpass (20-500Hz)',
        'params': {'N': 4, 'Wn': (20, 500), 'btype': 'bandpass', 'fs': 2048},
        'color': '#00D4FF'
    },
    'highpass': {
        'name': 'Highpass (20Hz)',
        'params': {'N': 4, 'Wn': 20, 'btype': 'highpass', 'fs': 2048},
        'color': '#FF6B6B'
    },
    'lowpass': {
        'name': 'Lowpass (500Hz)',
        'params': {'N': 4, 'Wn': 500, 'btype': 'lowpass', 'fs': 2048},
        'color': '#50FA7B'
    },
    'narrow_band': {
        'name': 'Narrow Band (30-200Hz)',
        'params': {'N': 4, 'Wn': (30, 200), 'btype': 'bandpass', 'fs': 2048},
        'color': '#FFB86C'
    }
}

def id2path(idx, is_train=True):
    path = r"D:\Code\g2net-gravitational-wave-detection"
    if is_train:
        path += "/train/" + idx[0] + "/" + idx[1] + "/" + idx[2] + "/" + idx + ".npy"
    else:
        path += "/test/" + idx[0] + "/" + idx[1] + "/" + idx[2] + "/" + idx + ".npy"
    return path

def apply_filter(wave, filter_type='bandpass'):
    """应用滤波器，保留信号特征"""
    if filter_type == 'raw':
        normalized = wave / (np.max(np.abs(wave)) + 1e-8)
        return normalized
    
    config = FILTER_CONFIGS[filter_type]['params']
    b, a = signal.butter(**config)
    
    # 先滤波
    filtered = signal.filtfilt(b, a, wave)
    
    # 应用轻微的窗函数（只在边缘衰减）
    window = signal.windows.tukey(len(filtered), alpha=0.1)
    filtered = filtered * window
    
    # 归一化到 [-1, 1]
    max_val = np.max(np.abs(filtered))
    if max_val > 1e-10:
        filtered = filtered / max_val
    
    return filtered

def create_spectrogram(wave, transform):
    wave_tensor = torch.from_numpy(wave.astype(np.float32))
    image = transform(wave_tensor)
    image = np.array(image)
    return np.transpose(image, (1, 2, 0))

def plot_waveform(ax, data, color, title, show_envelope=True, alpha=0.9):
    time = np.linspace(0, 2, len(data))
    ax.plot(time, data, color=color, linewidth=0.6, alpha=alpha)
    if show_envelope and np.std(data) > 0.01:
        analytic = signal.hilbert(data)
        envelope = np.abs(analytic)
        ax.fill_between(time, -envelope, envelope, color=color, alpha=0.1)
    ax.set_facecolor('#1E1E2E')
    ax.set_title(title, fontsize=9, color='white', fontweight='bold', pad=6)
    ax.set_xlim(0, 2)
    
    # 动态设置 y 轴范围
    data_range = np.max(np.abs(data))
    ax.set_ylim(-data_range * 1.3, data_range * 1.3)
    
    ax.grid(True, alpha=0.2, color=COLORS['grid'])
    ax.tick_params(colors='gray', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.5)

def plot_spectrogram(ax, image, title):
    im = ax.imshow(image, aspect='auto', cmap='magma', origin='lower')
    ax.set_facecolor('#1E1E2E')
    ax.set_title(title, fontsize=9, color='white', fontweight='bold', pad=6)
    ax.tick_params(colors='gray', labelsize=6)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.5)
    return im

def visualize_filter_comparison(idx, is_train=True):
    """比较不同滤波器对同一信号的效果，包含原始信号"""
    wave_data = np.load(id2path(idx, is_train))
    
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0D0D1A')
    
    # 5行（raw + 4个滤波器）x 4列（3个探测器 + 频谱图）
    filter_types = ['raw', 'bandpass', 'highpass', 'lowpass', 'narrow_band']
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.25)
    
    detector_names = ['LIGO Hanford', 'LIGO Livingston', 'Virgo']
    detector_indices = [1, 2, 0]  # 数据中的索引顺序
    
    transform = CQT1992v2(sr=2048, hop_length=64, fmin=20, fmax=500)
    
    for row, filter_type in enumerate(filter_types):
        filter_info = FILTER_CONFIGS[filter_type]
        filter_color = filter_info['color']
        
        for col, (det_name, det_idx) in enumerate(zip(detector_names, detector_indices)):
            ax = fig.add_subplot(gs[row, col])
            raw_wave = wave_data[det_idx, :]
            filtered_wave = apply_filter(raw_wave, filter_type)
            
            title = f"{det_name}\n{filter_info['name']}"
            plot_waveform(ax, filtered_wave, filter_color, title, show_envelope=(filter_type != 'raw'))
        
        # 频谱图
        ax_spec = fig.add_subplot(gs[row, 3])
        combined_wave = np.concatenate([
            apply_filter(wave_data[det_idx, :], filter_type) 
            for det_idx in detector_indices
        ])
        spec_image = create_spectrogram(combined_wave, transform)
        plot_spectrogram(ax_spec, spec_image, f"Spectrogram\n{filter_info['name']}")
    
    status = "GW Detected" if train_labels[train_labels['id'] == idx]['target'].values[0] == 1 else "No GW"
    fig.suptitle(f'Gravitational Wave Analysis - Filter Comparison\nSample: {idx} | Status: {status}', 
                 fontsize=14, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def visualize_single_detector_filters(idx, detector_idx=1, is_train=True):
    """单个探测器的所有滤波效果叠加对比"""
    wave_data = np.load(id2path(idx, is_train))
    raw_wave = wave_data[detector_idx, :]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0D0D1A')
    
    filter_types = ['raw', 'bandpass', 'highpass', 'lowpass', 'narrow_band']
    
    # 第一行：每个滤波器单独显示
    for i, filter_type in enumerate(filter_types):
        if i < 3:
            ax = axes[0, i]
        else:
            ax = axes[1, i - 3]
        
        filter_info = FILTER_CONFIGS[filter_type]
        filtered = apply_filter(raw_wave, filter_type)
        plot_waveform(ax, filtered, filter_info['color'], filter_info['name'], show_envelope=True)
    
    # 最后一个子图：所有滤波结果叠加
    ax_overlay = axes[1, 2]
    time = np.linspace(0, 2, len(raw_wave))
    
    for filter_type in filter_types:
        filter_info = FILTER_CONFIGS[filter_type]
        filtered = apply_filter(raw_wave, filter_type)
        alpha = 0.5 if filter_type == 'raw' else 0.7
        lw = 1.0 if filter_type == 'raw' else 0.6
        ax_overlay.plot(time, filtered, color=filter_info['color'], 
                       linewidth=lw, alpha=alpha, label=filter_info['name'])
    
    ax_overlay.set_facecolor('#1E1E2E')
    ax_overlay.set_title('All Filters Overlay', fontsize=9, color='white', fontweight='bold', pad=6)
    ax_overlay.set_xlim(0, 2)
    ax_overlay.set_ylim(-1.3, 1.3)
    ax_overlay.grid(True, alpha=0.2, color=COLORS['grid'])
    ax_overlay.legend(loc='upper right', fontsize=7, framealpha=0.3)
    ax_overlay.tick_params(colors='gray', labelsize=6)
    
    detector_name = ['Virgo', 'LIGO Hanford', 'LIGO Livingston'][detector_idx]
    status = "GW Detected" if train_labels[train_labels['id'] == idx]['target'].values[0] == 1 else "No GW"
    fig.suptitle(f'Filter Comparison - {detector_name}\nSample: {idx} | Status: {status}', 
                 fontsize=14, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def visualize_gw_comparison(target_idx, no_target_idx):
    """对比有引力波和无引力波的信号，包含原始信号"""
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#0D0D1A')
    
    # 5行 x 6列布局
    gs = GridSpec(5, 6, figure=fig, hspace=0.45, wspace=0.3, 
                  width_ratios=[1, 1, 0.15, 1, 1, 0.15])
    
    transform = CQT1992v2(sr=2048, hop_length=64, fmin=20, fmax=500)
    detector_names = ['Hanford', 'Livingston', 'Virgo']
    detector_indices = [1, 2, 0]
    
    for sample_idx, (idx, label, base_col) in enumerate([
        (target_idx, "GW DETECTED", 0),
        (no_target_idx, "NO GW", 3)
    ]):
        wave_data = np.load(id2path(idx, is_train=True))
        main_color = COLORS['hanford'] if label == "GW DETECTED" else COLORS['livingston']
        
        for row, (det_name, det_idx) in enumerate(zip(detector_names, detector_indices)):
            raw_wave = wave_data[det_idx, :]
            
            # 左列：原始信号
            ax1 = fig.add_subplot(gs[row, base_col])
            raw_normalized = apply_filter(raw_wave, 'raw')
            plot_waveform(ax1, raw_normalized, COLORS['raw'], 
                         f"{det_name} - Raw", show_envelope=False)
            
            # 右列：带通滤波后
            ax2 = fig.add_subplot(gs[row, base_col + 1])
            filtered = apply_filter(raw_wave, 'bandpass')
            plot_waveform(ax2, filtered, main_color, 
                         f"{det_name} - Bandpass", show_envelope=True)
        
        # 底部：频谱图对比（原始 vs 滤波后）
        ax_spec_raw = fig.add_subplot(gs[3, base_col:base_col+2])
        combined_raw = np.concatenate([
            apply_filter(wave_data[det_idx, :], 'raw') 
            for det_idx in detector_indices
        ])
        spec_raw = create_spectrogram(combined_raw, transform)
        plot_spectrogram(ax_spec_raw, spec_raw, f"Raw Spectrogram - {label}")
        
        ax_spec_filt = fig.add_subplot(gs[4, base_col:base_col+2])
        combined_filt = np.concatenate([
            apply_filter(wave_data[det_idx, :], 'bandpass') 
            for det_idx in detector_indices
        ])
        spec_filt = create_spectrogram(combined_filt, transform)
        plot_spectrogram(ax_spec_filt, spec_filt, f"Filtered Spectrogram - {label}")
    
    fig.suptitle('Gravitational Wave Detection: Raw vs Filtered Signal Comparison', 
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

if __name__ == "__main__":
    targets = train_labels[train_labels["target"] == 1]["id"].head(2).values
    no_targets = train_labels[train_labels["target"] == 0]["id"].head(2).values
    
    print("Generating Filter Comparison - GW Detected...")
    fig1 = visualize_filter_comparison(targets[0], is_train=True)
    
    print("Generating Filter Comparison - No GW...")
    fig2 = visualize_filter_comparison(no_targets[0], is_train=True)
    
    print("Generating GW vs No-GW Comparison (Raw + Filtered)...")
    fig3 = visualize_gw_comparison(targets[0], no_targets[0])
    
    plt.show()
