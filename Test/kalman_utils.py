"""
卡尔曼滤波工具函数库
包含数据处理、评估指标和可视化工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import seaborn as sns
from scipy import signal
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 数据预处理函数 ====================

def load_and_prepare_data(file_path: str, filter_cols: List[str],
                          time_column: str = 'date_time') -> Tuple[pd.DataFrame, np.ndarray]:
    """
    加载数据并准备为卡尔曼滤波格式

    参数:
    ----------
    file_path : str
        数据文件路径
    filter_cols : List[str]
        需要滤波的特征列名
    time_column : str, optional
        时间戳列名

    返回:
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (完整数据框, 观测序列数组)
    """
    print(f"正在加载数据: {file_path}")
    df = pd.read_excel(file_path)

    # 检查必要的列
    missing_cols = [col for col in filter_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下特征列在数据中不存在: {missing_cols}")

    # 提取观测序列
    observations = df[filter_cols].values.astype(np.float64)

    print(f"数据加载完成:")
    print(f"  样本数: {len(df)}")
    print(f"  特征数: {len(filter_cols)}")
    print(f"  时间范围: {df[time_column].min()} 到 {df[time_column].max()}")

    # 数据统计信息
    for i, col in enumerate(filter_cols):
        col_data = df[col].values
        print(f"  {col}: 均值={col_data.mean():.4f}, 标准差={col_data.std():.4f}, "
              f"范围=[{col_data.min():.4f}, {col_data.max():.4f}]")

    return df, observations


def extract_timestamps(df: pd.DataFrame, time_column: str = 'date_time') -> np.ndarray:
    """
    从数据框中提取时间戳序列

    参数:
    ----------
    df : pd.DataFrame
        包含时间戳的数据框
    time_column : str
        时间戳列名

    返回:
    -------
    np.ndarray
        时间戳序列（转换为数值）
    """
    if time_column not in df.columns:
        warnings.warn(f"时间戳列 '{time_column}' 不存在，使用索引作为时间")
        return np.arange(len(df))

    # 转换为datetime
    timestamps = pd.to_datetime(df[time_column])

    # 转换为数值（秒）
    time_diff = timestamps - timestamps.iloc[0]
    time_seconds = time_diff.dt.total_seconds().values

    # 检查采样间隔
    if len(time_seconds) > 1:
        intervals = np.diff(time_seconds)
        avg_interval = intervals.mean()
        std_interval = intervals.std()

        print(f"时间戳分析:")
        print(f"  平均采样间隔: {avg_interval:.2f} 秒 ({avg_interval/60:.2f} 分钟)")
        print(f"  采样间隔标准差: {std_interval:.2f} 秒")
        print(f"  时间跨度: {time_seconds[-1]/3600:.2f} 小时")

        if std_interval > avg_interval * 0.1:  # 间隔变化超过10%
            warnings.warn(f"采样间隔不均匀，标准差({std_interval:.2f}s)较大")

    return time_seconds


def validate_data_quality(df: pd.DataFrame, filter_cols: List[str]) -> Dict:
    """
    验证数据质量

    参数:
    ----------
    df : pd.DataFrame
        数据框
    filter_cols : List[str]
        需要检查的特征列

    返回:
    -------
    Dict
        数据质量报告
    """
    report = {
        'missing_values': {},
        'outliers': {},
        'statistics': {},
        'warnings': []
    }

    for col in filter_cols:
        if col not in df.columns:
            report['warnings'].append(f"列 '{col}' 不存在")
            continue

        data = df[col].values

        # 缺失值检查
        missing_count = pd.isna(data).sum()
        missing_percent = missing_count / len(data) * 100
        report['missing_values'][col] = {
            'count': missing_count,
            'percent': missing_percent
        }

        if missing_percent > 5:
            report['warnings'].append(f"列 '{col}' 缺失值较多: {missing_percent:.2f}%")

        # 统计信息
        if missing_count < len(data):
            valid_data = data[~pd.isna(data)]
            report['statistics'][col] = {
                'mean': float(np.mean(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'median': float(np.median(valid_data))
            }

            # 异常值检查（基于3σ原则）
            mean = report['statistics'][col]['mean']
            std = report['statistics'][col]['std']
            if std > 0:
                outliers = np.sum(np.abs(valid_data - mean) > 3 * std)
                outlier_percent = outliers / len(valid_data) * 100
                report['outliers'][col] = {
                    'count': int(outliers),
                    'percent': outlier_percent
                }

                if outlier_percent > 1:
                    report['warnings'].append(f"列 '{col}' 异常值较多: {outlier_percent:.2f}%")

    # 汇总报告
    total_samples = len(df)
    total_missing = sum([r['count'] for r in report['missing_values'].values()])
    total_missing_percent = total_missing / (total_samples * len(filter_cols)) * 100

    report['summary'] = {
        'total_samples': total_samples,
        'total_features': len(filter_cols),
        'total_missing': total_missing,
        'total_missing_percent': total_missing_percent,
        'warning_count': len(report['warnings'])
    }

    return report


# ==================== 评估指标函数 ====================

def calculate_smoothness(original: np.ndarray, filtered: np.ndarray) -> Dict:
    """
    计算数据平滑度改进

    参数:
    ----------
    original : np.ndarray
        原始数据，形状为(n_samples, n_features)
    filtered : np.ndarray
        滤波后数据，形状相同

    返回:
    -------
    Dict
        平滑度评估结果
    """
    n_samples, n_features = original.shape

    results = {
        'overall': {},
        'per_feature': [],
        'improvement_summary': {}
    }

    total_improvement = 0
    features_with_improvement = 0

    for i in range(n_features):
        orig_series = original[:, i]
        filt_series = filtered[:, i]

        # 移除NaN值
        mask = ~(np.isnan(orig_series) | np.isnan(filt_series))
        if np.sum(mask) < 10:  # 至少需要10个有效点
            continue

        orig_series = orig_series[mask]
        filt_series = filt_series[mask]

        # 计算差分标准差（平滑度指标）
        orig_diff_std = np.std(np.diff(orig_series))
        filt_diff_std = np.std(np.diff(filt_series))

        # 计算平滑度改进
        if orig_diff_std > 0:
            smoothness_improvement = (orig_diff_std - filt_diff_std) / orig_diff_std * 100
        else:
            smoothness_improvement = 0 if filt_diff_std == 0 else -100

        # 计算绝对平滑度（差分标准差的倒数）
        orig_smoothness = 1.0 / (orig_diff_std + 1e-10)
        filt_smoothness = 1.0 / (filt_diff_std + 1e-10)

        feature_result = {
            'feature_index': i,
            'original_diff_std': float(orig_diff_std),
            'filtered_diff_std': float(filt_diff_std),
            'smoothness_improvement_percent': float(smoothness_improvement),
            'original_smoothness': float(orig_smoothness),
            'filtered_smoothness': float(filt_smoothness),
            'samples_used': int(np.sum(mask))
        }

        results['per_feature'].append(feature_result)
        total_improvement += smoothness_improvement
        if smoothness_improvement > 0:
            features_with_improvement += 1

    # 计算总体指标
    if results['per_feature']:
        avg_improvement = total_improvement / len(results['per_feature'])
        improvement_ratio = features_with_improvement / len(results['per_feature'])

        results['overall'] = {
            'average_smoothness_improvement': float(avg_improvement),
            'features_with_improvement': int(features_with_improvement),
            'total_features_evaluated': len(results['per_feature']),
            'improvement_ratio': float(improvement_ratio)
        }

        # 改进总结
        improvements = [f['smoothness_improvement_percent'] for f in results['per_feature']]
        results['improvement_summary'] = {
            'min': float(np.min(improvements)),
            'max': float(np.max(improvements)),
            'mean': float(np.mean(improvements)),
            'median': float(np.median(improvements)),
            'std': float(np.std(improvements))
        }

    return results


def calculate_snr_improvement(original: np.ndarray, filtered: np.ndarray) -> Dict:
    """
    计算信噪比改进

    参数:
    ----------
    original : np.ndarray
        原始数据
    filtered : np.ndarray
        滤波后数据

    返回:
    -------
    Dict
        信噪比评估结果
    """
    n_samples, n_features = original.shape

    results = {
        'overall': {},
        'per_feature': [],
        'improvement_summary': {}
    }

    total_snr_improvement = 0
    features_with_improvement = 0

    for i in range(n_features):
        orig_series = original[:, i]
        filt_series = filtered[:, i]

        # 移除NaN值
        mask = ~(np.isnan(orig_series) | np.isnan(filt_series))
        if np.sum(mask) < 10:
            continue

        orig_series = orig_series[mask]
        filt_series = filt_series[mask]

        # 估计噪声（原始-滤波）
        noise_estimate = orig_series - filt_series

        # 计算信噪比
        signal_power_orig = np.var(orig_series)
        noise_power_orig = np.var(noise_estimate)

        signal_power_filt = np.var(filt_series)
        noise_power_filt = np.var(noise_estimate)  # 噪声功率不变

        if noise_power_orig > 0:
            snr_original = 10 * np.log10(signal_power_orig / noise_power_orig)
        else:
            snr_original = np.inf

        if noise_power_filt > 0:
            snr_filtered = 10 * np.log10(signal_power_filt / noise_power_filt)
        else:
            snr_filtered = np.inf

        # 计算SNR改进（dB）
        if np.isfinite(snr_original) and np.isfinite(snr_filtered):
            snr_improvement_db = snr_filtered - snr_original
            snr_improvement_ratio = 10 ** (snr_improvement_db / 10)  # 线性比例
        else:
            snr_improvement_db = 0
            snr_improvement_ratio = 1

        feature_result = {
            'feature_index': i,
            'snr_original_db': float(snr_original),
            'snr_filtered_db': float(snr_filtered),
            'snr_improvement_db': float(snr_improvement_db),
            'snr_improvement_ratio': float(snr_improvement_ratio),
            'signal_power_original': float(signal_power_orig),
            'signal_power_filtered': float(signal_power_filt),
            'noise_power': float(noise_power_orig),
            'samples_used': int(np.sum(mask))
        }

        results['per_feature'].append(feature_result)
        total_snr_improvement += snr_improvement_db
        if snr_improvement_db > 0:
            features_with_improvement += 1

    # 计算总体指标
    if results['per_feature']:
        avg_snr_improvement = total_snr_improvement / len(results['per_feature'])
        improvement_ratio = features_with_improvement / len(results['per_feature'])

        results['overall'] = {
            'average_snr_improvement_db': float(avg_snr_improvement),
            'features_with_snr_improvement': int(features_with_improvement),
            'total_features_evaluated': len(results['per_feature']),
            'improvement_ratio': float(improvement_ratio)
        }

        # 改进总结
        improvements = [f['snr_improvement_db'] for f in results['per_feature']]
        results['improvement_summary'] = {
            'min': float(np.min(improvements)),
            'max': float(np.max(improvements)),
            'mean': float(np.mean(improvements)),
            'median': float(np.median(improvements)),
            'std': float(np.std(improvements))
        }

    return results


def calculate_correlation_preservation(original: np.ndarray, filtered: np.ndarray,
                                       target_idx: Optional[int] = None) -> Dict:
    """
    计算滤波前后特征与目标变量的相关性保持情况

    参数:
    ----------
    original : np.ndarray
        原始数据
    filtered : np.ndarray
        滤波后数据
    target_idx : int, optional
        目标变量索引（默认为最后一列）

    返回:
    -------
    Dict
        相关性保持评估结果
    """
    n_samples, n_features = original.shape

    if target_idx is None:
        target_idx = n_features - 1  # 默认最后一列为目标变量

    if target_idx >= n_features:
        raise ValueError(f"目标变量索引 {target_idx} 超出特征范围 [0, {n_features-1}]")

    results = {
        'target_feature_index': target_idx,
        'correlation_changes': [],
        'preservation_summary': {}
    }

    target_original = original[:, target_idx]
    target_filtered = filtered[:, target_idx]

    correlation_changes = []

    for i in range(n_features):
        if i == target_idx:
            continue  # 跳过目标变量自身

        # 移除NaN值
        mask = ~(np.isnan(original[:, i]) | np.isnan(filtered[:, i]) |
                 np.isnan(target_original) | np.isnan(target_filtered))
        if np.sum(mask) < 10:
            continue

        feature_orig = original[mask, i]
        feature_filt = filtered[mask, i]
        target_orig = target_original[mask]
        target_filt = target_filtered[mask]

        # 计算相关系数
        corr_original = np.corrcoef(feature_orig, target_orig)[0, 1]
        corr_filtered = np.corrcoef(feature_filt, target_filt)[0, 1]

        # 计算相关性变化
        corr_change = corr_filtered - corr_original
        corr_change_percent = (corr_change / abs(corr_original)) * 100 if abs(corr_original) > 0.01 else 0

        # 计算相关性保持度
        if abs(corr_original) > 0.01:
            preservation = 1 - abs(corr_change) / abs(corr_original)
        else:
            preservation = 1.0 if abs(corr_change) < 0.01 else 0.0

        feature_result = {
            'feature_index': i,
            'correlation_original': float(corr_original),
            'correlation_filtered': float(corr_filtered),
            'correlation_change': float(corr_change),
            'correlation_change_percent': float(corr_change_percent),
            'correlation_preservation': float(preservation),
            'samples_used': int(np.sum(mask))
        }

        results['correlation_changes'].append(feature_result)
        correlation_changes.append(corr_change)

    # 计算汇总统计
    if results['correlation_changes']:
        preservations = [r['correlation_preservation'] for r in results['correlation_changes']]
        changes = [r['correlation_change'] for r in results['correlation_changes']]

        results['preservation_summary'] = {
            'average_preservation': float(np.mean(preservations)),
            'median_preservation': float(np.median(preservations)),
            'min_preservation': float(np.min(preservations)),
            'max_preservation': float(np.max(preservations)),
            'average_change': float(np.mean(changes)),
            'median_change': float(np.median(changes)),
            'std_change': float(np.std(changes)),
            'features_evaluated': len(results['correlation_changes'])
        }

    return results


def calculate_mse_improvement(original: np.ndarray, filtered: np.ndarray,
                              reference: Optional[np.ndarray] = None) -> Dict:
    """
    计算均方误差改进（如果有参考真值）

    参数:
    ----------
    original : np.ndarray
        原始数据
    filtered : np.ndarray
        滤波后数据
    reference : np.ndarray, optional
        参考真值数据（如果可用）

    返回:
    -------
    Dict
        MSE评估结果
    """
    n_samples, n_features = original.shape

    results = {
        'overall': {},
        'per_feature': [],
        'improvement_summary': {}
    }

    if reference is not None:
        if reference.shape != original.shape:
            raise ValueError(f"参考数据形状 {reference.shape} 与原始数据 {original.shape} 不匹配")

        total_mse_improvement = 0
        features_with_improvement = 0

        for i in range(n_features):
            # 移除NaN值
            mask = ~(np.isnan(original[:, i]) | np.isnan(filtered[:, i]) | np.isnan(reference[:, i]))
            if np.sum(mask) < 10:
                continue

            orig_series = original[mask, i]
            filt_series = filtered[mask, i]
            ref_series = reference[mask, i]

            # 计算MSE
            mse_original = np.mean((orig_series - ref_series) ** 2)
            mse_filtered = np.mean((filt_series - ref_series) ** 2)

            # 计算改进
            if mse_original > 0:
                mse_improvement = (mse_original - mse_filtered) / mse_original * 100
            else:
                mse_improvement = 0 if mse_filtered == 0 else -100

            feature_result = {
                'feature_index': i,
                'mse_original': float(mse_original),
                'mse_filtered': float(mse_filtered),
                'mse_improvement_percent': float(mse_improvement),
                'rmse_original': float(np.sqrt(mse_original)),
                'rmse_filtered': float(np.sqrt(mse_filtered)),
                'samples_used': int(np.sum(mask))
            }

            results['per_feature'].append(feature_result)
            total_mse_improvement += mse_improvement
            if mse_improvement > 0:
                features_with_improvement += 1

        # 计算总体指标
        if results['per_feature']:
            avg_improvement = total_mse_improvement / len(results['per_feature'])
            improvement_ratio = features_with_improvement / len(results['per_feature'])

            results['overall'] = {
                'average_mse_improvement': float(avg_improvement),
                'features_with_improvement': int(features_with_improvement),
                'total_features_evaluated': len(results['per_feature']),
                'improvement_ratio': float(improvement_ratio)
            }

            # 改进总结
            improvements = [f['mse_improvement_percent'] for f in results['per_feature']]
            results['improvement_summary'] = {
                'min': float(np.min(improvements)),
                'max': float(np.max(improvements)),
                'mean': float(np.mean(improvements)),
                'median': float(np.median(improvements)),
                'std': float(np.std(improvements))
            }

    return results


# ==================== 可视化函数 ====================

def plot_time_series_comparison(original: np.ndarray, filtered: np.ndarray,
                                feature_names: List[str], feature_indices: Optional[List[int]] = None,
                                timestamps: Optional[np.ndarray] = None,
                                save_path: Optional[str] = None):
    """
    绘制时间序列对比图

    参数:
    ----------
    original : np.ndarray
        原始数据
    filtered : np.ndarray
        滤波后数据
    feature_names : List[str]
        特征名称列表
    feature_indices : List[int], optional
        要绘制的特征索引列表（默认绘制前4个）
    timestamps : np.ndarray, optional
        时间戳序列
    save_path : str, optional
        保存图像的路径
    """
    n_samples, n_features = original.shape

    if feature_indices is None:
        feature_indices = list(range(min(4, n_features)))

    n_plots = len(feature_indices)
    if n_plots == 0:
        print("没有特征可绘制")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    if timestamps is None:
        x = np.arange(n_samples)
        xlabel = '样本索引'
    else:
        x = timestamps
        xlabel = '时间 (秒)'

    for idx, (ax, feat_idx) in enumerate(zip(axes, feature_indices)):
        if feat_idx >= n_features:
            continue

        # 提取数据
        orig_data = original[:, feat_idx]
        filt_data = filtered[:, feat_idx]

        # 移除NaN值
        mask = ~(np.isnan(orig_data) | np.isnan(filt_data))
        if np.sum(mask) < 10:
            continue

        x_plot = x[mask] if timestamps is not None else np.arange(np.sum(mask))
        orig_plot = orig_data[mask]
        filt_plot = filt_data[mask]

        # 绘制
        ax.plot(x_plot, orig_plot, 'b-', alpha=0.5, linewidth=1, label='原始数据')
        ax.plot(x_plot, filt_plot, 'r-', alpha=0.8, linewidth=1.5, label='卡尔曼滤波')

        # 计算并显示改进
        orig_diff_std = np.std(np.diff(orig_plot))
        filt_diff_std = np.std(np.diff(filt_plot))
        if orig_diff_std > 0:
            improvement = (orig_diff_std - filt_diff_std) / orig_diff_std * 100
            ax.text(0.02, 0.95, f'平滑度改进: {improvement:.1f}%',
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(xlabel)
        ax.set_ylabel('数值')
        ax.set_title(f'{feature_names[feat_idx]} - 时间序列对比')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"时间序列对比图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_frequency_domain_comparison(original: np.ndarray, filtered: np.ndarray,
                                     feature_names: List[str], sampling_rate: float = 1/60,
                                     feature_indices: Optional[List[int]] = None,
                                     save_path: Optional[str] = None):
    """
    绘制频域对比图（功率谱密度）

    参数:
    ----------
    original : np.ndarray
        原始数据
    filtered : np.ndarray
        滤波后数据
    feature_names : List[str]
        特征名称列表
    sampling_rate : float
        采样率（Hz），默认1/60 Hz（1分钟采样）
    feature_indices : List[int], optional
        要绘制的特征索引列表
    save_path : str, optional
        保存图像的路径
    """
    n_samples, n_features = original.shape

    if feature_indices is None:
        feature_indices = list(range(min(4, n_features)))

    n_plots = len(feature_indices)
    if n_plots == 0:
        print("没有特征可绘制")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    # 计算频率轴
    freqs = np.fft.rfftfreq(n_samples, d=sampling_rate)

    for idx, (ax, feat_idx) in enumerate(zip(axes, feature_indices)):
        if feat_idx >= n_features:
            continue

        # 提取数据
        orig_data = original[:, feat_idx]
        filt_data = filtered[:, feat_idx]

        # 移除NaN值（用前后值填充）
        orig_data = pd.Series(orig_data).fillna(method='ffill').fillna(method='bfill').values
        filt_data = pd.Series(filt_data).fillna(method='ffill').fillna(method='bfill').values

        # 计算功率谱密度
        orig_fft = np.abs(np.fft.rfft(orig_data - np.mean(orig_data))) ** 2
        filt_fft = np.abs(np.fft.rfft(filt_data - np.mean(filt_data))) ** 2

        # 归一化
        orig_fft = orig_fft / np.max(orig_fft)
        filt_fft = filt_fft / np.max(filt_fft)

        # 绘制
        ax.plot(freqs, orig_fft, 'b-', alpha=0.6, linewidth=1, label='原始数据')
        ax.plot(freqs, filt_fft, 'r-', alpha=0.8, linewidth=1.5, label='卡尔曼滤波')

        # 标记主要频率成分
        if len(freqs) > 10:
            # 找到原始数据的主要峰值
            peaks_orig, _ = signal.find_peaks(orig_fft, height=0.1)
            if len(peaks_orig) > 0:
                for peak in peaks_orig[:3]:  # 显示前3个主要峰值
                    ax.plot(freqs[peak], orig_fft[peak], 'bo', markersize=6)
                    ax.text(freqs[peak], orig_fft[peak] + 0.05,
                           f'{freqs[peak]:.4f} Hz', fontsize=8, ha='center')

        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('归一化功率')
        ax.set_title(f'{feature_names[feat_idx]} - 功率谱密度对比')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, min(0.1, freqs[-1])])  # 限制频率范围

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"频域对比图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_residual_analysis(original: np.ndarray, filtered: np.ndarray,
                           feature_names: List[str], feature_indices: Optional[List[int]] = None,
                           save_path: Optional[str] = None):
    """
    绘制残差分析图

    参数:
    ----------
    original : np.ndarray
        原始数据
    filtered : np.ndarray
        滤波后数据
    feature_names : List[str]
        特征名称列表
    feature_indices : List[int], optional
        要绘制的特征索引列表
    save_path : str, optional
        保存图像的路径
    """
    n_samples, n_features = original.shape

    if feature_indices is None:
        feature_indices = list(range(min(4, n_features)))

    n_plots = len(feature_indices)
    if n_plots == 0:
        print("没有特征可绘制")
        return

    fig, axes = plt.subplots(n_plots, 2, figsize=(14, 3 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, 2)

    for idx, feat_idx in enumerate(feature_indices):
        if feat_idx >= n_features:
            continue

        # 提取数据
        orig_data = original[:, feat_idx]
        filt_data = filtered[:, feat_idx]

        # 计算残差
        residuals = orig_data - filt_data

        # 移除NaN值
        mask = ~(np.isnan(residuals))
        if np.sum(mask) < 10:
            continue

        residuals_clean = residuals[mask]

        # 左图：残差分布
        ax_left = axes[idx, 0]
        ax_left.hist(residuals_clean, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax_left.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

        # 添加正态分布参考
        mu, sigma = np.mean(residuals_clean), np.std(residuals_clean)
        if sigma > 0:
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            normal_pdf = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
            ax_left.plot(x, normal_pdf, 'r-', linewidth=2, alpha=0.7, label=f'正态分布\nμ={mu:.3f}, σ={sigma:.3f}')

        ax_left.set_xlabel('残差值')
        ax_left.set_ylabel('概率密度')
        ax_left.set_title(f'{feature_names[feat_idx]} - 残差分布')
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)

        # 右图：残差自相关
        ax_right = axes[idx, 1]
        if len(residuals_clean) > 50:
            max_lag = min(50, len(residuals_clean) // 4)
            lags = np.arange(max_lag)
            autocorr = np.array([np.corrcoef(residuals_clean[:-lag], residuals_clean[lag:])[0, 1]
                                for lag in lags[1:]])

            ax_right.bar(lags[1:], autocorr, width=0.8, alpha=0.7, color='green', edgecolor='black')
            ax_right.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_right.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax_right.axhline(y=-0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)

            ax_right.set_xlabel('滞后 (样本数)')
            ax_right.set_ylabel('自相关系数')
            ax_right.set_title(f'{feature_names[feat_idx]} - 残差自相关')
            ax_right.set_ylim([-0.2, 0.2])
            ax_right.grid(True, alpha=0.3)

            # 检查是否满足白噪声特性（自相关接近0）
            significant_corr = np.sum(np.abs(autocorr) > 0.1)
            if significant_corr == 0:
                ax_right.text(0.05, 0.95, '白噪声检验: 通过',
                             transform=ax_right.transAxes, fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                ax_right.text(0.05, 0.95, f'白噪声检验: 失败 ({significant_corr}个显著滞后)',
                             transform=ax_right.transAxes, fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:
            ax_right.text(0.5, 0.5, '数据不足\n无法计算自相关',
                         transform=ax_right.transAxes, ha='center', va='center', fontsize=12)
            ax_right.set_title(f'{feature_names[feat_idx]} - 残差自相关')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"残差分析图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_evaluation_summary(evaluation_results: Dict, feature_names: List[str],
                            save_path: Optional[str] = None):
    """
    绘制评估结果总结图

    参数:
    ----------
    evaluation_results : Dict
        包含各种评估指标的结果字典
    feature_names : List[str]
        特征名称列表
    save_path : str, optional
        保存图像的路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 平滑度改进条形图
    ax1 = axes[0, 0]
    if 'smoothness' in evaluation_results and evaluation_results['smoothness']['per_feature']:
        features = []
        improvements = []

        for result in evaluation_results['smoothness']['per_feature']:
            feat_idx = result['feature_index']
            if feat_idx < len(feature_names):
                features.append(feature_names[feat_idx])
                improvements.append(result['smoothness_improvement_percent'])

        if features:
            y_pos = np.arange(len(features))
            colors = ['green' if imp > 0 else 'red' for imp in improvements]

            ax1.barh(y_pos, improvements, color=colors, alpha=0.7)
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(features)
            ax1.set_xlabel('平滑度改进 (%)')
            ax1.set_title('各特征平滑度改进')
            ax1.grid(True, alpha=0.3, axis='x')

            # 添加平均值线
            avg_improvement = evaluation_results['smoothness']['overall'].get('average_smoothness_improvement', 0)
            ax1.axvline(x=avg_improvement, color='blue', linestyle='--', linewidth=1.5,
                       label=f'平均值: {avg_improvement:.1f}%')
            ax1.legend()

    # 2. SNR改进条形图
    ax2 = axes[0, 1]
    if 'snr' in evaluation_results and evaluation_results['snr']['per_feature']:
        features = []
        snr_improvements = []

        for result in evaluation_results['snr']['per_feature']:
            feat_idx = result['feature_index']
            if feat_idx < len(feature_names):
                features.append(feature_names[feat_idx])
                snr_improvements.append(result['snr_improvement_db'])

        if features:
            y_pos = np.arange(len(features))
            colors = ['green' if imp > 0 else 'red' for imp in snr_improvements]

            ax2.barh(y_pos, snr_improvements, color=colors, alpha=0.7)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(features)
            ax2.set_xlabel('SNR改进 (dB)')
            ax2.set_title('各特征信噪比改进')
            ax2.grid(True, alpha=0.3, axis='x')

            # 添加平均值线
            avg_snr_improvement = evaluation_results['snr']['overall'].get('average_snr_improvement_db', 0)
            ax2.axvline(x=avg_snr_improvement, color='blue', linestyle='--', linewidth=1.5,
                       label=f'平均值: {avg_snr_improvement:.2f} dB')
            ax2.legend()

    # 3. 相关性保持雷达图
    ax3 = axes[1, 0]
    if 'correlation' in evaluation_results and evaluation_results['correlation']['correlation_changes']:
        features = []
        preservations = []

        for result in evaluation_results['correlation']['correlation_changes']:
            feat_idx = result['feature_index']
            if feat_idx < len(feature_names):
                features.append(feature_names[feat_idx])
                preservations.append(result['correlation_preservation'] * 100)  # 转换为百分比

        if len(features) >= 3:  # 雷达图至少需要3个特征
            # 雷达图设置
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            preservations += preservations[:1]

            ax3 = plt.subplot(2, 2, 3, projection='polar')
            ax3.plot(angles, preservations, 'o-', linewidth=2, alpha=0.7)
            ax3.fill(angles, preservations, alpha=0.25)

            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(features, fontsize=9)
            ax3.set_ylim([0, 100])
            ax3.set_yticks([25, 50, 75, 100])
            ax3.set_yticklabels(['25%', '50%', '75%', '100%'])
            ax3.set_title('相关性保持雷达图 (越高越好)')
            ax3.grid(True)

            # 添加平均值
            avg_preservation = evaluation_results['correlation']['preservation_summary'].get('average_preservation', 0) * 100
            ax3.text(0.5, -0.1, f'平均保持: {avg_preservation:.1f}%',
                    transform=ax3.transAxes, ha='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, '特征不足\n无法绘制雷达图',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('相关性保持')

    # 4. 总体评估总结
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = "卡尔曼滤波效果评估总结\n\n"

    # 平滑度总结
    if 'smoothness' in evaluation_results and evaluation_results['smoothness']['overall']:
        smooth = evaluation_results['smoothness']['overall']
        summary_text += f"平滑度改进:\n"
        summary_text += f"  平均改进: {smooth.get('average_smoothness_improvement', 0):.1f}%\n"
        summary_text += f"  改进特征比例: {smooth.get('improvement_ratio', 0)*100:.1f}%\n\n"

    # SNR总结
    if 'snr' in evaluation_results and evaluation_results['snr']['overall']:
        snr = evaluation_results['snr']['overall']
        summary_text += f"信噪比改进:\n"
        summary_text += f"  平均改进: {snr.get('average_snr_improvement_db', 0):.2f} dB\n"
        summary_text += f"  改进特征比例: {snr.get('improvement_ratio', 0)*100:.1f}%\n\n"

    # 相关性总结
    if 'correlation' in evaluation_results and evaluation_results['correlation']['preservation_summary']:
        corr = evaluation_results['correlation']['preservation_summary']
        summary_text += f"相关性保持:\n"
        summary_text += f"  平均保持度: {corr.get('average_preservation', 0)*100:.1f}%\n"
        summary_text += f"  评估特征数: {corr.get('features_evaluated', 0)}\n\n"

    # MSE总结（如果有参考数据）
    if 'mse' in evaluation_results and evaluation_results['mse']['overall']:
        mse = evaluation_results['mse']['overall']
        summary_text += f"均方误差改进:\n"
        summary_text += f"  平均改进: {mse.get('average_mse_improvement', 0):.1f}%\n"
        summary_text += f"  改进特征比例: {mse.get('improvement_ratio', 0)*100:.1f}%"

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 总体结论
    conclusion = "\n总体结论:\n"
    overall_score = 0
    criteria_met = 0

    # 检查是否达到预期目标
    if 'smoothness' in evaluation_results:
        avg_improvement = evaluation_results['smoothness']['overall'].get('average_smoothness_improvement', 0)
        if avg_improvement > 10:  # 平滑度改进>10%
            conclusion += "✓ 平滑度显著改进\n"
            overall_score += 1
            criteria_met += 1
        else:
            conclusion += "✗ 平滑度改进不足\n"

    if 'snr' in evaluation_results:
        avg_snr_improvement = evaluation_results['snr']['overall'].get('average_snr_improvement_db', 0)
        if avg_snr_improvement > 1:  # SNR改进>1dB
            conclusion += "✓ 信噪比显著提升\n"
            overall_score += 1
            criteria_met += 1
        else:
            conclusion += "✗ 信噪比提升有限\n"

    if 'correlation' in evaluation_results:
        avg_preservation = evaluation_results['correlation']['preservation_summary'].get('average_preservation', 0)
        if avg_preservation > 0.9:  # 相关性保持>90%
            conclusion += "✓ 相关性良好保持\n"
            overall_score += 1
            criteria_met += 1
        else:
            conclusion += "✗ 相关性有所损失\n"

    conclusion += f"\n成功标准达成: {criteria_met}/3"

    ax4.text(0.1, 0.4, conclusion, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen' if criteria_met >= 2 else 'lightcoral', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"评估总结图已保存至: {save_path}")
        plt.close()
    else:
        plt.show()


# ==================== 数据修复函数 ====================

def fix_current_flow_outliers_simple(flow_data: np.ndarray, threshold: float = 100.0) -> np.ndarray:
    """
    简单修复current_flow异常突变
    使用线性插值替换超过阈值的突变点

    参数:
    ----------
    flow_data : np.ndarray
        current_flow数据序列
    threshold : float
        突变阈值，超过此阈值的差分被视为异常

    返回:
    -------
    np.ndarray
        修复后的数据
    """
    fixed_data = flow_data.copy()
    diffs = np.diff(flow_data)

    # 找出超过阈值的突变点
    outlier_mask = np.abs(diffs) > threshold

    if np.any(outlier_mask):
        print(f"发现 {np.sum(outlier_mask)} 个异常突变点 (阈值={threshold})")

        # 获取异常点索引
        outlier_indices = np.where(outlier_mask)[0]

        # 对每个异常点，用前后正常值的线性插值替换
        for idx in outlier_indices:
            # 寻找前后正常点
            prev_idx = idx - 1
            next_idx = idx + 2  # +2因为idx是差分索引，对应原数据idx和idx+1

            # 确保索引有效
            while prev_idx >= 0 and outlier_mask[prev_idx]:
                prev_idx -= 1
            while next_idx < len(diffs) and outlier_mask[next_idx-1]:
                next_idx += 1

            if prev_idx >= 0 and next_idx < len(flow_data):
                # 线性插值
                t = np.linspace(0, 1, next_idx - prev_idx + 1)
                interpolated = flow_data[prev_idx] + t * (flow_data[next_idx] - flow_data[prev_idx])
                fixed_data[prev_idx+1:next_idx] = interpolated[1:-1]
                print(f"  修复位置 {idx}: {flow_data[idx]:.2f} -> {flow_data[idx+1]:.2f}, "
                      f"变化 {diffs[idx]:.2f}")

    return fixed_data


def fix_time_series_outliers(data: np.ndarray, feature_names: List[str],
                             method: str = 'median_filter', **kwargs) -> np.ndarray:
    """
    修复时间序列异常值

    参数:
    ----------
    data : np.ndarray
        原始数据，形状 (n_samples, n_features)
    feature_names : List[str]
        特征名称列表
    method : str
        修复方法: 'median_filter', 'z_score', 'iqr'
    **kwargs : dict
        方法特定参数

    返回:
    -------
    np.ndarray
        修复后的数据
    """
    cleaned = data.copy()

    for i, col in enumerate(feature_names):
        if col == 'current_flow':
            # 对current_flow使用专门的方法
            threshold = kwargs.get('threshold', 100.0)
            cleaned[:, i] = fix_current_flow_outliers_simple(data[:, i], threshold)
            continue

        if method == 'median_filter':
            # 中值滤波
            window_size = kwargs.get('window_size', 5)
            from scipy.signal import medfilt
            cleaned[:, i] = medfilt(data[:, i], kernel_size=window_size)

        elif method == 'z_score':
            # Z-score异常值检测
            mean = np.nanmean(data[:, i])
            std = np.nanstd(data[:, i])
            if std > 0:
                z_scores = np.abs((data[:, i] - mean) / std)
                z_threshold = kwargs.get('z_threshold', 3.0)
                outliers = z_scores > z_threshold
                if np.any(outliers):
                    # 使用中位数替换异常值
                    median_val = np.nanmedian(data[:, i])
                    cleaned[outliers, i] = median_val
                    print(f"  {col}: 发现 {np.sum(outliers)} 个Z-score异常值 (阈值={z_threshold})")

    return cleaned


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("卡尔曼滤波工具函数库测试")
    print("-" * 40)

    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 5

    # 创建测试数据（正弦波加噪声）
    t = np.linspace(0, 10*np.pi, n_samples)
    true_states = np.zeros((n_samples, n_features))
    for i in range(n_features):
        amplitude = 1.0 + 0.2 * i
        frequency = 0.5 + 0.1 * i
        phase = 0.1 * i
        true_states[:, i] = amplitude * np.sin(frequency * t + phase) + 10

    # 添加噪声
    noise = np.random.normal(0, 0.3, (n_samples, n_features))
    observations = true_states + noise

    # 模拟滤波结果（简单低通滤波）
    filtered = np.zeros_like(observations)
    alpha = 0.1  # 平滑系数
    filtered[0] = observations[0]
    for i in range(1, n_samples):
        filtered[i] = alpha * observations[i] + (1 - alpha) * filtered[i-1]

    feature_names = [f'特征_{i}' for i in range(n_features)]

    print("1. 计算平滑度改进")
    smoothness_results = calculate_smoothness(observations, filtered)
    print(f"   平均改进: {smoothness_results['overall'].get('average_smoothness_improvement', 0):.2f}%")

    print("\n2. 计算SNR改进")
    snr_results = calculate_snr_improvement(observations, filtered)
    print(f"   平均SNR改进: {snr_results['overall'].get('average_snr_improvement_db', 0):.2f} dB")

    print("\n3. 计算相关性保持")
    correlation_results = calculate_correlation_preservation(observations, filtered, target_idx=n_features-1)
    print(f"   平均相关性保持: {correlation_results['preservation_summary'].get('average_preservation', 0)*100:.2f}%")

    print("\n4. 计算MSE改进")
    mse_results = calculate_mse_improvement(observations, filtered, true_states)
    if mse_results['overall']:
        print(f"   平均MSE改进: {mse_results['overall'].get('average_mse_improvement', 0):.2f}%")

    # 创建评估结果字典
    eval_results = {
        'smoothness': smoothness_results,
        'snr': snr_results,
        'correlation': correlation_results,
        'mse': mse_results
    }

    print("\n5. 生成可视化图表")
    # 创建输出目录
    import os
    output_dir = '../pic/kalman_test'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制各种图表
    plot_time_series_comparison(
        observations, filtered, feature_names,
        feature_indices=[0, 1, 2],
        save_path=os.path.join(output_dir, 'time_series_comparison.png')
    )

    plot_frequency_domain_comparison(
        observations, filtered, feature_names,
        sampling_rate=0.1,  # 10秒采样
        feature_indices=[0, 1],
        save_path=os.path.join(output_dir, 'frequency_domain_comparison.png')
    )

    plot_residual_analysis(
        observations, filtered, feature_names,
        feature_indices=[0, 1],
        save_path=os.path.join(output_dir, 'residual_analysis.png')
    )

    plot_evaluation_summary(
        eval_results, feature_names,
        save_path=os.path.join(output_dir, 'evaluation_summary.png')
    )

    print("\n" + "="*40)
    print("工具函数库测试完成!")
    print(f"所有图表已保存至: {output_dir}")