"""
卡尔曼滤波流程集成脚本
将卡尔曼滤波集成到现有数据流程中
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from typing import Optional, Dict, Tuple
import warnings

# 导入卡尔曼滤波模块
try:
    from kalman_filter import MultiVariateKalmanFilter, create_kalman_filter_for_features
    from kalman_utils import load_and_prepare_data, extract_timestamps, validate_data_quality
    from kalman_config import FILTER_COLS, NON_FILTER_COLS, KALMAN_CONFIG, OUTPUT_CONFIG
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有依赖并正确配置模块")
    sys.exit(1)

# 设置命令行参数
def parse_arguments():
    parser = argparse.ArgumentParser(description='卡尔曼滤波流程集成')
    parser.add_argument('--input', type=str, default='data_feature_engineered_v5.xlsx',
                       help='输入数据文件路径 (默认: data_feature_engineered_v5.xlsx)')
    parser.add_argument('--output', type=str, default='data_feature_engineered_v5_kalman.xlsx',
                       help='输出数据文件路径 (默认: data_feature_engineered_v5_kalman.xlsx)')
    parser.add_argument('--intermediate', type=str, default='data_kalman_filtered.xlsx',
                       help='中间数据文件路径 (默认: data_kalman_filtered.xlsx)')
    parser.add_argument('--config', type=str, default='kalman_config.py',
                       help='配置文件路径 (默认: kalman_config.py)')
    parser.add_argument('--features', type=str, nargs='+', default=None,
                       help='指定要滤波的特征列 (默认使用配置文件中的FILTER_COLS)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='生成可视化图表 (默认: True)')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细输出 (默认: True)')
    parser.add_argument('--test', action='store_true', default=False,
                       help='测试模式，只处理前1000个样本')
    parser.add_argument('--skip-validation', action='store_true', default=False,
                       help='跳过数据验证步骤')
    return parser.parse_args()


def check_prerequisites():
    """检查前置条件"""
    print("检查前置条件...")

    # 检查必要的Python包
    required_packages = ['numpy', 'pandas', 'filterpy', 'matplotlib']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"错误: 缺少必要的Python包: {missing_packages}")
        print("请使用以下命令安装: pip install " + " ".join(missing_packages))
        return False

    print("  所有必要的Python包已安装")
    return True


def load_data_with_validation(input_file: str, filter_cols: list,
                             skip_validation: bool = False) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    加载数据并进行验证

    参数:
    ----------
    input_file : str
        输入文件路径
    filter_cols : list
        需要滤波的特征列
    skip_validation : bool
        是否跳过验证

    返回:
    -------
    Tuple[pd.DataFrame, np.ndarray]
        (数据框, 观测序列)
    """
    print(f"\n1. 加载数据: {input_file}")

    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        print("请先运行 data_construct.py 和 features_engineering.py")
        sys.exit(1)

    # 加载数据
    df, observations = load_and_prepare_data(input_file, filter_cols)

    # 数据验证
    if not skip_validation:
        print("\n2. 验证数据质量")
        validation_report = validate_data_quality(df, filter_cols)

        # 检查缺失值
        total_missing = validation_report['summary']['total_missing']
        total_missing_percent = validation_report['summary']['total_missing_percent']

        if total_missing_percent > 20:
            print(f"警告: 数据缺失较多 ({total_missing_percent:.2f}%)")
            print("  卡尔曼滤波对缺失值敏感，建议先处理缺失值")

        # 显示警告
        if validation_report['warnings']:
            print(f"  发现 {len(validation_report['warnings'])} 个警告:")
            for warning in validation_report['warnings'][:5]:  # 只显示前5个
                print(f"    - {warning}")
            if len(validation_report['warnings']) > 5:
                print(f"    ... 还有 {len(validation_report['warnings']) - 5} 个警告")

    return df, observations


def apply_kalman_filter(df: pd.DataFrame, observations: np.ndarray,
                       filter_cols: list, test_mode: bool = False,
                       verbose: bool = True) -> pd.DataFrame:
    """
    应用卡尔曼滤波

    参数:
    ----------
    df : pd.DataFrame
        原始数据框
    observations : np.ndarray
        观测序列
    filter_cols : list
        需要滤波的特征列
    test_mode : bool
        测试模式（只处理部分数据）
    verbose : bool
        显示详细输出

    返回:
    -------
    pd.DataFrame
        滤波后的数据框
    """
    print("\n3. 应用卡尔曼滤波")

    # 提取时间戳
    timestamps = extract_timestamps(df)

    # 测试模式：只处理部分数据
    if test_mode:
        n_samples = min(1000, len(observations))
        observations = observations[:n_samples]
        timestamps = timestamps[:n_samples]
        print(f"  测试模式: 只处理前 {n_samples} 个样本")

    # 创建卡尔曼滤波器
    print(f"  创建卡尔曼滤波器，特征数: {len(filter_cols)}")
    kf = create_kalman_filter_for_features(
        feature_names=filter_cols,
        config={'verbose': verbose}
    )

    # 应用滤波
    print(f"  开始滤波处理，总样本数: {len(observations)}")
    filtered_states = kf.filter_sequence(observations, timestamps)

    # 打印统计信息
    if verbose:
        kf.print_statistics()

    # 更新数据框
    df_filtered = df.copy()
    if test_mode:
        # 测试模式：只更新前n_samples行
        n_samples = filtered_states.shape[0]
        for i, col in enumerate(filter_cols):
            df_filtered.loc[:n_samples-1, col] = filtered_states[:, i]
    else:
        # 完整模式：更新所有行
        for i, col in enumerate(filter_cols):
            df_filtered[col] = filtered_states[:, i]

    # 确保目标变量 system_cop 保持原始值（未被滤波）
    if 'system_cop' in df.columns:
        df_filtered['system_cop'] = df['system_cop'].values

    # 保存滤波历史（用于调试）
    if KALMAN_CONFIG.get('save_intermediate', False):
        history_file = 'kalman_filter_history.npz'
        np.savez(history_file,
                states=kf.get_state_estimates(),
                innovations=kf.get_innovation_sequence(),
                covariances=kf.get_covariance_history(),
                kalman_gains=kf.get_kalman_gain_history())
        print(f"  滤波历史已保存至: {history_file}")

    return df_filtered, kf


def save_filtered_data(df_filtered: pd.DataFrame, intermediate_file: str,
                      output_file: str, filter_cols: list, non_filter_cols: list):
    """
    保存滤波后的数据

    参数:
    ----------
    df_filtered : pd.DataFrame
        滤波后的数据框
    intermediate_file : str
        中间文件路径（包含所有列）
    output_file : str
        输出文件路径（只包含特征工程后的列）
    filter_cols : list
        滤波的特征列
    non_filter_cols : list
        未滤波的特征列
    """
    print("\n4. 保存滤波后的数据")

    # 保存中间数据（所有列）
    df_filtered.to_excel(intermediate_file, index=False)
    print(f"  中间数据已保存至: {intermediate_file}")
    print(f"    行数: {len(df_filtered)}, 列数: {len(df_filtered.columns)}")

    # 保存最终数据（只包含特征工程后的列）
    # 确定需要保留的列
    all_features = filter_cols + non_filter_cols
    output_cols = [col for col in all_features if col in df_filtered.columns]

    # 添加必要的系统列
    system_cols = ['date_time', 'total_power_kw', 'calc_Q_kw', 'system_cop']
    for col in system_cols:
        if col in df_filtered.columns and col not in output_cols:
            output_cols.append(col)

    # 创建输出数据框
    df_output = df_filtered[output_cols].copy()
    df_output.to_excel(output_file, index=False)

    print(f"  最终数据已保存至: {output_file}")
    print(f"    行数: {len(df_output)}, 列数: {len(df_output.columns)}")
    print(f"    包含特征: {output_cols}")


def generate_visualizations(df_original: pd.DataFrame, df_filtered: pd.DataFrame,
                           kf, filter_cols: list, visualize: bool = True):
    """
    生成可视化图表

    参数:
    ----------
    df_original : pd.DataFrame
        原始数据框
    df_filtered : pd.DataFrame
        滤波后的数据框
    kf : MultiVariateKalmanFilter
        卡尔曼滤波器实例
    filter_cols : list
        滤波的特征列
    visualize : bool
        是否生成可视化
    """
    if not visualize:
        return

    print("\n5. 生成可视化图表")

    # 创建输出目录
    vis_dir = OUTPUT_CONFIG.get('visualization_dir', '../pic/kalman_filter')
    os.makedirs(vis_dir, exist_ok=True)

    try:
        # 导入可视化函数
        from kalman_utils import (
            plot_time_series_comparison,
            plot_frequency_domain_comparison,
            plot_residual_analysis,
            plot_evaluation_summary,
            calculate_smoothness,
            calculate_snr_improvement,
            calculate_correlation_preservation
        )

        # 提取数据
        original_data = df_original[filter_cols].values
        filtered_data = df_filtered[filter_cols].values

        # 计算评估指标
        print("  计算评估指标...")
        evaluation_results = {}

        # 平滑度评估
        smoothness_results = calculate_smoothness(original_data, filtered_data)
        evaluation_results['smoothness'] = smoothness_results

        # SNR评估
        snr_results = calculate_snr_improvement(original_data, filtered_data)
        evaluation_results['snr'] = snr_results

        # 相关性保持评估
        # system_cop 不在 filter_cols 中（目标变量不滤波），需要从原始数据中获取
        target_col = 'system_cop'
        if target_col in filter_cols:
            target_idx = filter_cols.index(target_col)
            correlation_results = calculate_correlation_preservation(
                original_data, filtered_data, target_idx=target_idx
            )
        else:
            # 将 system_cop 追加到末尾做相关性分析
            orig_with_target = np.column_stack([original_data, df_original[target_col].values])
            filt_with_target = np.column_stack([filtered_data, df_original[target_col].values])
            target_idx = orig_with_target.shape[1] - 1
            correlation_results = calculate_correlation_preservation(
                orig_with_target, filt_with_target, target_idx=target_idx
            )
        evaluation_results['correlation'] = correlation_results

        # 1. 时间序列对比图
        print("  生成时间序列对比图...")
        plot_time_series_comparison(
            original_data, filtered_data, filter_cols,
            feature_indices=[0, 1, 2, 3],  # 显示前4个特征
            timestamps=extract_timestamps(df_original),
            save_path=os.path.join(vis_dir, 'time_series_comparison.png')
        )

        # 2. 频域分析图
        print("  生成频域分析图...")
        # 计算采样率（假设1分钟采样）
        sampling_rate = 1/60  # Hz
        plot_frequency_domain_comparison(
            original_data, filtered_data, filter_cols,
            sampling_rate=sampling_rate,
            feature_indices=[0, 1],  # 显示前2个特征
            save_path=os.path.join(vis_dir, 'frequency_domain_comparison.png')
        )

        # 3. 残差分析图
        print("  生成残差分析图...")
        plot_residual_analysis(
            original_data, filtered_data, filter_cols,
            feature_indices=[0, 1],  # 显示前2个特征
            save_path=os.path.join(vis_dir, 'residual_analysis.png')
        )

        # 4. 评估总结图
        print("  生成评估总结图...")
        plot_evaluation_summary(
            evaluation_results, filter_cols,
            save_path=os.path.join(vis_dir, 'evaluation_summary.png')
        )

        # 5. 滤波器收敛图
        print("  生成滤波器收敛图...")
        kf.plot_convergence(save_path=os.path.join(vis_dir, 'filter_convergence.png'))

        print(f"  所有图表已保存至: {vis_dir}")

    except ImportError as e:
        print(f"  警告: 无法导入可视化模块: {e}")
    except Exception as e:
        print(f"  警告: 生成可视化时出错: {e}")


def update_features_engineering_for_kalman():
    """
    修改features_engineering.py以支持卡尔曼滤波数据

    这个函数会检查features_engineering.py是否需要修改，
    并提供修改建议。
    """
    print("\n6. 检查features_engineering.py集成")

    fe_file = 'features_engineering.py'
    if not os.path.exists(fe_file):
        print(f"  警告: {fe_file} 不存在")
        return

    # 读取文件内容
    with open(fe_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否需要修改
    needs_modification = False
    modifications = []

    # 检查第5行是否是固定的文件路径
    lines = content.split('\n')
    if len(lines) >= 5 and 'data_deep_learning_final_v3.xlsx' in lines[4]:
        modifications.append("第5行: 将固定文件路径改为可配置参数")
        needs_modification = True

    # 检查是否有输入参数支持
    if 'import argparse' not in content and 'sys.argv' not in content:
        modifications.append("添加命令行参数支持")
        needs_modification = True

    if needs_modification:
        print(f"  {fe_file} 需要修改以支持卡尔曼滤波:")
        for mod in modifications:
            print(f"    - {mod}")

        # 创建备份
        backup_file = f'{fe_file}.backup'
        import shutil
        shutil.copy2(fe_file, backup_file)
        print(f"  原文件已备份至: {backup_file}")

        # 提供修改建议
        print("\n  修改建议:")
        print("  1. 添加命令行参数解析")
        print("  2. 将固定文件路径改为可配置参数")
        print("  3. 添加--input参数支持")
        print("\n  或直接使用以下命令:")
        print(f"    python {fe_file} --input data_kalman_filtered.xlsx")
    else:
        print(f"  {fe_file} 已支持可配置输入")


def run_pipeline_test():
    """运行完整的管道测试"""
    print("\n7. 运行管道测试")

    # 检查必要的文件
    required_files = [
        'data_construct.py',
        'features_engineering.py',
        'RF-train.py',
        'XGboost-train.py',
        'ANN-train.py',
        'LSTM-train-final.py'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"  警告: 以下文件缺失: {missing_files}")
        print("  管道测试可能不完整")

    # 测试命令
    test_commands = [
        "# 1. 原始数据构建",
        "# python data_construct.py",
        "#",
        "# 2. 特征工程（使用原始数据）",
        "# python features_engineering.py",
        "#",
        "# 3. 卡尔曼滤波",
        "# python kalman_integration.py --test",
        "#",
        "# 4. 特征工程（使用滤波数据）",
        "# python features_engineering.py --input data_kalman_filtered.xlsx",
        "#",
        "# 5. 模型训练对比",
        "# python RF-train.py  # 使用原始数据",
        "# python RF-train.py --input data_feature_engineered_v5_kalman.xlsx  # 使用滤波数据"
    ]

    print("  完整的管道测试命令:")
    for cmd in test_commands:
        print(f"    {cmd}")


def main():
    """主函数"""
    print("=" * 60)
    print("卡尔曼滤波流程集成")
    print("=" * 60)

    # 解析命令行参数
    args = parse_arguments()

    # 检查前置条件
    if not check_prerequisites():
        sys.exit(1)

    # 确定要滤波的特征
    if args.features:
        filter_cols = args.features
        print(f"使用命令行指定的特征: {filter_cols}")
    else:
        filter_cols = FILTER_COLS
        print(f"使用配置文件中的特征: {filter_cols}")

    # 加载和验证数据
    df_original, observations = load_data_with_validation(
        args.input, filter_cols, args.skip_validation
    )

    # 应用卡尔曼滤波
    df_filtered, kf = apply_kalman_filter(
        df_original, observations, filter_cols, args.test, args.verbose
    )

    # 保存滤波后的数据
    save_filtered_data(
        df_filtered, args.intermediate, args.output,
        filter_cols, NON_FILTER_COLS
    )

    # 生成可视化
    generate_visualizations(
        df_original, df_filtered, kf, filter_cols, args.visualize
    )

    # 检查features_engineering.py集成
    update_features_engineering_for_kalman()

    # 运行管道测试
    run_pipeline_test()

    # 总结
    print("\n" + "=" * 60)
    print("流程集成完成!")
    print("=" * 60)

    print(f"\n输出文件:")
    print(f"  1. 中间数据: {args.intermediate}")
    print(f"  2. 最终数据: {args.output}")
    print(f"  3. 可视化图表: {OUTPUT_CONFIG.get('visualization_dir', '../pic/kalman_filter')}")

    print(f"\n下一步:")
    print(f"  1. 运行特征工程使用滤波数据:")
    print(f"     python features_engineering.py --input {args.intermediate}")
    print(f"  2. 对比模型性能:")
    print(f"     python RF-train.py  # 原始数据")
    print(f"     python RF-train.py --input {args.output}  # 滤波数据")

    print(f"\n快速测试命令:")
    print(f"  python kalman_integration.py --test --verbose")

    print(f"\n完整管道测试:")
    print(f"  # 1. 原始数据构建")
    print(f"  python data_construct.py")
    print(f"  # 2. 特征工程（原始）")
    print(f"  python features_engineering.py")
    print(f"  # 3. 卡尔曼滤波")
    print(f"  python kalman_integration.py")
    print(f"  # 4. 特征工程（滤波）")
    print(f"  python features_engineering.py --input {args.intermediate}")
    print(f"  # 5. 模型训练对比")
    print(f"  python RF-train.py")
    print(f"  python RF-train.py --input {args.output}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)