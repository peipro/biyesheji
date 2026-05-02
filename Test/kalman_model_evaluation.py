"""
卡尔曼滤波模型性能评估脚本
比较滤波前后各模型的性能，生成对比报告
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional
import warnings

# 尝试导入模型训练函数
try:
    # 由于模型训练脚本是独立的，我们将通过子进程调用它们
    import subprocess
    MODEL_SCRIPTS = {
        'RF': 'RF-train.py',
        'XGBoost': 'XGboost-train.py',
        'ANN': 'ANN-train.py',
        'LSTM': 'LSTM-train-final.py'
    }
except ImportError:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='卡尔曼滤波模型性能评估')
    parser.add_argument('--original-data', type=str, default='data_feature_engineered_v5.xlsx',
                       help='原始数据文件路径 (默认: data_feature_engineered_v5.xlsx)')
    parser.add_argument('--filtered-data', type=str, default='data_feature_engineered_v5_kalman.xlsx',
                       help='滤波后数据文件路径 (默认: data_feature_engineered_v5_kalman.xlsx)')
    parser.add_argument('--models', type=str, nargs='+', default=['RF', 'XGBoost', 'ANN', 'LSTM'],
                       help='要评估的模型列表 (默认: RF XGBoost ANN LSTM)')
    parser.add_argument('--output', type=str, default='kalman_model_evaluation_report.md',
                       help='输出报告文件路径 (默认: kalman_model_evaluation_report.md)')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='生成可视化图表 (默认: True)')
    parser.add_argument('--test-mode', action='store_true', default=False,
                       help='测试模式，只运行快速评估')
    parser.add_argument('--skip-existing', action='store_true', default=False,
                       help='跳过已存在的结果，只运行缺失的评估')
    return parser.parse_args()


def check_data_files(original_file: str, filtered_file: str) -> bool:
    """检查数据文件是否存在"""
    print("检查数据文件...")

    files_to_check = [
        (original_file, "原始数据"),
        (filtered_file, "滤波后数据")
    ]

    missing_files = []
    for file_path, file_desc in files_to_check:
        if not os.path.exists(file_path):
            missing_files.append((file_path, file_desc))
            print(f"  ✗ {file_desc}不存在: {file_path}")
        else:
            print(f"  ✓ {file_desc}存在: {file_path}")

    if missing_files:
        print("\n缺失文件:")
        for file_path, file_desc in missing_files:
            print(f"  - {file_desc}: {file_path}")

        print("\n建议:")
        if 'data_feature_engineered_v5.xlsx' in original_file:
            print("  1. 运行特征工程生成原始数据:")
            print("     python features_engineering.py")
        if 'data_feature_engineered_v5_kalman.xlsx' in filtered_file:
            print("  2. 运行卡尔曼滤波生成滤波数据:")
            print("     python kalman_integration.py --test")

        return False

    # 检查数据一致性
    try:
        df_original = pd.read_excel(original_file)
        df_filtered = pd.read_excel(filtered_file)

        print(f"\n数据统计:")
        print(f"  原始数据: {len(df_original)} 行, {len(df_original.columns)} 列")
        print(f"  滤波数据: {len(df_filtered)} 行, {len(df_filtered.columns)} 列")

        # 检查特征列
        common_cols = set(df_original.columns) & set(df_filtered.columns)
        if len(common_cols) < 10:  # 至少应该有10个共同特征
            print(f"  警告: 共同特征较少 ({len(common_cols)}个)")

        return True

    except Exception as e:
        print(f"  错误: 读取数据文件时出错: {e}")
        return False


def run_model_training(model_name: str, data_file: str, test_mode: bool = False) -> Optional[Dict]:
    """
    运行模型训练并提取结果

    参数:
    ----------
    model_name : str
        模型名称
    data_file : str
        数据文件路径
    test_mode : bool
        测试模式

    返回:
    -------
    Optional[Dict]
        训练结果字典
    """
    print(f"\n运行 {model_name} 模型训练...")

    if model_name not in MODEL_SCRIPTS:
        print(f"  错误: 未知模型 {model_name}")
        return None

    script_name = MODEL_SCRIPTS[model_name]
    if not os.path.exists(script_name):
        print(f"  错误: 训练脚本不存在: {script_name}")
        return None

    # 准备参数
    cmd = ['python', script_name, '--input', data_file]

    # 如果是测试模式，添加测试参数（如果脚本支持）
    if test_mode:
        # 检查脚本是否支持测试参数
        with open(script_name, 'r', encoding='utf-8') as f:
            content = f.read()
            if '--test' in content or 'test_mode' in content:
                cmd.append('--test')

    # 记录开始时间
    start_time = time.time()

    try:
        # 运行训练脚本
        print(f"  执行命令: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )

        # 计算运行时间
        run_time = time.time() - start_time

        # 解析输出
        stdout = result.stdout
        stderr = result.stderr

        # 提取指标（根据不同模型的输出格式）
        metrics = extract_metrics_from_output(stdout, model_name)

        if metrics:
            metrics['run_time'] = run_time
            metrics['success'] = True
            print(f"  训练成功: R² = {metrics.get('r2', 'N/A'):.4f}, 时间 = {run_time:.1f}s")
        else:
            metrics = {
                'success': False,
                'error': '无法解析输出指标',
                'run_time': run_time,
                'stdout_preview': stdout[:500] if stdout else '无输出',
                'stderr_preview': stderr[:500] if stderr else '无错误'
            }
            print(f"  警告: 无法解析训练输出")

        return metrics

    except Exception as e:
        print(f"  错误: 运行训练脚本时出错: {e}")
        return {
            'success': False,
            'error': str(e),
            'run_time': time.time() - start_time
        }


def extract_metrics_from_output(output: str, model_name: str) -> Optional[Dict]:
    """
    从模型输出中提取指标

    参数:
    ----------
    output : str
        模型输出文本
    model_name : str
        模型名称

    返回:
    -------
    Optional[Dict]
        提取的指标字典
    """
    metrics = {}

    # 通用指标提取
    lines = output.split('\n')

    for line in lines:
        line_lower = line.lower()

        # R²分数
        if 'r2' in line_lower or 'r²' in line or 'r^2' in line:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    value = float(parts[1].strip().split()[0])
                    metrics['r2'] = value
                except:
                    pass

        # MAE
        if 'mae' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    value = float(parts[1].strip().split()[0])
                    metrics['mae'] = value
                except:
                    pass

        # RMSE
        if 'rmse' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    value = float(parts[1].strip().split()[0])
                    metrics['rmse'] = value
                except:
                    pass

        # MSE
        if 'mse' in line_lower and 'rmse' not in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    value = float(parts[1].strip().split()[0])
                    metrics['mse'] = value
                except:
                    pass

    # 如果没找到R²，尝试其他模式
    if 'r2' not in metrics:
        for line in lines:
            if 'score' in line_lower and (':' in line or '=' in line):
                try:
                    # 尝试提取数字
                    import re
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    if numbers:
                        value = float(numbers[0])
                        if 0 <= value <= 1:  # R²通常在0-1之间
                            metrics['r2'] = value
                            break
                except:
                    pass

    return metrics if metrics else None


def evaluate_models_for_data(data_file: str, data_type: str, models: List[str],
                           test_mode: bool = False, skip_existing: bool = False) -> Dict:
    """
    为特定数据评估所有模型

    参数:
    ----------
    data_file : str
        数据文件路径
    data_type : str
        数据类型 ('original' 或 'filtered')
    models : List[str]
        模型列表
    test_mode : bool
        测试模式
    skip_existing : bool
        跳过已存在的结果

    返回:
    -------
    Dict
        评估结果
    """
    print(f"\n{'='*60}")
    print(f"评估 {data_type} 数据")
    print(f"{'='*60}")

    results = {}
    cache_file = f'model_results_{data_type}.json'

    # 检查缓存
    if skip_existing and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_results = json.load(f)
            print(f"  从缓存加载结果: {cache_file}")
            return cached_results
        except:
            print(f"  警告: 无法读取缓存文件 {cache_file}")

    # 运行每个模型的训练
    for model in models:
        print(f"\n  {model} 模型:")

        # 运行训练
        model_results = run_model_training(model, data_file, test_mode)

        if model_results:
            results[model] = model_results
        else:
            results[model] = {
                'success': False,
                'error': '训练失败',
                'r2': None,
                'mae': None,
                'rmse': None
            }

    # 保存到缓存
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存到缓存: {cache_file}")
    except Exception as e:
        print(f"  警告: 无法保存缓存: {e}")

    return results


def compare_results(original_results: Dict, filtered_results: Dict) -> Dict:
    """
    比较原始数据和滤波数据的模型结果

    参数:
    ----------
    original_results : Dict
        原始数据结果
    filtered_results : Dict
        滤波数据结果

    返回:
    -------
    Dict
        比较结果
    """
    comparison = {}

    for model in set(original_results.keys()) | set(filtered_results.keys()):
        if model in original_results and model in filtered_results:
            orig = original_results[model]
            filt = filtered_results[model]

            # 检查是否都有R²值
            if orig.get('success') and filt.get('success') and 'r2' in orig and 'r2' in filt:
                orig_r2 = orig['r2']
                filt_r2 = filt['r2']

                # 计算改进
                if orig_r2 is not None and filt_r2 is not None:
                    r2_improvement = filt_r2 - orig_r2
                    r2_improvement_percent = (r2_improvement / abs(orig_r2)) * 100 if orig_r2 != 0 else 0

                    comparison[model] = {
                        'original_r2': orig_r2,
                        'filtered_r2': filt_r2,
                        'r2_improvement': r2_improvement,
                        'r2_improvement_percent': r2_improvement_percent,
                        'original_mae': orig.get('mae'),
                        'filtered_mae': filt.get('mae'),
                        'original_rmse': orig.get('rmse'),
                        'filtered_rmse': filt.get('rmse'),
                        'original_run_time': orig.get('run_time'),
                        'filtered_run_time': filt.get('run_time'),
                        'success': True
                    }
                else:
                    comparison[model] = {
                        'success': False,
                        'error': '缺少R²值',
                        'original_success': orig.get('success'),
                        'filtered_success': filt.get('success')
                    }
            else:
                comparison[model] = {
                    'success': False,
                    'error': '训练失败或缺少指标',
                    'original_success': orig.get('success'),
                    'filtered_success': filt.get('success')
                }

    return comparison


def generate_report(comparison_results: Dict, args, output_file: str):
    """
    生成评估报告

    参数:
    ----------
    comparison_results : Dict
        比较结果
    args : argparse.Namespace
        命令行参数
    output_file : str
        输出文件路径
    """
    print(f"\n{'='*60}")
    print(f"生成评估报告")
    print(f"{'='*60}")

    # 创建报告内容
    report = []
    report.append("# 卡尔曼滤波模型性能评估报告")
    report.append("")
    report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 评估配置
    report.append("## 评估配置")
    report.append("")
    report.append(f"- **原始数据**: `{args.original_data}`")
    report.append(f"- **滤波数据**: `{args.filtered_data}`")
    report.append(f"- **评估模型**: {', '.join(args.models)}")
    report.append(f"- **测试模式**: {'是' if args.test_mode else '否'}")
    report.append("")

    # 结果摘要
    report.append("## 结果摘要")
    report.append("")

    successful_models = []
    improved_models = []
    best_improvement = {'model': None, 'improvement': -float('inf')}

    for model, results in comparison_results.items():
        if results.get('success'):
            successful_models.append(model)
            improvement = results.get('r2_improvement', 0)
            if improvement > 0:
                improved_models.append(model)
                if improvement > best_improvement['improvement']:
                    best_improvement = {'model': model, 'improvement': improvement}

    report.append(f"- **成功评估模型**: {len(successful_models)}/{len(comparison_results)}")
    report.append(f"- **性能提升模型**: {len(improved_models)}/{len(successful_models)}")

    if best_improvement['model']:
        report.append(f"- **最大提升模型**: {best_improvement['model']} (R²提升: {best_improvement['improvement']:.4f})")
    report.append("")

    # 详细结果表
    report.append("## 详细结果")
    report.append("")
    report.append("| 模型 | 原始数据 R² | 滤波数据 R² | R²提升 | 提升百分比 | 状态 |")
    report.append("|------|------------|------------|--------|------------|------|")

    for model in sorted(comparison_results.keys()):
        results = comparison_results[model]

        if results.get('success'):
            orig_r2 = results['original_r2']
            filt_r2 = results['filtered_r2']
            improvement = results['r2_improvement']
            improvement_percent = results['r2_improvement_percent']

            # 确定状态
            if improvement > 0.01:  # 提升超过0.01
                status = "✅ 显著提升"
            elif improvement > 0:
                status = "⚠️ 轻微提升"
            elif improvement == 0:
                status = "➖ 无变化"
            else:
                status = "❌ 性能下降"

            report.append(f"| {model} | {orig_r2:.4f} | {filt_r2:.4f} | {improvement:+.4f} | {improvement_percent:+.1f}% | {status} |")
        else:
            error_msg = results.get('error', '未知错误')
            report.append(f"| {model} | - | - | - | - | ❌ {error_msg} |")

    report.append("")

    # 分析与建议
    report.append("## 分析与建议")
    report.append("")

    if improved_models:
        report.append("### 成功案例")
        report.append("")
        report.append("以下模型从卡尔曼滤波中受益:")
        report.append("")
        for model in improved_models:
            results = comparison_results[model]
            improvement = results['r2_improvement']
            improvement_percent = results['r2_improvement_percent']
            report.append(f"- **{model}**: R²提升 {improvement:.4f} (相对提升 {improvement_percent:.1f}%)")
        report.append("")

        report.append("### 建议")
        report.append("")
        report.append("1. **推荐使用卡尔曼滤波**，特别是对于受益明显的模型")
        report.append("2. **参数调优**: 考虑进一步优化卡尔曼滤波参数")
        report.append("3. **特征选择**: 分析哪些特征最受益于滤波")
        report.append("")
    else:
        report.append("### 分析")
        report.append("")
        report.append("本次评估中，卡尔曼滤波未显示出明显的性能提升。")
        report.append("")
        report.append("### 可能原因")
        report.append("")
        report.append("1. **数据质量**: 原始数据噪声水平较低")
        report.append("2. **滤波参数**: 卡尔曼滤波参数可能需要调整")
        report.append("3. **模型特性**: 某些模型对噪声不敏感")
        report.append("4. **评估方法**: 需要更多数据或交叉验证")
        report.append("")
        report.append("### 建议")
        report.append("")
        report.append("1. **参数调优**: 重新调整卡尔曼滤波参数")
        report.append("2. **特征分析**: 检查哪些特征需要滤波")
        report.append("3. **模型调整**: 考虑其他滤波方法或模型")
        report.append("")

    # 下一步
    report.append("## 下一步")
    report.append("")
    report.append("1. **深入分析**: 检查各个特征的滤波效果")
    report.append("2. **参数优化**: 使用`kalman_parameter_tuning.py`优化滤波参数")
    report.append("3. **扩展评估**: 增加交叉验证，使用更多评估指标")
    report.append("4. **模型调优**: 对受益模型进行超参数调优")
    report.append("")

    # 保存报告
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        print(f"  报告已保存至: {output_file}")
    except Exception as e:
        print(f"  错误: 保存报告时出错: {e}")
        # 打印到控制台
        print('\n'.join(report))

    return report


def visualize_comparison(comparison_results: Dict, output_dir: str = '../pic/kalman_evaluation'):
    """
    可视化比较结果

    参数:
    ----------
    comparison_results : Dict
        比较结果
    output_dir : str
        输出目录
    """
    if not comparison_results:
        print("  没有有效结果可可视化")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    models = []
    original_r2 = []
    filtered_r2 = []
    improvements = []

    for model, results in comparison_results.items():
        if results.get('success'):
            models.append(model)
            original_r2.append(results['original_r2'])
            filtered_r2.append(results['filtered_r2'])
            improvements.append(results['r2_improvement'])

    if not models:
        print("  没有成功模型可可视化")
        return

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. R²对比条形图
    ax1 = axes[0, 0]
    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, original_r2, width, label='原始数据', alpha=0.7, color='blue')
    ax1.bar(x + width/2, filtered_r2, width, label='滤波数据', alpha=0.7, color='green')

    ax1.set_xlabel('模型')
    ax1.set_ylabel('R²分数')
    ax1.set_title('模型性能对比 (R²)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加数值标签
    for i, (orig, filt) in enumerate(zip(original_r2, filtered_r2)):
        ax1.text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, filt + 0.01, f'{filt:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. R²提升条形图
    ax2 = axes[0, 1]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(models, improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('模型')
    ax2.set_ylabel('R²提升')
    ax2.set_title('卡尔曼滤波带来的R²提升')
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # 添加数值标签
    for i, imp in enumerate(improvements):
        ax2.text(i, imp + (0.001 if imp >= 0 else -0.003), f'{imp:+.3f}',
                ha='center', va='bottom' if imp >= 0 else 'top', fontsize=8)

    # 3. 提升百分比
    ax3 = axes[1, 0]
    improvement_percents = []
    for i, imp in enumerate(improvements):
        if original_r2[i] != 0:
            percent = (imp / abs(original_r2[i])) * 100
        else:
            percent = 0
        improvement_percents.append(percent)

    colors_percent = ['green' if p > 0 else 'red' for p in improvement_percents]
    ax3.bar(models, improvement_percents, color=colors_percent, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax3.set_xlabel('模型')
    ax3.set_ylabel('提升百分比 (%)')
    ax3.set_title('相对性能提升百分比')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # 添加数值标签
    for i, p in enumerate(improvement_percents):
        ax3.text(i, p + (0.5 if p >= 0 else -1), f'{p:+.1f}%',
                ha='center', va='bottom' if p >= 0 else 'top', fontsize=8)

    # 4. 综合评估
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 计算统计
    avg_improvement = np.mean(improvements) if improvements else 0
    avg_percent = np.mean(improvement_percents) if improvement_percents else 0
    positive_count = sum(1 for imp in improvements if imp > 0)
    total_count = len(improvements)

    summary_text = f"""
    综合评估结果

    模型数量: {total_count}
    性能提升模型: {positive_count}

    平均R²提升: {avg_improvement:.4f}
    平均相对提升: {avg_percent:.1f}%

    评估结论:
    """

    if positive_count == total_count:
        summary_text += "所有模型都从卡尔曼滤波中受益，强烈推荐使用。"
    elif positive_count >= total_count / 2:
        summary_text += "大部分模型受益，建议使用卡尔曼滤波。"
    elif positive_count > 0:
        summary_text += "部分模型受益，可选择性使用。"
    else:
        summary_text += "没有模型明显受益，需要重新评估滤波参数。"

    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  可视化图表已保存至: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("卡尔曼滤波模型性能评估")
    print("=" * 60)

    # 解析参数
    args = parse_arguments()

    # 检查数据文件
    if not check_data_files(args.original_data, args.filtered_data):
        print("\n错误: 数据文件检查失败")
        sys.exit(1)

    # 评估原始数据
    original_results = evaluate_models_for_data(
        args.original_data, 'original', args.models,
        args.test_mode, args.skip_existing
    )

    # 评估滤波数据
    filtered_results = evaluate_models_for_data(
        args.filtered_data, 'filtered', args.models,
        args.test_mode, args.skip_existing
    )

    # 比较结果
    comparison_results = compare_results(original_results, filtered_results)

    # 生成报告
    report = generate_report(comparison_results, args, args.output)

    # 可视化
    if args.visualize:
        visualize_comparison(comparison_results)

    # 总结
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)

    # 显示关键结果
    successful_comparisons = [r for r in comparison_results.values() if r.get('success')]
    if successful_comparisons:
        improvements = [r['r2_improvement'] for r in successful_comparisons]
        avg_improvement = np.mean(improvements)

        print(f"\n关键结果:")
        print(f"  成功比较模型: {len(successful_comparisons)}/{len(comparison_results)}")
        print(f"  平均R²提升: {avg_improvement:.4f}")

        if avg_improvement > 0:
            print(f"  ✅ 卡尔曼滤波整体上提高了模型性能")
        else:
            print(f"  ⚠️ 卡尔曼滤波未提高模型性能")
    else:
        print(f"\n没有成功的模型比较")

    print(f"\n输出文件:")
    print(f"  1. 评估报告: {args.output}")
    print(f"  2. 可视化图表: ../pic/kalman_evaluation/")
    print(f"  3. 原始数据结果缓存: model_results_original.json")
    print(f"  4. 滤波数据结果缓存: model_results_filtered.json")

    print(f"\n下一步建议:")
    print(f"  1. 查看详细报告: {args.output}")
    print(f"  2. 优化滤波参数: python kalman_parameter_tuning.py")
    print(f"  3. 重新评估: python {__file__} --skip-existing")


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