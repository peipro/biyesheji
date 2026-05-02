"""
快速测试卡尔曼滤波修复效果
修复current_flow异常突变并测试效果
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from kalman_utils import fix_current_flow_outliers_simple

def test_fix_current_flow():
    """测试修复current_flow异常"""
    print("=" * 60)
    print("快速测试卡尔曼滤波修复效果")
    print("=" * 60)

    # 1. 加载数据
    data_file = 'data_feature_engineered_v5.xlsx'
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        print("请先运行 features_engineering.py 生成数据")
        return None

    df = pd.read_excel(data_file)
    flow_original = df['current_flow'].values

    print(f"数据文件: {data_file}")
    print(f"数据长度: {len(flow_original)}")
    print(f"原始数据均值: {np.mean(flow_original):.2f}")
    print(f"原始数据标准差: {np.std(flow_original):.2f}")

    # 2. 分析原始数据
    diffs_original = np.diff(flow_original)
    print(f"\n原始数据差分统计:")
    print(f"  差分标准差: {np.std(diffs_original):.2f}")
    print(f"  最大绝对差分: {np.max(np.abs(diffs_original)):.2f}")

    # 测试不同阈值
    thresholds = [50, 100, 200]
    best_threshold = None
    best_improvement = 0
    best_fixed_data = None

    print(f"\n测试不同修复阈值:")

    for threshold in thresholds:
        print(f"\n阈值 = {threshold}:")

        # 修复数据
        flow_fixed = fix_current_flow_outliers_simple(flow_original, threshold)
        diffs_fixed = np.diff(flow_fixed)

        # 计算改进
        diff_std_original = np.std(diffs_original)
        diff_std_fixed = np.std(diffs_fixed)

        if diff_std_original > 0:
            improvement = (diff_std_original - diff_std_fixed) / diff_std_original * 100
        else:
            improvement = 0

        print(f"  修复后差分标准差: {diff_std_fixed:.2f}")
        print(f"  改进比例: {improvement:.1f}%")

        # 检查修复是否过度
        if improvement > 0 and diff_std_fixed > 0.5:  # 避免过度平滑
            if improvement > best_improvement:
                best_improvement = improvement
                best_threshold = threshold
                best_fixed_data = flow_fixed

    # 3. 使用最佳阈值修复
    if best_threshold is not None:
        print(f"\n选择最佳阈值: {best_threshold} (改进: {best_improvement:.1f}%)")

        # 最终修复
        final_fixed = fix_current_flow_outliers_simple(flow_original, best_threshold)
        diffs_final = np.diff(final_fixed)

        print(f"最终修复效果:")
        print(f"  原始差分标准差: {np.std(diffs_original):.2f}")
        print(f"  修复后差分标准差: {np.std(diffs_final):.2f}")
        print(f"  总改进: {best_improvement:.1f}%")

        # 4. 更新数据框并保存
        df_fixed = df.copy()
        df_fixed['current_flow'] = final_fixed

        output_file = 'data_feature_engineered_v5_fixed.xlsx'
        df_fixed.to_excel(output_file, index=False)
        print(f"\n修复后的数据已保存至: {output_file}")
        print(f"  行数: {len(df_fixed)}, 列数: {len(df_fixed.columns)}")

        # 5. 生成对比图
        generate_comparison_plot(flow_original, final_fixed, best_threshold)

        return df_fixed
    else:
        print("\n警告: 未找到合适的修复阈值")
        return None

def generate_comparison_plot(original, fixed, threshold):
    """生成修复前后对比图"""
    output_dir = '../pic'
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 时间序列对比
    ax1 = axes[0, 0]
    ax1.plot(original, linewidth=0.8, alpha=0.7, color='blue', label='原始数据')
    ax1.plot(fixed, linewidth=0.8, alpha=0.7, color='red', label='修复后数据', linestyle='--')
    ax1.set_xlabel('样本索引')
    ax1.set_ylabel('current_flow值')
    ax1.set_title('修复前后对比 (阈值={})'.format(threshold))
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 差分分布对比
    ax2 = axes[0, 1]
    diffs_original = np.diff(original)
    diffs_fixed = np.diff(fixed)

    bins = np.linspace(min(np.min(diffs_original), np.min(diffs_fixed)),
                      max(np.max(diffs_original), np.max(diffs_fixed)), 50)

    ax2.hist(diffs_original, bins=bins, alpha=0.5, label='原始数据', color='blue', density=True)
    ax2.hist(diffs_fixed, bins=bins, alpha=0.5, label='修复后数据', color='red', density=True)
    ax2.axvline(x=threshold, color='green', linestyle='--', linewidth=1, label=f'阈值={threshold}')
    ax2.axvline(x=-threshold, color='green', linestyle='--', linewidth=1)
    ax2.set_xlabel('差分值')
    ax2.set_ylabel('概率密度')
    ax2.set_title('差分分布对比')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 差分标准差变化
    ax3 = axes[1, 0]
    categories = ['原始数据', '修复后数据']
    values = [np.std(diffs_original), np.std(diffs_fixed)]

    bars = ax3.bar(categories, values, color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('差分标准差')
    ax3.set_title('平滑度改进')
    ax3.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')

    improvement = (values[0] - values[1]) / values[0] * 100 if values[0] > 0 else 0

    stats_text = f"""
    修复效果统计 (阈值={threshold})

    原始数据:
      均值: {np.mean(original):.2f}
      标准差: {np.std(original):.2f}
      差分标准差: {values[0]:.2f}
      最大绝对差分: {np.max(np.abs(diffs_original)):.2f}

    修复后数据:
      均值: {np.mean(fixed):.2f}
      标准差: {np.std(fixed):.2f}
      差分标准差: {values[1]:.2f}
      最大绝对差分: {np.max(np.abs(diffs_fixed)):.2f}

    改进效果:
      差分标准差降低: {improvement:.1f}%
      最大突变降低: {(np.max(np.abs(diffs_original)) - np.max(np.abs(diffs_fixed)))/np.max(np.abs(diffs_original))*100:.1f}%
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, 'current_flow_fix_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"对比图已保存至: {output_path}")

def main():
    """主函数"""
    try:
        result = test_fix_current_flow()

        if result is not None:
            print("\n" + "=" * 60)
            print("测试完成!")
            print("=" * 60)

            # 检查修复后的文件
            if os.path.exists('data_feature_engineered_v5_fixed.xlsx'):
                df_check = pd.read_excel('data_feature_engineered_v5_fixed.xlsx')
                print(f"修复文件验证:")
                print(f"  文件大小: {os.path.getsize('data_feature_engineered_v5_fixed.xlsx') / 1024:.1f} KB")
                print(f"  数据形状: {df_check.shape}")
                print(f"  current_flow统计:")
                print(f"    均值: {df_check['current_flow'].mean():.2f}")
                print(f"    标准差: {df_check['current_flow'].std():.2f}")
                print(f"    差分标准差: {df_check['current_flow'].diff().std():.2f}")

        return result

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()