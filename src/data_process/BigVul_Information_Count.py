import pandas as pd
import numpy as np
from collections import Counter
import argparse
import re
from pathlib import Path

"""一、分析整体BigVul数据集漏洞分布情况"""
def normalize_cwe(cwe_value):
    """
    标准化CWE值

    Args:
        cwe_value: CWE值

    Returns:
        标准化后的CWE字符串
    """
    if pd.isna(cwe_value):
        return 'Non-CWE'

    cwe_str = str(cwe_value).strip()

    # 处理空字符串或无效值
    if not cwe_str or cwe_str.lower() == 'nan' or cwe_str == 'None' or cwe_str == 'null':
        return 'Non-CWE'

    # 如果已经是Non-CWE
    if cwe_str.lower() == 'non-cwe' or cwe_str.lower() == 'none-cwe':
        return 'Non-CWE'

    # 标准化CWE格式
    cwe_str = cwe_str.upper()

    # 如果以"CWE-"开头，确保格式正确
    if cwe_str.startswith('CWE-'):
        # 提取CWE编号
        cwe_num = cwe_str[4:].strip()
        # 只保留数字部分
        if cwe_num.isdigit():
            return f'CWE-{cwe_num}'
        else:
            # 尝试提取数字
            match = re.search(r'(\d+)', cwe_num)
            if match:
                return f'CWE-{match.group(1)}'
            else:
                return 'Non-CWE'

    # 如果只包含数字
    if cwe_str.isdigit():
        return f'CWE-{cwe_str}'

    # 其他格式，尝试提取CWE编号
    match = re.search(r'CWE[_\s\-]*(\d+)', cwe_str, re.IGNORECASE)
    if match:
        return f'CWE-{match.group(1)}'

    # 尝试提取纯数字
    match = re.search(r'(\d+)', cwe_str)
    if match:
        return f'CWE-{match.group(1)}'

    # 无法识别，标记为Non-CWE
    return 'Non-CWE'


def find_vulnerability_column(df):
    """
    自动查找漏洞标签列

    Args:
        df: DataFrame

    Returns:
        漏洞标签列名
    """
    # 优先查找名为"vul"的列
    if 'vul' in df.columns:
        return 'vul'

    # 其次查找常见的漏洞标签列名（不区分大小写）
    common_vul_columns = ['vul', 'label', 'vulnerability', 'is_vulnerability', 'target',
                          'vulnerable', 'is_vul', 'vuln', 'vul_flag']

    # 首先检查精确匹配（不区分大小写）
    for common_col in common_vul_columns:
        for col in df.columns:
            if col.lower() == common_col.lower():
                return col

    # 然后检查部分匹配
    for col in df.columns:
        col_lower = col.lower()
        for common_col in common_vul_columns:
            if common_col in col_lower:
                return col

    # 如果没有找到，尝试使用第一个数值型列（假设是标签列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"警告: 未找到明确的漏洞标签列，使用数值列 '{numeric_cols[0]}' 作为漏洞标签")
        return numeric_cols[0]

    # 如果也没有数值型列，返回None
    print("错误: 未找到漏洞标签列")
    return None


def convert_vul_column_to_numeric(df, vul_column):
    """
    将漏洞标签列转换为数值型（0或1）

    Args:
        df: DataFrame
        vul_column: 漏洞标签列名

    Returns:
        转换后的DataFrame
    """
    print(f"正在处理漏洞标签列: {vul_column}")

    # 首先查看列的数据类型和前几个值
    print(f"  列的数据类型: {df[vul_column].dtype}")
    print(f"  列的唯一值: {df[vul_column].unique()[:10]}")

    # 备份原始列
    df[f'{vul_column}_original'] = df[vul_column]

    # 尝试转换为数值型
    try:
        # 先尝试直接转换为数值
        df[vul_column] = pd.to_numeric(df[vul_column], errors='coerce')

        # 如果转换成功，检查是否已经是0和1
        unique_values = df[vul_column].dropna().unique()
        print(f"  转换后的唯一值: {unique_values}")

        # 如果不是0和1，尝试映射
        if len(unique_values) > 0 and (not set(unique_values).issubset({0, 1})):
            print(f"  警告: 漏洞标签列包含非二进制值: {unique_values}")
            print(f"  尝试将大于0的值映射为1...")
            df[vul_column] = (df[vul_column] > 0).astype(int)
    except:
        print(f"  无法直接转换为数值，尝试基于字符串映射...")

        # 将列转换为字符串类型进行处理
        df[vul_column] = df[vul_column].astype(str).str.strip().str.lower()

        # 常见表示漏洞的字符串
        vul_true_values = ['true', '1', 'yes', 'y', 't', 'vulnerable', 'vul', 'positive']
        vul_false_values = ['false', '0', 'no', 'n', 'f', 'non-vulnerable', 'non-vul', 'negative', 'clean']

        # 创建映射函数
        def map_vul_value(val):
            if pd.isna(val) or val == 'nan':
                return 0
            if val in vul_true_values:
                return 1
            elif val in vul_false_values:
                return 0
            else:
                # 尝试从字符串中提取数字
                try:
                    num_val = float(val)
                    return 1 if num_val > 0 else 0
                except:
                    # 如果无法转换，默认设为0
                    print(f"    警告: 无法识别的漏洞标签值: '{val}'，默认设为0")
                    return 0

        # 应用映射
        df[vul_column] = df[vul_column].apply(map_vul_value)

    # 检查转换后的值
    unique_values = df[vul_column].unique()
    print(f"  最终的唯一值: {unique_values}")

    # 验证转换结果
    vul_count = df[df[vul_column] == 1].shape[0]
    non_vul_count = df[df[vul_column] == 0].shape[0]
    print(f"  漏洞样本数: {vul_count}")
    print(f"  非漏洞样本数: {non_vul_count}")

    return df


def count_cwe_types(input_file, output_file='cwe_statistics.csv'):
    """
    统计BigVul数据集中CWE类型的分布

    Args:
        input_file: BigVul数据集文件路径
        output_file: 输出CSV文件路径
    """

    # 读取数据集，指定低内存模式以避免警告
    print(f"正在读取数据集: {input_file}")
    try:
        # 尝试读取所有列作为字符串以避免类型混合警告
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 检查数据集中是否有CWE列
    cwe_column = None
    for col in df.columns:
        if 'cwe' in col.lower():
            cwe_column = col
            break

    if cwe_column is None:
        print("错误: 数据集中未找到CWE列")
        print(f"可用的列: {list(df.columns)}")
        return

    print(f"找到CWE列: {cwe_column}")

    # 查找漏洞标签列
    vul_column = find_vulnerability_column(df)
    if vul_column is None:
        print("错误: 未找到漏洞标签列")
        return

    print(f"找到漏洞标签列: {vul_column}")

    # 转换漏洞标签列为数值型
    df = convert_vul_column_to_numeric(df, vul_column)

    # 应用CWE标准化
    print("正在标准化CWE值...")
    df['normalized_cwe'] = df[cwe_column].apply(normalize_cwe)

    # 统计所有CWE类型的数量
    cwe_counts = Counter(df['normalized_cwe'])

    # 计算每个CWE类型的漏洞样本数量
    cwe_vul_counts = {}
    for cwe_type in cwe_counts.keys():
        vul_count = df[(df['normalized_cwe'] == cwe_type) & (df[vul_column] == 1)].shape[0]
        cwe_vul_counts[cwe_type] = vul_count

    # 计算总样本数
    total_samples = len(df)
    print(f"总样本数: {total_samples}")

    # 计算总漏洞样本数
    total_vul_samples = df[df[vul_column] == 1].shape[0]
    print(f"总漏洞样本数: {total_vul_samples} ({total_vul_samples / total_samples * 100:.2f}%)")

    # 将Non-CWE单独处理
    non_cwe_count = cwe_counts.get('Non-CWE', 0)
    non_cwe_vul_count = cwe_vul_counts.get('Non-CWE', 0)

    # 移除Non-CWE以进行排序
    cwe_counts_for_sorting = {k: v for k, v in cwe_counts.items() if k != 'Non-CWE'}

    # 按数量排序（降序）
    sorted_cwe = sorted(cwe_counts_for_sorting.items(), key=lambda x: x[1], reverse=True)

    # 获取前10个最多的CWE类型
    top_n = min(10, len(sorted_cwe))
    top_cwe = sorted_cwe[:top_n]

    # 计算other类型的数量（前10个之外的所有非Non-CWE类型）
    other_cwe_count = sum(count for _, count in sorted_cwe[top_n:])
    other_cwe_vul_count = sum(cwe_vul_counts.get(cwe_type, 0) for cwe_type, _ in sorted_cwe[top_n:])

    # 准备结果数据
    results = []
    total_counted = 0
    total_vul_counted = 0

    # 添加前10个CWE类型
    for cwe_type, count in top_cwe:
        vul_count = cwe_vul_counts.get(cwe_type, 0)
        vul_ratio = (vul_count / count * 100) if count > 0 else 0
        total_percentage = (count / total_samples) * 100
        # 新增：计算该CWE漏洞数占总漏洞样本的比例
        vul_percentage_of_total = (vul_count / total_vul_samples * 100) if total_vul_samples > 0 else 0

        results.append({
            'CWE_Type': cwe_type,
            'Count': count,
            'Percentage_in_Total': f"{total_percentage:.2f}%",
            'Vulnerability_Count': vul_count,
            'Vulnerability_Percentage_in_CWE': f"{vul_ratio:.2f}%",
            'Vulnerability_Percentage_of_Total_Vul': f"{vul_percentage_of_total:.2f}%"  # 新增列
        })

        total_counted += count
        total_vul_counted += vul_count

    # 添加other类型（如果有的话）
    if other_cwe_count > 0:
        other_vul_ratio = (other_cwe_vul_count / other_cwe_count * 100) if other_cwe_count > 0 else 0
        other_total_percentage = (other_cwe_count / total_samples) * 100
        # 新增：计算other漏洞数占总漏洞样本的比例
        other_vul_percentage_of_total = (other_cwe_vul_count / total_vul_samples * 100) if total_vul_samples > 0 else 0

        results.append({
            'CWE_Type': 'other',
            'Count': other_cwe_count,
            'Percentage_in_Total': f"{other_total_percentage:.2f}%",
            'Vulnerability_Count': other_cwe_vul_count,
            'Vulnerability_Percentage_in_CWE': f"{other_vul_ratio:.2f}%",
            'Vulnerability_Percentage_of_Total_Vul': f"{other_vul_percentage_of_total:.2f}%"  # 新增列
        })

        total_counted += other_cwe_count
        total_vul_counted += other_cwe_vul_count

    # 添加Non-CWE类型
    non_cwe_vul_ratio = (non_cwe_vul_count / non_cwe_count * 100) if non_cwe_count > 0 else 0
    non_cwe_total_percentage = (non_cwe_count / total_samples) * 100
    # 新增：计算Non-CWE漏洞数占总漏洞样本的比例
    non_cwe_vul_percentage_of_total = (non_cwe_vul_count / total_vul_samples * 100) if total_vul_samples > 0 else 0

    results.append({
        'CWE_Type': 'Non-CWE',
        'Count': non_cwe_count,
        'Percentage_in_Total': f"{non_cwe_total_percentage:.2f}%",
        'Vulnerability_Count': non_cwe_vul_count,
        'Vulnerability_Percentage_in_CWE': f"{non_cwe_vul_ratio:.2f}%",
        'Vulnerability_Percentage_of_Total_Vul': f"{non_cwe_vul_percentage_of_total:.2f}%"  # 新增列
    })

    total_counted += non_cwe_count
    total_vul_counted += non_cwe_vul_count

    # 验证总和
    print(f"\n验证统计结果:")
    print(f"统计的总样本数: {total_counted}")
    print(f"实际总样本数: {total_samples}")
    print(f"统计的漏洞样本数: {total_vul_counted}")
    print(f"实际漏洞样本数: {total_vul_samples}")

    if total_counted != total_samples:
        print(f"警告: 统计的总和({total_counted})与实际总样本数({total_samples})不匹配!")
        print(f"差异: {total_samples - total_counted} 个样本")

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 保存到CSV文件
    results_df.to_csv(output_file, index=False)

    # 打印结果
    print("\n" + "=" * 80)
    print("CWE类型统计结果:")
    print("=" * 80)
    print(f"总样本数: {total_samples}")
    print(f"漏洞样本数: {total_vul_samples} ({total_vul_samples / total_samples * 100:.2f}% of total)")
    print("-" * 80)

    # 更新打印格式，增加新列
    print(f"{'CWE_Type':20} {'Count':>8} {'%_in_Total':>12} {'Vul_Count':>10} {'Vul_%_in_CWE':>12} {'Vul_%_of_Total':>14}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(
            f"{row['CWE_Type']:20} {row['Count']:8} {row['Percentage_in_Total']:>12} "
            f"{row['Vulnerability_Count']:10} {row['Vulnerability_Percentage_in_CWE']:>12} "
            f"{row['Vulnerability_Percentage_of_Total_Vul']:>14}"
        )

    print("=" * 80)
    print(f"结果已保存到: {output_file}")

    # 额外统计信息
    print(f"\n额外统计信息:")
    print(f"- 不同CWE类型的总数: {len(cwe_counts_for_sorting)}")
    print(f"- Non-CWE样本数: {non_cwe_count}")
    print(f"- 前{top_n}个CWE类型覆盖的样本数: {sum(count for _, count in top_cwe)}")
    print(f"- 漏洞样本占总样本的比例: {total_vul_samples / total_samples * 100:.2f}%")

    # 计算不同CWE类型的漏洞比例分布
    vul_ratios = []
    for cwe_type in cwe_counts:
        if cwe_type != 'Non-CWE' and cwe_counts[cwe_type] > 0:
            vul_ratio = cwe_vul_counts.get(cwe_type, 0) / cwe_counts[cwe_type] * 100
            vul_ratios.append(vul_ratio)

    if vul_ratios:
        print(f"- CWE类型的平均漏洞比例: {np.mean(vul_ratios):.2f}%")
        print(f"- CWE类型的最大漏洞比例: {np.max(vul_ratios):.2f}%")
        print(f"- CWE类型的最小漏洞比例: {np.min(vul_ratios):.2f}%")

    return results_df



"""二、提取实验数据集中，CWE数量前十名的样本，分别组成CWE测试集"""

TOP10_CWE_TYPES = [
    'CWE-119', 'CWE-20', 'CWE-399', 'CWE-264', 'CWE-416',
    'CWE-200', 'CWE-125', 'CWE-189', 'CWE-362', 'CWE-476'
]


def extract_top10_cwe_samples(
        input_file=r"D:\Python_Line-level_Vulnerability-detection\new\Most_Information_Dataset\test_with_cwe_balanced.csv",
        output_dir=r"D:\Python_Line-level_Vulnerability-detection\new\Most_Information_Dataset\test_with_cwe",
        cwe_column='CWE_id',
        top_cwe_types=None):
    """
    从测试集中提取指定Top10 CWE类型样本，并分别保存为独立CSV文件。

    Args:
        input_file: 含CWE字段的测试集路径
        output_dir: 子集输出目录
        cwe_column: CWE列名（默认: CWE_id）
        top_cwe_types: 需要提取的CWE类型列表

    Returns:
        提取统计结果DataFrame
    """
    if top_cwe_types is None:
        top_cwe_types = TOP10_CWE_TYPES

    print(f"正在读取测试集: {input_file}")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

    # 自动兜底查找CWE列
    if cwe_column not in df.columns:
        fallback_cols = [col for col in df.columns if 'cwe' in str(col).lower()]
        if fallback_cols:
            print(f"警告: 未找到指定列 '{cwe_column}'，改用 '{fallback_cols[0]}'")
            cwe_column = fallback_cols[0]
        else:
            print("错误: 数据集中未找到CWE相关列")
            print(f"可用的列: {list(df.columns)}")
            return None

    # 统一规范化，避免CWE格式不一致影响筛选
    df['_normalized_cwe'] = df[cwe_column].apply(normalize_cwe)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = []
    print("开始提取Top10 CWE样本...")

    for cwe in top_cwe_types:
        normalized_target = normalize_cwe(cwe)
        subset = df[df['_normalized_cwe'] == normalized_target].copy()

        match = re.search(r'(\d+)', normalized_target)
        cwe_id = match.group(1) if match else normalized_target.replace('-', '_')
        output_file = output_path / f"test_with_cwe_balanced_CWE{cwe_id}.csv"

        subset = subset.drop(columns=['_normalized_cwe'])
        subset.to_csv(output_file, index=False, encoding='utf-8')

        print(f"{normalized_target:10} -> {len(subset):6} 样本, 保存到: {output_file}")
        stats.append({
            'CWE_Type': normalized_target,
            'Count': len(subset),
            'Output_File': str(output_file)
        })

    summary_df = pd.DataFrame(stats)
    summary_file = output_path / 'top10_cwe_extraction_summary.csv'
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')

    print("\n" + "=" * 80)
    print("Top10 CWE样本提取完成")
    print("=" * 80)
    print(summary_df)
    print(f"汇总文件已保存到: {summary_file}")

    return summary_df






if __name__ == "__main__":
    """一、分析整体BigVul数据集漏洞分布情况"""
    parser = argparse.ArgumentParser(description='统计BigVul数据集中CWE类型的分布')
    parser.add_argument('input_file', help='BigVul数据集文件路径')
    parser.add_argument('-o', '--output', default='cwe_statistics.csv',
                        help='输出CSV文件路径 (默认: cwe_statistics.csv)')
    parser.add_argument('--vul_column', default=None,
                        help='指定漏洞标签列名 (默认: 自动查找)')

    args = parser.parse_args()

    count_cwe_types(args.input_file, args.output)



    # """二、提取实验数据集中，CWE数量前十名的样本，分别组成CWE测试集"""
    # parser = argparse.ArgumentParser(description='统计BigVul数据集中CWE类型的分布')
    # parser.add_argument('input_file', nargs='?',
    #                     help='BigVul数据集文件路径（用于CWE分布统计）')
    # parser.add_argument('-o', '--output', default='cwe_statistics.csv',
    #                     help='输出CSV文件路径 (默认: cwe_statistics.csv)')
    # parser.add_argument('--vul_column', default=None,
    #                     help='指定漏洞标签列名 (默认: 自动查找)')
    # parser.add_argument('--extract_top10_cwe', action='store_true',
    #                     help='提取测试集中Top10 CWE样本并分别保存为CSV')
    # parser.add_argument('--extract_input_file',
    #                     default=r"D:\Python_Line-level_Vulnerability-detection\new\Most_Information_Dataset\test_with_cwe_balanced.csv",
    #                     help='提取任务输入文件路径')
    # parser.add_argument('--extract_output_dir',
    #                     default=r"D:\Python_Line-level_Vulnerability-detection\new\Most_Information_Dataset\test_with_cwe",
    #                     help='提取任务输出目录')
    # parser.add_argument('--extract_cwe_column', default='CWE_id',
    #                     help='提取任务使用的CWE列名 (默认: CWE_id)')

    # args = parser.parse_args()
    # has_run_task = False

    # if args.input_file:
    #     count_cwe_types(args.input_file, args.output)
    #     has_run_task = True

    # if args.extract_top10_cwe:
    #     extract_top10_cwe_samples(
    #         input_file=args.extract_input_file,
    #         output_dir=args.extract_output_dir,
    #         cwe_column=args.extract_cwe_column,
    #         top_cwe_types=TOP10_CWE_TYPES
    #     )
    #     has_run_task = True

    # if not has_run_task:
    #     parser.print_help()
    #     print("\n提示: 统计任务请提供 input_file；提取任务请使用 --extract_top10_cwe")




