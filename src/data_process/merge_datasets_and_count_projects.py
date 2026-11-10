import argparse
import os
import pandas as pd

"""合并 BigVul 数据集的 train/val/test 三个子集为一个数据集，
并统计不同 project 下的所有样本数量。

用法示例（在服务器上运行）:
    # 1) 首次运行：合并 train/val/test 并统计
    python merge_datasets_and_count_projects.py

    # 2) 已有 merged_dataset.csv，仅重新统计（不重复合并）
    python merge_datasets_and_count_projects.py --only_stats

默认读取 /data/prqu/prqu_files/Bigvul_dataset/ 下的
    train.csv / val.csv / test.csv，
合并结果保存为 /data/prqu/prqu_files/data/Most_Information_UnbalancedDataset/merged_dataset.csv，
项目统计结果保存为同目录下的 project_statistics.csv。
"""


def find_column(df, candidates, fallback_substr=None):
    """按候选名精确匹配查找列，找不到时按子串兜底匹配。

    Args:
        df: DataFrame
        candidates: 候选列名列表（按优先级）
        fallback_substr: 兜底子串（不区分大小写）

    Returns:
        匹配到的列名，找不到时返回 None
    """
    for cand in candidates:
        if cand in df.columns:
            return cand
    # 不区分大小写精确匹配
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # 子串兜底匹配
    if fallback_substr:
        for col in df.columns:
            if fallback_substr.lower() in str(col).lower():
                return col
    return None


def find_label_column(df):
    """自动查找漏洞标签列（优先 target，其次 vul / fun_label）。"""
    return find_column(df, ['target', 'vul', 'fun_label', 'label'], fallback_substr='vul')


def find_project_column(df):
    """自动查找 project 列。"""
    return find_column(df, ['project', 'repo', 'project_name'], fallback_substr='project')


def load_split(path):
    """读取单个子集 CSV。"""
    print(f"读取: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path, low_memory=False, encoding='utf-8')


def merge_datasets(input_dir, train_file, val_file, test_file, add_split_column):
    """读取并合并 train/val/test 三个子集。"""
    train = load_split(os.path.join(input_dir, train_file))
    val = load_split(os.path.join(input_dir, val_file))
    test = load_split(os.path.join(input_dir, test_file))

    print(f"train 样本数: {len(train)}")
    print(f"val   样本数: {len(val)}")
    print(f"test  样本数: {len(test)}")

    if add_split_column:
        train = train.copy()
        val = val.copy()
        test = test.copy()
        train['split'] = 'train'
        val['split'] = 'val'
        test['split'] = 'test'

    merged = pd.concat([train, val, test], ignore_index=True)
    print(f"合并后总样本数: {len(merged)}")
    return merged


def count_by_project(merged, project_column, label_column, stats_path):
    """按 project 统计样本数量并保存。"""
    if project_column is None:
        print("警告: 未找到 project 列，无法按项目统计。")
        print(f"可用列: {list(merged.columns)}")
        return None

    print(f"按列 '{project_column}' 统计各 project 样本数量...")

    total_samples = len(merged)

    records = []
    # groupby 保留计数，可应对 project 含空值的情况
    for project, group in merged.groupby(project_column, dropna=False):
        sample_count = len(group)
        record = {
            'project': project,
            'sample_count': sample_count,
            # 该 project 样本数占整体 BigVul 数据集的比重
            'sample_percentage': f"{sample_count / total_samples * 100:.4f}%" if total_samples > 0 else "0.0000%",
        }
        if label_column is not None:
            vul_count = int((group[label_column] == 1).sum())
            record['vul_count'] = vul_count
            record['non_vul_count'] = sample_count - vul_count
            record['vul_ratio'] = f"{vul_count / sample_count * 100:.2f}%" if sample_count > 0 else "0.00%"
        records.append(record)

    stats_df = pd.DataFrame(records).sort_values(
        by='sample_count', ascending=False, ignore_index=True)

    # 累计占比：便于按 10% 阈值切分测试集/验证集与训练集
    # （从上往下累积，达到 10% 时前面的 project 即可作为测试集/验证集的候选）
    if total_samples > 0:
        cum_counts = stats_df['sample_count'].cumsum()
        stats_df['cumulative_percentage'] = (cum_counts / total_samples * 100).map(lambda v: f"{v:.4f}%")

    stats_df.to_csv(stats_path, index=False, encoding='utf-8')
    print(f"项目统计结果已保存到: {stats_path}")
    print(f"共 {len(stats_df)} 个不同的 project，样本总数: {stats_df['sample_count'].sum()}")
    return stats_df


def main():
    parser = argparse.ArgumentParser(
        description='合并 BigVul 的 train/val/test 数据集，并统计各 project 样本数量')
    parser.add_argument('--input_dir', default='/data/prqu/prqu_files/Bigvul_dataset',
                        help='输入目录，包含 train/val/test 三个 CSV (默认: /data/prqu/prqu_files/Bigvul_dataset)')
    parser.add_argument('--output_dir', default='/data/prqu/prqu_files/data/Most_Information_UnbalancedDataset',
                        help='输出目录 (默认: /data/prqu/prqu_files/data/Most_Information_UnbalancedDataset)')
    parser.add_argument('--train_file', default='train.csv', help='训练集文件名 (默认: train.csv)')
    parser.add_argument('--val_file', default='val.csv', help='验证集文件名 (默认: val.csv)')
    parser.add_argument('--test_file', default='test.csv', help='测试集文件名 (默认: test.csv)')
    parser.add_argument('--merged_file', default='merged_dataset.csv',
                        help='合并后的数据集文件名 (默认: merged_dataset.csv)')
    parser.add_argument('--stats_file', default='project_statistics.csv',
                        help='项目统计结果文件名 (默认: project_statistics.csv)')
    parser.add_argument('--project_column', default=None,
                        help='指定 project 列名 (默认: 自动查找)')
    parser.add_argument('--no-split-column', action='store_true',
                        help='不添加表示来源子集(train/val/test)的 split 列')
    parser.add_argument('--only_stats', action='store_true',
                        help='跳过合并步骤，直接读取 output_dir 下已存在的 merged_file 重新统计')

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    merged_path = os.path.join(output_dir, args.merged_file)

    if args.only_stats:
        # 合并步骤已执行过，直接读取已有合并数据集，仅重新统计
        print(f"[仅统计模式] 跳过合并，直接读取: {merged_path}")
        if not os.path.exists(merged_path):
            raise FileNotFoundError(
                f"未找到已合并的数据集: {merged_path}\n"
                f"请先不带 --only_stats 运行一次以生成合并数据集。")
        merged = pd.read_csv(merged_path, low_memory=False, encoding='utf-8')
        print(f"已读取合并数据集，总样本数: {len(merged)}")
    else:
        merged = merge_datasets(
            args.input_dir, args.train_file, args.val_file, args.test_file,
            add_split_column=not args.no_split_column)

        merged.to_csv(merged_path, index=False, encoding='utf-8')
        print(f"合并数据集已保存到: {merged_path}")

    label_column = find_label_column(merged)
    project_column = args.project_column or find_project_column(merged)
    print(f"漏洞标签列: {label_column}")
    print(f"project 列: {project_column}")

    stats_path = os.path.join(output_dir, args.stats_file)
    stats_df = count_by_project(merged, project_column, label_column, stats_path)

    # 控制台简要汇总
    if stats_df is not None:
        print("\n各 project 样本数量 (前 20):")
        print(stats_df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
