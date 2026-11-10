import argparse
import os
import sys

import pandas as pd

"""为复现 LineVD 构造 1:1 平衡的 train / val / test 数据集。

处理流程（每个子集独立执行，互不影响）:
    1. 读取原始 train.csv / val.csv / test.csv；
    2. 保留全部漏洞样本（label == 1）；
    3. 从非漏洞样本（label == 0）中随机抽取与漏洞样本等量的样本；
    4. 合并、打乱，并把"该样本的所有列"原样写出，不做任何字段裁剪，
       以便 LineVD 在预处理阶段（Joern 图构建等）使用完整信息。

输出目录:
    /data/prqu/prqu_files/data/LineVD_All_Information/
    下含 train.csv / val.csv / test.csv 以及 sampling_summary.csv。

用法示例（在服务器上运行）:
    python LineVD_Balanced-Dataset.py

    # 指定其它输入目录（例如已提取好的行级数据集）
    python LineVD_Balanced-Dataset.py \
        --input_dir /data/prqu/prqu_files/data/Most_Information_UnbalancedDataset

    # 只查看采样统计，不写文件
    python LineVD_Balanced-Dataset.py --dry_run
"""

# 默认输入：README 中下载的原始 BigVul train/val/test
DEFAULT_INPUT_DIR = "/data/prqu/prqu_files/Bigvul_dataset"
# 默认输出：保留样本全部列，供 LineVD 预处理使用
DEFAULT_OUTPUT_DIR = "/data/prqu/prqu_files/data/LineVD_All_Information"

# 不同数据集对标签列 / CWE 列的命名可能不同，按优先级自动识别
LABEL_CANDIDATES = [
    'target', 'vul', 'fun_label', 'label',
    'is_vulnerability', 'vulnerability',
]
CWE_CANDIDATES = [
    'CWE_id', 'CWE_ID', 'cwe_id', 'CWE', 'cwe', 'CWE ID', 'cwe id',
]


def find_column(df, candidates, fallback_substr=None):
    """按候选名精确匹配（大小写不敏感），找不到时按子串兜底。"""
    for cand in candidates:
        if cand in df.columns:
            return cand
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if fallback_substr:
        for col in df.columns:
            if fallback_substr.lower() in str(col).lower():
                return col
    return None


def load_split(path):
    """读取子集，保持原始字符串内容（不把空串/NA 当作缺失值）。"""
    print(f"读取: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    # keep_default_na=False: 保证空字段写回时仍为空，不引入 NaN 文本差异
    return pd.read_csv(
        path, low_memory=False, encoding='utf-8', keep_default_na=False)


def downsample_split(df, label_column, seed, strict_1_to_1):
    """保留全部漏洞样本，随机抽取等量非漏洞样本。

    返回:
        balanced_df: 打乱后的平衡数据集（源数据所有列均保留）
        stats:       采样统计字典
    """
    labels = pd.to_numeric(df[label_column], errors='coerce')

    invalid_mask = ~labels.isin([0, 1])
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(f"  警告: {invalid_count} 条样本的 '{label_column}' 不是 0/1，已排除")
        print(f"        取值示例: {df.loc[invalid_mask, label_column].head(5).tolist()}")

    vul_df = df[labels == 1]
    non_vul_df = df[labels == 0]

    n_vul = len(vul_df)
    n_non_vul = len(non_vul_df)

    if n_vul == 0:
        raise ValueError(
            f"数据集中没有漏洞样本（'{label_column}' == 1），无法构造平衡集。"
            f"请检查输入文件或标签列 {label_column} 是否正确。")

    # 下采样：非漏洞样本至多保留与漏洞样本相同的数量
    n_non_vul_keep = min(n_vul, n_non_vul)
    if n_non_vul < n_vul:
        message = (
            f"非漏洞样本不足: 漏洞 {n_vul} 条，非漏洞仅 {n_non_vul} 条，"
            f"无法达到严格 1:1（已保留全部非漏洞样本，未做上采样）。")
        if strict_1_to_1:
            raise ValueError(message)
        print(f"  警告: {message}")

    non_vul_sampled = non_vul_df.sample(n=n_non_vul_keep, random_state=seed)

    balanced = pd.concat([vul_df, non_vul_sampled], axis=0)
    # 打乱顺序，避免漏洞/非漏洞分段排列（与 useful_column_toCSV.py 一致）
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)

    stats = {
        'original_rows': len(df),
        'invalid_label_rows': invalid_count,
        'vul_rows': n_vul,
        'non_vul_rows': n_non_vul,
        'sampled_non_vul_rows': n_non_vul_keep,
        'dropped_non_vul_rows': n_non_vul - n_non_vul_keep,
        'balanced_rows': len(balanced),
        'vul_ratio_after': (n_vul / len(balanced)) if len(balanced) else 0.0,
    }
    return balanced, stats


def process_split(name, input_path, output_dir, label_column, cwe_column,
                  seed, strict_1_to_1, dry_run):
    """处理单个子集：读取 -> 下采样 -> 原样写出全部列。"""
    print(f"\n===== 处理 {name} 集 =====")
    df = load_split(input_path)

    if label_column is None:
        raise KeyError(
            f"{input_path} 中未找到标签列（候选: {LABEL_CANDIDATES}）。\n"
            f"可用列: {list(df.columns)}")
    if label_column not in df.columns:
        raise KeyError(
            f"指定的标签列 '{label_column}' 不存在。可用列: {list(df.columns)}")
    if cwe_column is not None and cwe_column not in df.columns:
        raise KeyError(
            f"指定的 CWE 列 '{cwe_column}' 不存在。可用列: {list(df.columns)}")

    balanced, stats = downsample_split(
        df, label_column, seed, strict_1_to_1)

    print(f"  源数据列数: {len(df.columns)}（输出时全部保留）")
    print(f"  标签列: {label_column} | CWE 列: {cwe_column}")
    print(f"  原始样本: {stats['original_rows']} "
          f"(漏洞 {stats['vul_rows']} / 非漏洞 {stats['non_vul_rows']})")
    print(f"  下采样后: {stats['balanced_rows']} "
          f"(漏洞 {stats['vul_rows']} / 非漏洞 {stats['sampled_non_vul_rows']})")
    print(f"  丢弃非漏洞样本: {stats['dropped_non_vul_rows']}")

    output_path = os.path.join(output_dir, f"{name}.csv")
    if dry_run:
        print(f"  [dry_run] 跳过写出: {output_path}")
    else:
        balanced.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  已保存到: {output_path}")

    stats.update({
        'split': name,
        'source_file': input_path,
        'output_file': output_path,
        'label_column': label_column,
        'cwe_column': cwe_column,
        'num_columns': len(df.columns),
    })
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='对 train/val/test 分别下采样至漏洞:非漏洞=1:1，并保留全部信息'
                    '供 LineVD 预处理使用')
    parser.add_argument('--input_dir', default=DEFAULT_INPUT_DIR,
                        help=f'原始数据集目录 (默认: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--train_file', default='train.csv',
                        help='训练集文件名 (默认: train.csv)')
    parser.add_argument('--val_file', default='val.csv',
                        help='验证集文件名 (默认: val.csv)')
    parser.add_argument('--test_file', default='test.csv',
                        help='测试集文件名 (默认: test.csv)')
    parser.add_argument('--label_column', default=None,
                        help='标签列名，默认自动识别 (target / vul / fun_label / ...)')
    parser.add_argument('--cwe_column', default=None,
                        help='CWE 列名，默认自动识别；仅用于日志展示')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机采样种子 (默认: 42)')
    parser.add_argument('--strict_1_to_1', action='store_true',
                        help='非漏洞样本不足时直接报错，而不是尽量平衡')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印采样统计，不写出文件')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = {
        'train': os.path.join(args.input_dir, args.train_file),
        'val': os.path.join(args.input_dir, args.val_file),
        'test': os.path.join(args.input_dir, args.test_file),
    }

    # 只读取表头来确定标签列 / CWE 列（三个子集列结构应一致）
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    if not os.path.exists(files['train']):
        raise FileNotFoundError(f"文件不存在: {files['train']}")
    probe = pd.read_csv(
        files['train'], nrows=0, encoding='utf-8', keep_default_na=False)
    label_column = args.label_column or find_column(
        probe, LABEL_CANDIDATES, fallback_substr='vul')
    cwe_column = args.cwe_column or find_column(
        probe, CWE_CANDIDATES, fallback_substr='cwe')
    del probe

    if label_column is None:
        raise KeyError(
            "未能自动识别标签列，请用 --label_column 指定。\n"
            f"可用列: {list(pd.read_csv(files['train'], nrows=0).columns)}")

    all_stats = []
    for name, path in files.items():
        stats = process_split(
            name=name,
            input_path=path,
            output_dir=args.output_dir,
            label_column=label_column,
            cwe_column=cwe_column,
            seed=args.seed,
            strict_1_to_1=args.strict_1_to_1,
            dry_run=args.dry_run,
        )
        all_stats.append(stats)

    summary = pd.DataFrame(all_stats)
    summary_path = os.path.join(args.output_dir, 'sampling_summary.csv')
    if args.dry_run:
        print(f"\n[dry_run] 跳过写出汇总: {summary_path}")
    else:
        summary.to_csv(summary_path, index=False, encoding='utf-8')
        print(f"\n采样汇总已保存到: {summary_path}")

    print("\n===== 采样汇总 =====")
    columns = [
        'split', 'original_rows', 'vul_rows', 'non_vul_rows',
        'sampled_non_vul_rows', 'balanced_rows', 'dropped_non_vul_rows',
    ]
    print(summary[columns].to_string(index=False))


if __name__ == '__main__':
    main()
