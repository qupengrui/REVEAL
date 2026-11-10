import argparse
import json
import os
import sys
import pandas as pd

"""在已合并的 BigVul 数据集上重新划分 train/val/test，并按行级粒度提取漏洞信息。

划分策略（与 merge_datasets_and_count_projects.py 的统计结果对应）:
    测试集 (test): 指定的若干 project 的全部样本（这些 project 合计约占 BigVul 的 10%）
    验证集 (val) : 从"非测试集"样本中随机抽取，数量为整体 BigVul 数据集总样本数的 10%
    训练集 (train): 其余全部样本

随后按 useful_column_toCSV.py 中"3、函数+标签、行代码+标签保存在同一文件"的
方式，把三个子集分别提取为行级数据集，保存为 train.csv / val.csv / test.csv。

用法示例（在服务器上运行）:
    python split_and_extract_unbalanced_dataset.py \\
        --test_projects openssl libtiff ffmpeg php linux

    # 只想看划分比例、不真正写文件
    python split_and_extract_unbalanced_dataset.py --test_projects ... --dry_run

默认输入为 output_dir 下的 merged_dataset.csv（由 merge_datasets_and_count_projects.py 生成）。
"""

# 图中加黑的 5 个 project（可按需在命令行用 --test_projects 覆盖）
# 若两者都为空，脚本会直接报错退出，避免生成错误的测试集。
TEST_PROJECTS = [
    'Android', 'qemu', 'php', 'ImageMagick', 'savannah',
]

# 提取后的行级数据集列顺序（与 useful_column_toCSV.py 保持一致）
OUTPUT_COLUMNS = [
    "func_id", "CWE_id", "func_after", "func_before", "fun_label",
    "flaw_line", "flaw_line_index", "lines", "line_labels",
]

# 行级提取所需的原始列
REQUIRED_COLUMNS = [
    'func_before', 'func_after', 'flaw_line', 'flaw_line_index',
]
LABEL_CANDIDATES = ['target', 'vul', 'fun_label', 'label']


def find_column(df, candidates, fallback_substr=None):
    """按候选名精确匹配查找列，找不到时按子串兜底匹配。"""
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


def find_cwe_column(df):
    """自动识别 CWE 列名（兼容不同数据集命名）。"""
    return find_column(
        df,
        ['CWE_id', 'CWE_ID', 'cwe_id', 'CWE', 'cwe', 'CWE ID', 'cwe id'],
        fallback_substr='cwe')


def validate_columns(df):
    """校验提取所需列是否齐全，缺列时给出明确报错。"""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    label_column = find_column(df, LABEL_CANDIDATES, fallback_substr='vul')
    if label_column is None:
        missing.append('target (或 vul / fun_label / label)')

    if missing:
        raise KeyError(
            "输入数据集缺少必要列: " + ", ".join(missing) + "\n"
            "可用列: " + str(list(df.columns)))

    return label_column


def split_by_project(df, test_projects, project_column, val_ratio, seed):
    """按 project 划分测试集，并从未进入测试集的样本中抽取验证集。

    Args:
        df: 合并后的完整数据集
        test_projects: 作为测试集的 project 名称列表
        project_column: project 列名
        val_ratio: 验证集占整体数据集的比例（默认 0.1）
        seed: 随机种子

    Returns:
        (train_df, val_df, test_df)
    """
    total = len(df)

    # 允许 project 名大小写/空白差异
    normalized = df[project_column].astype(str).str.strip()
    project_map = {p.strip().lower(): p for p in normalized.unique()}

    matched, unmatched = [], []
    for p in test_projects:
        key = p.strip().lower()
        if key in project_map:
            matched.append(project_map[key])
        else:
            unmatched.append(p)

    if unmatched:
        raise ValueError(
            "以下 project 在数据集中不存在: " + ", ".join(unmatched) + "\n"
            "数据集中的 project 列表: " + str(sorted(project_map.values())))

    test_mask = normalized.isin(matched)
    test_df = df[test_mask].copy()
    pool_df = df[~test_mask].copy()

    print(f"测试集 project: {matched}")
    print(f"测试集样本数: {len(test_df)} ({len(test_df) / total * 100:.2f}% of total)")

    # 验证集：从"非测试集"样本中随机抽取，数量为整体总量的 val_ratio
    val_size = int(round(total * val_ratio))
    if val_size > len(pool_df):
        print(f"警告: 计划抽取的验证集样本数({val_size})超过可用样本数({len(pool_df)})，"
              f"已调整为 {len(pool_df)}")
        val_size = len(pool_df)

    val_df = pool_df.sample(n=val_size, random_state=seed)
    train_df = pool_df.drop(index=val_df.index)

    print(f"验证集样本数: {len(val_df)} ({len(val_df) / total * 100:.2f}% of total)")
    print(f"训练集样本数: {len(train_df)} ({len(train_df) / total * 100:.2f}% of total)")

    # 校验三份互不重叠、且合起来等于全集
    assert len(train_df) + len(val_df) + len(test_df) == total, "划分后样本总数不匹配"
    assert len(set(train_df.index) & set(val_df.index)) == 0, "训练集与验证集存在重叠"

    # 每份都重新编号，便于下游按 0..n-1 使用
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def build_line_level_rows(df, label_column, cwe_column, flaw_line_1_based=False):
    """按行级粒度提取漏洞信息（对应 useful_column_toCSV.py 的第 3 步）。

    Args:
        df: 待处理子集
        label_column: 函数级标签列名
        cwe_column: CWE 列名，可为 None
        flaw_line_1_based: 是否将 flaw_line_index 视为 1-based 并减 1。
            默认 False，即沿用 useful_column_toCSV.py 的原始实现。

            该默认值已用真实数据验证过：在 data/Most_Information_Dataset/
            test_with_cwe.csv 的 788 条含 flaw_line 的样本中，有 296 条满足
            flaw_line == lines[flaw_line_index]，仅 7 条满足
            flaw_line == lines[flaw_line_index - 1]，说明 BigVul 的
            flaw_line_index 本身即 0-based，原始实现未减 1 是正确的
            （原代码注释"转 0-based"易引起误解，但代码行为正确）。
            如后续换用其它数据源且确认索引为 1-based，再打开此开关。

    Returns:
        行级 DataFrame（列顺序同 OUTPUT_COLUMNS）
    """
    new_rows = []

    for idx, row in df.iterrows():
        func_before = row['func_before']
        if pd.isna(func_before):
            # 原始实现会在此处抛异常，这里跳过并计数，避免整批中断
            print(f"  警告: 第 {idx} 行 func_before 为空，已跳过")
            continue

        func_id = idx
        func_after = row['func_after']
        vul = row[label_column]
        cwe_id = row[cwe_column] if cwe_column is not None else ""
        flaw_line = row['flaw_line']
        flaw_line_index = row['flaw_line_index']

        # 分割函数代码为行列表
        code_lines = [line.rstrip('\n') for line in func_before.split('\n')]

        # 初始化行标签（默认全 0）
        line_labels = [0] * len(code_lines)

        # 处理漏洞行（若存在漏洞）
        if pd.notna(flaw_line_index) and str(flaw_line_index).strip():
            for line_str in str(flaw_line_index).split(','):
                line_str = line_str.strip()
                if line_str:
                    try:
                        line_num = int(float(line_str))
                    except ValueError:
                        continue  # 忽略无效行号格式
                    if flaw_line_1_based:
                        line_num -= 1
                    if 0 <= line_num < len(code_lines):
                        line_labels[line_num] = 1

        new_rows.append({
            "func_id": func_id,
            "func_after": func_after,
            "func_before": func_before,
            "fun_label": vul,
            "CWE_id": cwe_id,
            "flaw_line": flaw_line,
            "flaw_line_index": flaw_line_index,
            "lines": json.dumps(code_lines, ensure_ascii=False),
            "line_labels": json.dumps(line_labels, ensure_ascii=False),
        })

    return pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS)


def save_composition(splits, project_column, path):
    """保存各子集的 project 组成，便于核对划分是否符合预期。"""
    rows = []
    for name, df in splits.items():
        for project, group in df.groupby(project_column, dropna=False):
            rows.append({
                'split': name,
                'project': project,
                'sample_count': len(group),
            })
    comp = pd.DataFrame(rows).sort_values(
        by=['split', 'sample_count'], ascending=[True, False], ignore_index=True)
    comp.to_csv(path, index=False, encoding='utf-8')
    print(f"划分组成明细已保存到: {path}")


def main():
    parser = argparse.ArgumentParser(
        description='按 project 划分 test，并从剩余样本抽 10% 作 val，其余作 train；'
                    '再按行级粒度提取并保存三个数据集')
    parser.add_argument('--input_file', default=None,
                        help='已合并的数据集路径 (默认: output_dir/merged_dataset.csv)')
    parser.add_argument('--output_dir',
                        default='/data/prqu/prqu_files/data/Most_Information_UnbalancedDataset',
                        help='输出目录 (默认: /data/prqu/prqu_files/data/Most_Information_UnbalancedDataset)')
    parser.add_argument('--test_projects', nargs='+', default=None,
                        help='作为测试集的 project 名称列表 (默认取脚本顶部的 TEST_PROJECTS)')
    parser.add_argument('--project_column', default=None,
                        help='指定 project 列名 (默认: 自动查找)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='验证集占整体数据集的比例 (默认: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--flaw-line-1-based', action='store_true',
                        help='将 flaw_line_index 视为 1-based 并减 1；默认关闭以沿用原始实现')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印划分结果，不写出任何文件')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_file = args.input_file or os.path.join(args.output_dir, 'merged_dataset.csv')

    test_projects = args.test_projects if args.test_projects else TEST_PROJECTS
    if not test_projects:
        parser.error(
            "未指定测试集 project。请用 --test_projects 传入，"
            "或直接修改脚本顶部的 TEST_PROJECTS 常量。")

    print(f"读取合并数据集: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"未找到已合并的数据集: {input_file}\n"
            f"请先运行 merge_datasets_and_count_projects.py 生成。")
    df = pd.read_csv(input_file, low_memory=False, encoding='utf-8')
    print(f"总样本数: {len(df)}")

    label_column = validate_columns(df)
    project_column = args.project_column or find_column(
        df, ['project', 'repo', 'project_name'], fallback_substr='project')
    cwe_column = find_cwe_column(df)

    if project_column is None:
        raise KeyError(
            "未找到 project 列，无法按 project 划分测试集。\n"
            "可用列: " + str(list(df.columns)))
    print(f"标签列: {label_column} | project 列: {project_column} | CWE 列: {cwe_column}")

    train_df, val_df, test_df = split_by_project(
        df, test_projects, project_column, args.val_ratio, args.seed)

    if args.dry_run:
        print("\n[dry_run] 已跳过文件写出。")
        save_composition(
            {'train': train_df, 'val': val_df, 'test': test_df},
            project_column, os.path.join(args.output_dir, 'split_composition.csv'))
        return

    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    for name, subset in splits.items():
        print(f"\n提取 {name} 集的行级漏洞信息 (样本数: {len(subset)})...")
        out_df = build_line_level_rows(
            subset, label_column, cwe_column,
            flaw_line_1_based=args.flaw_line_1_based)

        out_path = os.path.join(args.output_dir, f'{name}.csv')
        out_df.to_csv(out_path, index=False, encoding='utf-8')

        # 简要校验
        vul_funcs = int((out_df['fun_label'] == 1).sum())
        vul_lines = sum(json.loads(s).count(1) for s in out_df['line_labels'])
        print(f"  函数级: 漏洞 {vul_funcs} / 非漏洞 {len(out_df) - vul_funcs}")
        print(f"  行级  : 漏洞行 {vul_lines} / 总行数 {sum(len(json.loads(s)) for s in out_df['lines'])}")
        print(f"  已保存到: {out_path}")

    save_composition(splits, project_column,
                     os.path.join(args.output_dir, 'split_composition.csv'))


if __name__ == '__main__':
    main()
