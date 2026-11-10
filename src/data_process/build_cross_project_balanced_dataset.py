import argparse
import json
import os
import pandas as pd

"""在已合并的 BigVul 数据集上构建"跨项目"平衡数据集（train / val / test 均为 1:1）。

跨项目划分策略（与 merge_datasets_and_count_projects.py 的统计结果对应）:
    测试集 (test): 从指定的 5 个 project 中取"全部漏洞样本"，并在同一批 project 内
                   抽取等量非漏洞样本，构造成 1:1 测试集。
                   由于这 5 个 project 的非漏洞样本都远多于漏洞样本，
                   "取全部漏洞 + 同 project 抽取等量非漏洞"可完整保留所有漏洞样本。
    训练/验证集 :  从"剩余 BigVul 数据集"（即不属于上述 5 个 project 的所有样本）中，
                   按 project 内部构造 1:1 平衡开发集，再按 project 内部分层切分为
                   train / val。
                   "漏洞与非漏洞来自同一 project" 是通过"每个 project 内部各自 1:1"
                   保证的：每个 project 保留全部漏洞样本，并从该 project 自身抽取
                   等量非漏洞样本；极少 project 非漏洞不足时取 min(vul, non_vul)，
                   多余漏洞样本被丢弃并打印出来（总计约十几个样本）。

随后按 useful_column_toCSV.py 中"3、函数+标签、行代码+标签保存在同一文件"的方式，
把三个子集分别提取为行级数据集，保存为 train.csv / val.csv / test.csv。

用法示例（在服务器上运行）:
    # 默认读取 Most_Information_UnbalancedDataset/merged_dataset.csv，
    # 输出到 Cross-project_Balanced-dataset/
    python build_cross_project_balanced_dataset.py

    # 只想看划分比例、不真正写文件
    python build_cross_project_balanced_dataset.py --dry_run

输出列与之前完全一致:
    func_id, CWE_id, func_after, func_before, fun_label,
    flaw_line, flaw_line_index, lines, line_labels
"""

# 作为跨项目测试集的 5 个 project（与 split_and_extract_unbalanced_dataset.py 一致，
# 合计约占 BigVul 总样本的 10% 左右）
TEST_PROJECTS = [
    'Android', 'qemu', 'php', 'ImageMagick', 'savannah',
]

# 提取后的行级数据集列顺序（与 useful_column_toCSV.py 保持一致）
OUTPUT_COLUMNS = [
    "func_id", "CWE_id", "func_after", "func_before", "fun_label",
    "flaw_line", "flaw_line_index", "lines", "line_labels",
]

# 行级提取所需的原始列
REQUIRED_COLUMNS = ['func_before', 'func_after', 'flaw_line', 'flaw_line_index']
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


def _project_matched_balanced(sub, label_column, seed):
    """把一个 project 内的子集构造为 1:1 平衡集（漏洞与非漏洞来自同一 project）。

    保留全部漏洞样本，并从同一 project 抽取等量非漏洞样本；
    若非漏洞不足，则取 min(vul, non_vul)，多余漏洞样本被丢弃。

    Returns:
        (balanced_df, dropped_vul) —— balanced_df 为该 project 的 1:1 子集，
        dropped_vul 为因非漏洞不足而被丢弃的漏洞样本数（仅该 project 内）。
    """
    vul = sub[sub[label_column] == 1]
    non_vul = sub[sub[label_column] == 0]
    n = min(len(vul), len(non_vul))

    dropped_vul = len(vul) - n
    if dropped_vul > 0:
        vul = vul.sample(n=n, random_state=seed)

    non_vul_sampled = non_vul.sample(n=n, random_state=seed)
    return pd.concat([vul, non_vul_sampled]), dropped_vul


def split_cross_project(df, test_projects, project_column, label_column, val_ratio, seed):
    """跨项目平衡划分：test 取 5 个 project 全部漏洞并配平；train/val 从剩余数据
    按 project 内部 1:1 构造并切分。

    Args:
        df: 合并后的完整数据集
        test_projects: 作为测试集的 project 名称列表
        project_column: project 列名
        label_column: 漏洞标签列名
        val_ratio: 验证集占"平衡开发集"的比例（默认 0.1）
        seed: 随机种子

    Returns:
        (train_df, val_df, test_df, dropped_vul_by_project)
        dropped_vul_by_project: 因非漏洞不足被丢弃漏洞样本的 project 明细列表
    """
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
    test_pool = df[test_mask].copy()
    dev_pool = df[~test_mask].copy()

    # ---------- 1. 测试集：每个测试 project 内部 1:1（保留全部漏洞） ----------
    test_parts = []
    test_dropped = 0
    for p in matched:
        sub = test_pool[test_pool[project_column] == p]
        bal, dropped = _project_matched_balanced(sub, label_column, seed)
        test_dropped += dropped
        if len(bal) > 0:
            test_parts.append(bal)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    # ---------- 2. 开发集（train + val）：每个剩余 project 内部 1:1 ----------
    dev_parts = []
    dropped_vul_by_project = []
    for p, sub in dev_pool.groupby(project_column):
        bal, dropped = _project_matched_balanced(sub, label_column, seed)
        if dropped > 0:
            dropped_vul_by_project.append({
                'project': p,
                'vul_total': int((sub[label_column] == 1).sum()),
                'non_vul_total': int((sub[label_column] == 0).sum()),
                'dropped_vul': dropped,
            })
        if len(bal) > 0:
            dev_parts.append(bal)
    dev_df = pd.concat(dev_parts, ignore_index=True) if dev_parts else pd.DataFrame()

    # ---------- 3. 按 project 内部分层切分 train / val ----------
    train_parts, val_parts = [], []
    for p, sub in dev_df.groupby(project_column):
        vul = sub[sub[label_column] == 1]
        non_vul = sub[sub[label_column] == 0]

        # 每个 project 的漏洞/非漏洞各抽相同比例进验证集，保证切分后仍各自 1:1
        n_val = int(round(len(vul) * val_ratio))
        if n_val >= len(vul):
            # 样本过少时全部放入验证集，训练集该 project 为空
            val_parts.append(pd.concat([vul, non_vul]))
            continue

        val_vul = vul.sample(n=n_val, random_state=seed)
        val_non = non_vul.sample(n=n_val, random_state=seed)
        train_vul = vul.drop(val_vul.index)
        train_non = non_vul.drop(val_non.index)

        val_parts.append(pd.concat([val_vul, val_non]))
        train_parts.append(pd.concat([train_vul, train_non]))

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()

    # 各自打乱顺序，避免同 project 连续排列
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df, dropped_vul_by_project


def build_line_level_rows(df, label_column, cwe_column, flaw_line_1_based=False):
    """按行级粒度提取漏洞信息（对应 useful_column_toCSV.py 的第 3 步）。

    与 split_and_extract_unbalanced_dataset.py 完全一致；flaw_line_index 默认
    视为 0-based（已用真实数据验证，见该脚本内的说明）。
    """
    new_rows = []

    for idx, row in df.iterrows():
        func_before = row['func_before']
        if pd.isna(func_before):
            print(f"  警告: 第 {idx} 行 func_before 为空，已跳过")
            continue

        func_after = row['func_after']
        vul = row[label_column]
        cwe_id = row[cwe_column] if cwe_column is not None else ""
        flaw_line = row['flaw_line']
        flaw_line_index = row['flaw_line_index']

        code_lines = [line.rstrip('\n') for line in func_before.split('\n')]
        line_labels = [0] * len(code_lines)

        if pd.notna(flaw_line_index) and str(flaw_line_index).strip():
            for line_str in str(flaw_line_index).split(','):
                line_str = line_str.strip()
                if line_str:
                    try:
                        line_num = int(float(line_str))
                    except ValueError:
                        continue
                    if flaw_line_1_based:
                        line_num -= 1
                    if 0 <= line_num < len(code_lines):
                        line_labels[line_num] = 1

        new_rows.append({
            "func_id": idx,
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


def save_composition(splits, project_column, label_column, path):
    """保存各子集按 project 的 漏洞/非漏洞 组成，用于核对 1:1 与"同 project"性质。"""
    rows = []
    for name, df in splits.items():
        for project, group in df.groupby(project_column, dropna=False):
            vul = int((group[label_column] == 1).sum())
            rows.append({
                'split': name,
                'project': project,
                'vul_count': vul,
                'non_vul_count': len(group) - vul,
                'total': len(group),
            })
    comp = pd.DataFrame(rows).sort_values(
        by=['split', 'total'], ascending=[True, False], ignore_index=True)
    comp.to_csv(path, index=False, encoding='utf-8')
    print(f"划分组成明细已保存到: {path}")
    return comp


def validate_balance(splits, project_column, label_column):
    """校验每个子集整体 1:1，且每个子集中每个 project 的 漏洞数 == 非漏洞数。"""
    for name, df in splits.items():
        vul = int((df[label_column] == 1).sum())
        non_vul = int((df[label_column] == 0).sum())
        assert vul == non_vul, f"{name} 集不是 1:1: 漏洞 {vul} / 非漏洞 {non_vul}"

        mismatched = []
        for project, group in df.groupby(project_column, dropna=False):
            pv = int((group[label_column] == 1).sum())
            pn = len(group) - pv
            if pv != pn:
                mismatched.append((project, pv, pn))
        assert not mismatched, f"{name} 集存在 project 内不平衡: {mismatched}"
    print("校验通过: train/val/test 整体 1:1，且每个 project 内部 漏洞数 == 非漏洞数")


def main():
    parser = argparse.ArgumentParser(
        description='跨项目平衡数据集构造: test 取 5 个 project 全部漏洞并配平 1:1，'
                    'train/val 从剩余数据按 project 内部 1:1 构造；'
                    '再按行级粒度提取并保存三个数据集')
    parser.add_argument('--input_file',
                        default='/data/prqu/prqu_files/data/Most_Information_UnbalancedDataset/merged_dataset.csv',
                        help='已合并的数据集路径 (默认: Most_Information_UnbalancedDataset/merged_dataset.csv)')
    parser.add_argument('--output_dir',
                        default='/data/prqu/prqu_files/data/Cross-project_Balanced-dataset',
                        help='输出目录 (默认: /data/prqu/prqu_files/data/Cross-project_Balanced-dataset)')
    parser.add_argument('--test_projects', nargs='+', default=None,
                        help='作为测试集的 project 名称列表 (默认取脚本顶部的 TEST_PROJECTS)')
    parser.add_argument('--project_column', default=None,
                        help='指定 project 列名 (默认: 自动查找)')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='验证集占平衡开发集的比例 (默认: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--flaw-line-1-based', action='store_true',
                        help='将 flaw_line_index 视为 1-based 并减 1；默认关闭以沿用原始实现')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印划分结果，不写出任何文件')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test_projects = args.test_projects if args.test_projects else TEST_PROJECTS
    if not test_projects:
        parser.error(
            "未指定测试集 project。请用 --test_projects 传入，"
            "或直接修改脚本顶部的 TEST_PROJECTS 常量。")

    print(f"读取合并数据集: {args.input_file}")
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(
            f"未找到已合并的数据集: {args.input_file}\n"
            f"请先运行 merge_datasets_and_count_projects.py 生成。")
    df = pd.read_csv(args.input_file, low_memory=False, encoding='utf-8')
    print(f"总样本数: {len(df)}")

    label_column = validate_columns(df)
    project_column = args.project_column or find_column(
        df, ['project', 'repo', 'project_name'], fallback_substr='project')
    cwe_column = find_cwe_column(df)

    if project_column is None:
        raise KeyError(
            "未找到 project 列，无法按 project 划分。\n"
            "可用列: " + str(list(df.columns)))
    print(f"标签列: {label_column} | project 列: {project_column} | CWE 列: {cwe_column}")

    train_df, val_df, test_df, dropped = split_cross_project(
        df, test_projects, project_column, label_column,
        args.val_ratio, args.seed)

    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    # 打印划分概览
    print("\n===== 跨项目划分概览 =====")
    total_vul = int((df[label_column] == 1).sum())
    for name in ('test', 'val', 'train'):
        sub = splits[name]
        vul = int((sub[label_column] == 1).sum())
        print(f"{name:>5}: 总 {len(sub):>6} | 漏洞 {vul:>5} | 非漏洞 {len(sub) - vul:>5} "
              f"| 漏洞占比 {vul / len(sub) * 100:.2f}%")

    if dropped:
        print("\n因非漏洞样本不足而被丢弃漏洞样本的 project（已按 min(vul, non_vul) 配平）:")
        for d in dropped:
            print(f"  {d['project']}: 漏洞 {d['vul_total']} / 非漏洞 {d['non_vul_total']} "
                  f"-> 丢弃漏洞 {d['dropped_vul']}")
    print(f"\n合并数据集漏洞样本总数: {total_vul}")
    print(f"测试集使用漏洞样本: {int((test_df[label_column] == 1).sum())} "
          f"(来自 5 个测试 project 的全部漏洞)")

    validate_balance(splits, project_column, label_column)

    if args.dry_run:
        print("\n[dry_run] 已跳过文件写出。")
        save_composition(splits, project_column, label_column,
                         os.path.join(args.output_dir, 'split_composition.csv'))
        return

    # 逐个子集提取行级漏洞信息并保存
    for name, subset in splits.items():
        print(f"\n提取 {name} 集的行级漏洞信息 (样本数: {len(subset)})...")
        out_df = build_line_level_rows(
            subset, label_column, cwe_column,
            flaw_line_1_based=args.flaw_line_1_based)

        out_path = os.path.join(args.output_dir, f'{name}.csv')
        out_df.to_csv(out_path, index=False, encoding='utf-8')

        vul_funcs = int((out_df['fun_label'] == 1).sum())
        vul_lines = sum(json.loads(s).count(1) for s in out_df['line_labels'])
        print(f"  函数级: 漏洞 {vul_funcs} / 非漏洞 {len(out_df) - vul_funcs}")
        print(f"  行级  : 漏洞行 {vul_lines} / 总行数 {sum(len(json.loads(s)) for s in out_df['lines'])}")
        print(f"  已保存到: {out_path}")

    save_composition(splits, project_column, label_column,
                     os.path.join(args.output_dir, 'split_composition.csv'))


if __name__ == '__main__':
    main()
