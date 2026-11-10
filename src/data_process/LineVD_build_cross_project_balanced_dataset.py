import argparse
import os

import pandas as pd

"""为复现 LineVD 构建"跨项目"1:1 平衡数据集，并完整保留 BigVul 原始样本信息。

跨项目划分策略（与 build_cross_project_balanced_dataset.py 保持一致）:
    测试集 (test): 从指定的 5 个 project 中取"全部漏洞样本"，并在同一批 project 内
                   抽取等量非漏洞样本，构造成 1:1 测试集；
                   若某个 project 的非漏洞样本不足，则取 min(vul, non_vul)。
    训练/验证集 :  从"剩余 project"中，按 project 内部各自构造 1:1 平衡集，
                   再按 project 内部分层切分 train / val（val 占每个 project 平衡集的
                   val_ratio），保证每个 project 在 train / val 内部仍然是 1:1。

与 build_cross_project_balanced_dataset.py 的关键区别:
    - 输入直接使用原始 BigVul 的 train.csv / val.csv / test.csv（先合并），
      不再依赖已经裁剪过列的 merged_dataset.csv；
    - 选中样本后，默认原封不动地写出该样本的全部原始列（包括 id、project、
      commit、CVE ID、flaw_line、flaw_line_index 等），不做行级字段重构；
    - 原始 BigVul 的 files_changed 列是提交级文件列表，会在同一 commit 的每个
      函数行重复，单行可达数 MB。若不需要该列，可用 --drop_columns files_changed
      丢弃（LineVD 图预处理不需要它），可把测试集从数 GB 降到几十 MB；
    - 输出可直接作为 LineVD 预处理阶段（Joern 图构建）的输入。

用法示例（在服务器上运行）:
    # 默认读取 /data/prqu/prqu_files/Bigvul_dataset/{train,val,test}.csv
    python LineVD_build_cross_project_balanced_dataset.py

    # 也可以直接指定一个已合并的数据集
    python LineVD_build_cross_project_balanced_dataset.py \
        --input_file /data/prqu/prqu_files/data/Most_Information_UnbalancedDataset/merged_dataset.csv

    # 只查看划分结果，不写出数据集
    python LineVD_build_cross_project_balanced_dataset.py --dry_run
"""

# 作为跨项目测试集的 5 个 project（与 build_cross_project_balanced_dataset.py 一致）
TEST_PROJECTS = [
    'Android', 'qemu', 'php', 'ImageMagick', 'savannah',
]

DEFAULT_INPUT_DIR = "/data/prqu/prqu_files/Bigvul_dataset"
DEFAULT_OUTPUT_DIR = "/data/prqu/prqu_files/data/LineVD-Cross_project-Balanced_dataset"

LABEL_CANDIDATES = [
    'target', 'vul', 'fun_label', 'label',
    'is_vulnerability', 'vulnerability',
]
PROJECT_CANDIDATES = ['project', 'repo', 'project_name']
CWE_CANDIDATES = [
    'CWE_id', 'CWE_ID', 'cwe_id', 'CWE', 'cwe', 'CWE ID', 'cwe id',
]
ID_CANDIDATES = ['id', 'func_id', 'function_id', 'Unnamed: 0', 'index', 'idx']

# 内部工作列：把标签统一转成数值用于筛选，写出前删除，避免改动原始标签文本
WORK_LABEL_COL = "__linevd_label__"


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


def read_csv(path):
    """读取 CSV，保持原始字符串内容（不把空串/NA 当作缺失值）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"读取: {path} ({size_mb:.1f} MB)")
    return pd.read_csv(
        path, low_memory=False, encoding='utf-8', keep_default_na=False)


def load_dataset(args):
    """读取并合并原始 train/val/test，或直接读取 --input_file。"""
    if args.input_file:
        df = read_csv(args.input_file)
        return df

    files = {
        'train': os.path.join(args.input_dir, args.train_file),
        'val': os.path.join(args.input_dir, args.val_file),
        'test': os.path.join(args.input_dir, args.test_file),
    }

    frames = []
    base_columns = None
    for name, path in files.items():
        frame = read_csv(path)
        print(f"  {name}: {len(frame)} 条")
        if base_columns is None:
            base_columns = list(frame.columns)
        elif list(frame.columns) != base_columns:
            raise ValueError(
                f"{path} 的列与 {files['train']} 不一致，无法直接合并。\n"
                f"train 列: {base_columns}\n{name} 列: {list(frame.columns)}")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    print(f"合并后样本总数: {len(df)}")
    return df


def add_work_label(df, label_column):
    """新增数值标签工作列，并把标签非 0/1 的样本单独返回（不参与划分）。"""
    labels = pd.to_numeric(df[label_column], errors='coerce')
    invalid_mask = ~labels.isin([0, 1])

    invalid_df = df.loc[invalid_mask].copy()
    valid_df = df.loc[~invalid_mask].copy()
    valid_df[WORK_LABEL_COL] = labels.loc[~invalid_mask].astype(int)

    if len(invalid_df):
        print(f"  警告: {len(invalid_df)} 条样本的 '{label_column}' 不是 0/1，"
              f"将单独保存、不参与跨项目划分")
        print(f"        取值示例: {invalid_df[label_column].head(5).tolist()}")

    return valid_df, invalid_df


def _project_matched_balanced(sub, label_column, seed):
    """在一个 project 内部构造 1:1 平衡集。

    保留全部漏洞样本，并从同一 project 抽取等量非漏洞样本；
    若非漏洞不足，则取 min(vul, non_vul)，多余漏洞样本被丢弃。

    返回:
        (balanced_df, dropped_vul)
    """
    vul = sub[sub[label_column] == 1]
    non_vul = sub[sub[label_column] == 0]
    n = min(len(vul), len(non_vul))

    dropped_vul = len(vul) - n
    if dropped_vul > 0:
        vul = vul.sample(n=n, random_state=seed)

    non_vul_sampled = non_vul.sample(n=n, random_state=seed)
    return pd.concat([vul, non_vul_sampled]), dropped_vul


def split_cross_project(df, test_projects, project_column, label_column,
                        val_ratio, seed):
    """跨项目平衡划分（与 build_cross_project_balanced_dataset.py 完全一致）。

    返回:
        (train_df, val_df, test_df, dropped_vul_by_project)
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

    # ---------- 1. 测试集：每个测试 project 内部 1:1（尽量保留全部漏洞） ----------
    test_parts = []
    for p in matched:
        sub = test_pool[test_pool[project_column] == p]
        bal, _ = _project_matched_balanced(sub, label_column, seed)
        if len(bal) > 0:
            test_parts.append(bal)
    test_df = (pd.concat(test_parts, ignore_index=True)
               if test_parts else pd.DataFrame(columns=df.columns))

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
    dev_df = (pd.concat(dev_parts, ignore_index=True)
              if dev_parts else pd.DataFrame(columns=df.columns))

    # ---------- 3. 按 project 内部分层切分 train / val ----------
    train_parts, val_parts = [], []
    if len(dev_df) > 0:
        for p, sub in dev_df.groupby(project_column):
            vul = sub[sub[label_column] == 1]
            non_vul = sub[sub[label_column] == 0]

            # 每个 project 的漏洞/非漏洞各抽相同数量进验证集，保证切分后仍各自 1:1
            n_val = int(round(len(vul) * val_ratio))
            if n_val >= len(vul):
                # 样本过少时全部放入验证集，该 project 在训练集中为空
                val_parts.append(pd.concat([vul, non_vul]))
                continue

            val_vul = vul.sample(n=n_val, random_state=seed)
            val_non = non_vul.sample(n=n_val, random_state=seed)
            train_vul = vul.drop(val_vul.index)
            train_non = non_vul.drop(val_non.index)

            val_parts.append(pd.concat([val_vul, val_non]))
            train_parts.append(pd.concat([train_vul, train_non]))

    empty_df = pd.DataFrame(columns=df.columns)
    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else empty_df.copy()
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else empty_df.copy()

    # 各自打乱顺序，避免同一 project 连续排列
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df, dropped_vul_by_project


def validate_balance(splits, project_column, label_column):
    """校验每个子集整体 1:1，且每个 project 内部漏洞数 == 非漏洞数。"""
    for name, df in splits.items():
        vul = int((df[label_column] == 1).sum())
        non_vul = int((df[label_column] == 0).sum())
        assert vul == non_vul, f"{name} 集不是 1:1: 漏洞 {vul} / 非漏洞 {non_vul}"

        mismatched = []
        for project, group in df.groupby(project_column, dropna=False):
            pv = int((group[label_column] == 1).sum())
            pn = int((group[label_column] == 0).sum())
            if pv != pn:
                mismatched.append((project, pv, pn))
        assert not mismatched, f"{name} 集存在 project 内不平衡: {mismatched}"

    print("校验通过: train/val/test 整体 1:1，且每个 project 内部漏洞数 == 非漏洞数")


def check_ids(df, id_column, output_dir, dry_run):
    """检查 id 是否唯一（LineVD 以 id 作为 CPG / gitdiff 缓存文件名）。"""
    if id_column is None:
        print("\n警告: 未找到 id 列。LineVD 预处理需要唯一 id 作为文件名，"
              "请确认原始数据中是否存在 id / Unnamed: 0 等列。")
        return

    duplicate_mask = df[id_column].duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count == 0:
        print(f"\nid 列检查通过: '{id_column}' 在合并数据中唯一")
        return

    print(f"\n警告: id 列 '{id_column}' 存在重复，共 {duplicate_count} 条涉及重复。"
          f"LineVD 以 id 命名 CPG 缓存文件，重复 id 会互相覆盖。")
    print(f"      示例: {df.loc[duplicate_mask, id_column].head(10).tolist()}")
    if not dry_run:
        dup_path = os.path.join(output_dir, 'duplicate_ids.csv')
        df.loc[duplicate_mask].to_csv(dup_path, index=False, encoding='utf-8')
        print(f"      重复样本已保存到: {dup_path}")


def save_composition(splits, project_column, label_column, path):
    """保存各子集按 project 的漏洞/非漏洞组成。"""
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


def save_summary(splits, total_rows, dropped, test_projects, path):
    """保存整体采样汇总。"""
    rows = []
    for name in ('train', 'val', 'test'):
        df = splits[name]
        vul = int((df[WORK_LABEL_COL] == 1).sum())
        rows.append({
            'split': name,
            'rows': len(df),
            'vul_rows': vul,
            'non_vul_rows': len(df) - vul,
        })
    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {
        'split': 'TOTAL(source)',
        'rows': total_rows,
        'vul_rows': '',
        'non_vul_rows': '',
    }
    summary['test_projects'] = ''
    summary.loc[summary['split'] == 'test', 'test_projects'] = ','.join(test_projects)
    summary.to_csv(path, index=False, encoding='utf-8')

    if dropped:
        dropped_df = pd.DataFrame(dropped)
        dropped_path = os.path.splitext(path)[0].replace('sampling_summary', 'dropped_vul_by_project') + '.csv'
        dropped_df.to_csv(dropped_path, index=False, encoding='utf-8')
        print(f"被丢弃漏洞样本明细已保存到: {dropped_path}")


def main():
    parser = argparse.ArgumentParser(
        description='构建 LineVD 跨项目 1:1 平衡数据集：test 取 5 个 project 的漏洞并配平，'
                    'train/val 从剩余 project 内部 1:1 构造；所有原始列完整保留')
    parser.add_argument('--input_dir', default=DEFAULT_INPUT_DIR,
                        help=f'原始 train/val/test 所在目录 (默认: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--input_file', default=None,
                        help='可选：直接读取一个已合并的 CSV（优先于 --input_dir）')
    parser.add_argument('--train_file', default='train.csv',
                        help='训练集文件名 (默认: train.csv)')
    parser.add_argument('--val_file', default='val.csv',
                        help='验证集文件名 (默认: val.csv)')
    parser.add_argument('--test_file', default='test.csv',
                        help='测试集文件名 (默认: test.csv)')
    parser.add_argument('--output_dir', default=DEFAULT_OUTPUT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--test_projects', nargs='+', default=None,
                        help='作为测试集的 project 名称列表 (默认取脚本顶部 TEST_PROJECTS)')
    parser.add_argument('--project_column', default=None,
                        help='project 列名 (默认: 自动识别)')
    parser.add_argument('--label_column', default=None,
                        help='标签列名 (默认: 自动识别 target / vul / fun_label / ...)')
    parser.add_argument('--drop_columns', nargs='+', default=None,
                        help='可选：输出前丢弃的列。原始 BigVul 的 files_changed 列是'
                             '提交级文件列表，会在每个函数行重复，常达几百 KB/行，'
                             '建议设为 files_changed（LineVD 预处理不需要该列）')
    parser.add_argument('--val_ratio', type=float, default=0.10,
                        help='验证集占每个 project 平衡集的比例 (默认: 0.10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印划分结果，不写出 train/val/test 数据集')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    test_projects = args.test_projects if args.test_projects else TEST_PROJECTS
    if not test_projects:
        parser.error(
            "未指定测试集 project。请用 --test_projects 传入，"
            "或直接修改脚本顶部的 TEST_PROJECTS 常量。")

    print(f"输出目录: {args.output_dir}")
    df = load_dataset(args)
    source_columns = list(df.columns)

    drop_columns = list(dict.fromkeys(args.drop_columns or []))
    missing_drop = [c for c in drop_columns if c not in df.columns]
    if missing_drop:
        raise KeyError(
            f"--drop_columns 指定的列不存在: {missing_drop}\n"
            f"可用列: {source_columns}")

    label_column = args.label_column or find_column(
        df, LABEL_CANDIDATES, fallback_substr='vul')
    if label_column is None:
        raise KeyError(
            "未找到标签列，请用 --label_column 指定。\n"
            f"可用列: {source_columns}")
    if label_column not in df.columns:
        raise KeyError(
            f"指定的标签列 '{label_column}' 不存在。可用列: {source_columns}")

    project_column = args.project_column or find_column(
        df, PROJECT_CANDIDATES, fallback_substr='project')
    if project_column is None:
        raise KeyError(
            "未找到 project 列，无法做跨项目划分。\n"
            f"可用列: {source_columns}")

    forbidden_drop = [c for c in (label_column, project_column) if c in drop_columns]
    if forbidden_drop:
        raise ValueError(
            f"划分必需的列不能丢弃: {forbidden_drop}")

    if drop_columns:
        print(f"根据 --drop_columns 丢弃列: {drop_columns}")
        df = df.drop(columns=drop_columns)
        print(f"  丢弃后列数: {len(source_columns)} -> {len(df.columns)}")

    original_columns = list(df.columns)
    cwe_column = find_column(df, CWE_CANDIDATES, fallback_substr='cwe')
    id_column = find_column(df, ID_CANDIDATES)
    print(f"标签列: {label_column} | project 列: {project_column} | "
          f"CWE 列: {cwe_column} | id 列: {id_column}")
    print(f"输出列数: {len(original_columns)}（除 --drop_columns 外全部保留）")

    check_ids(df, id_column, args.output_dir, args.dry_run)

    working_df, invalid_df = add_work_label(df, label_column)
    if not args.dry_run and len(invalid_df):
        invalid_path = os.path.join(args.output_dir, 'invalid_label_rows.csv')
        invalid_df.to_csv(invalid_path, index=False, encoding='utf-8')
        print(f"  标签非法样本已保存到: {invalid_path}")

    train_df, val_df, test_df, dropped = split_cross_project(
        working_df, test_projects, project_column, WORK_LABEL_COL,
        args.val_ratio, args.seed)
    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    # ---------- 划分概览 ----------
    print("\n===== 跨项目划分概览 =====")
    for name in ('test', 'val', 'train'):
        sub = splits[name]
        vul = int((sub[WORK_LABEL_COL] == 1).sum())
        print(f"{name:>5}: 总 {len(sub):>6} | 漏洞 {vul:>5} | 非漏洞 {len(sub) - vul:>5} "
              f"| 漏洞占比 {vul / len(sub) * 100 if len(sub) else 0:.2f}%")

    if dropped:
        print("\n因非漏洞样本不足而被丢弃漏洞样本的 project（已按 min(vul, non_vul) 配平）:")
        for d in dropped:
            print(f"  {d['project']}: 漏洞 {d['vul_total']} / 非漏洞 {d['non_vul_total']} "
                  f"-> 丢弃漏洞 {d['dropped_vul']}")

    validate_balance(splits, project_column, WORK_LABEL_COL)

    # ---------- 报告 ----------
    save_composition(
        splits, project_column, WORK_LABEL_COL,
        os.path.join(args.output_dir, 'split_composition.csv'))
    save_summary(
        splits, len(df), dropped, test_projects,
        os.path.join(args.output_dir, 'sampling_summary.csv'))

    if args.dry_run:
        print("\n[dry_run] 已跳过 train/val/test 数据集写出。")
        return

    # ---------- 原样写出全部原始列 ----------
    for name, subset in splits.items():
        out_df = subset.drop(columns=[WORK_LABEL_COL], errors='ignore')
        # 恢复源数据的列顺序，确保没有增删任何原始列
        out_df = out_df.loc[:, original_columns]
        out_path = os.path.join(args.output_dir, f'{name}.csv')
        out_df.to_csv(out_path, index=False, encoding='utf-8')

        vul = int((subset[WORK_LABEL_COL] == 1).sum())
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"\n{name}: {len(out_df)} 条 (漏洞 {vul} / 非漏洞 {len(out_df) - vul})"
              f" | {size_mb:.1f} MB -> {out_path}")

    if drop_columns:
        print(f"\n完成。已按 --drop_columns 丢弃列: {drop_columns}；"
              f"其余原始列全部保留。")
    else:
        print("\n完成。所有输出均保留源数据全部列，未做任何字段裁剪。")


if __name__ == '__main__':
    main()
