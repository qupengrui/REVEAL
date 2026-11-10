import argparse
import gc
import json
import os
import random
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

"""UnixCoder 函数级漏洞检测：5 折交叉验证。

与 Only_FunVD_Unixcoder_FineTune.py 的训练逻辑保持一致（只使用函数级输入、
冻结 UnixCoder 并解冻最后 6 个 encoder layer、按验证集 F1 选择最佳模型），
不同之处是一次运行 5 折，并汇总每折测试集指标。

默认直接使用远程已经切分好的每折数据集，不再重新切分：
    /data/prqu/FuncVul/5-fold-cv_dataset_split/
        fold1/train.csv
        fold1/val.csv
        fold1/test.csv
        ...
        fold5/train.csv
        fold5/val.csv
        fold5/test.csv

同时兼容 fold_1 / fold-1 / Fold1 / fold01 等文件夹命名；如果某折没有
val.csv，会从该折 train.csv 中分层切分出一部分作为验证集（默认 10%）。

每折训练完成后，只保留验证集 F1 最高的 1 个模型，并在该折独立测试集上
计算 F1、Accuracy、Precision、Recall。最终汇总 5 折的 mean、std 和
95% 置信区间（ci95，区间形式），同时给出各折 Min-Max。

默认每折训练 30 个 epoch（与 Only_FunVD_CodeT5_5fold-cv.py 当前默认值一致），
可通过 --epochs 覆盖。

所有结果保存到：
    /data/prqu/prqu_files/save_file/FuncVD-Unixcoder_5fold-cv/
        cv_per_fold.csv          # 每折测试集/验证集指标（宽表）
        per_fold_test_metrics.csv# 每折 F1/Accuracy/Precision/Recall（简洁表）
        cv_per_fold_long.csv     # 每折指标（长表）
        cv_summary.csv           # mean/std/ci95/min-max（含 test/val）
        final_test_summary.csv   # 最终测试集 mean/std/ci95/min-max
        cv_summary.txt           # 可直接阅读的汇总
        fold_discovery.csv       # 每折使用的数据路径
        run_config.json          # 本次运行配置
        training_cost.csv        # 每折/每 epoch 耗时与显存
        training_log_all_folds.txt
        tensorboard/<run_id>/
        fold1/
            metrics.csv              # 该折 F1/Accuracy/Precision/Recall
            training_log.txt
            models/epoch_xxx__f1_xxxx.pth
        ...

用法（建议先 dry_run 检查数据路径）：
    python Only_FunVD_Unixcoder_5fold-cv.py --dry_run
    python Only_FunVD_Unixcoder_5fold-cv.py

只运行部分折：
    python Only_FunVD_Unixcoder_5fold-cv.py --folds 1 2

快速冒烟测试（1 个 epoch）：
    python Only_FunVD_Unixcoder_5fold-cv.py --epochs 1 --folds 1
"""

DEFAULT_DATA_DIR = "/data/prqu/FuncVul/5-fold-cv_dataset_split"
DEFAULT_RESULT_DIR = "/data/prqu/prqu_files/save_file/FuncVD-Unixcoder_5fold-cv"
DEFAULT_PRETRAINED_PATH = "/data/prqu/prqu_files/pretrained_models/Unixcoder"

METRIC_DEFS = [
    ("f1", "F1"),
    ("acc", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
]

FUNC_COL_CANDIDATES = [
    "func_before", "function_before", "func", "function",
    "code_before", "code", "source",
]
LABEL_COL_CANDIDATES = [
    "fun_label", "target", "vul", "label", "labels",
    "is_vulnerability", "vulnerability", "is_vulnerable", "vulnerable",
]

FOLD_DIR_TEMPLATES = [
    "fold{fold}", "fold_{fold}", "fold-{fold}",
    "Fold{fold}", "fold{fold:02d}", "fold_{fold:02d}",
]


class EmptySplitError(ValueError):
    """某个数据子集在过滤无效标签后为空。"""


# ============================================================================
# 数据路径发现
# ============================================================================

def _find_case_insensitive(directory, filename):
    """在目录中按大小写不敏感的方式查找文件。"""
    target = filename.lower()
    try:
        for name in os.listdir(directory):
            full_path = os.path.join(directory, name)
            if name.lower() == target and os.path.isfile(full_path):
                return full_path
    except FileNotFoundError:
        return None
    return None


def _find_split_file(fold_dir, split):
    """在 fold 目录中查找 train/val/test 文件。"""
    names = {
        "train": ["train.csv"],
        "val": ["val.csv", "valid.csv", "validation.csv", "dev.csv"],
        "test": ["test.csv"],
    }[split]
    for name in names:
        path = _find_case_insensitive(fold_dir, name)
        if path is not None:
            return path
    return None


def _flat_candidates(data_dir, split, fold_id):
    """查找平铺式文件，例如 train_fold1.csv / fold1_train.csv。"""
    split_names = {
        "train": ["train", "training"],
        "val": ["val", "valid", "validation", "dev"],
        "test": ["test", "testing"],
    }[split]

    names = []
    for split_name in split_names:
        names.extend([
            f"{split_name}_fold{fold_id}.csv",
            f"{split_name}_fold_{fold_id}.csv",
            f"fold{fold_id}_{split_name}.csv",
            f"fold_{fold_id}_{split_name}.csv",
        ])

    for name in names:
        path = _find_case_insensitive(data_dir, name)
        if path is not None:
            return path
    return None


def _generic_find(data_dir, split, fold_id):
    """兜底扫描：按路径中的 fold 编号和文件名中的 split 关键词匹配。"""
    split_aliases = {
        "train": {"train", "training"},
        "val": {"val", "valid", "validation", "dev"},
        "test": {"test", "testing"},
    }
    fold_re = re.compile(
        r"(?<![a-z0-9])fold[\s_\-]?0*%d(?![0-9])" % fold_id,
        re.IGNORECASE,
    )

    matches = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, data_dir)
            stem = os.path.splitext(filename)[0].lower()
            tokens = [t for t in re.split(r"[_\-\s.]+", stem) if t]

            found_splits = {
                name for name, aliases in split_aliases.items()
                if any(t in aliases for t in tokens)
            }
            if found_splits != {split}:
                continue

            has_fold = bool(fold_re.search(rel_path))
            if not has_fold:
                parts = [p for p in re.split(r"[\\/]+", rel_path) if p]
                has_fold = any(
                    p.isdigit() and int(p) == fold_id for p in parts[:-1]
                )
            if not has_fold:
                has_fold = any(
                    t.isdigit() and int(t) == fold_id for t in tokens
                )

            if has_fold:
                matches.append(full_path)

    matches.sort(key=lambda p: (len(p), p))
    return matches[0] if matches else None


def resolve_fold_files(data_dir, fold_id):
    """定位某一折的 train/val/test 文件。"""
    fold_dir = None
    for template in FOLD_DIR_TEMPLATES:
        candidate = os.path.join(data_dir, template.format(fold=fold_id))
        if os.path.isdir(candidate):
            fold_dir = candidate
            break

    paths = {}
    if fold_dir is not None:
        for split in ("train", "val", "test"):
            path = _find_split_file(fold_dir, split)
            if path is not None:
                paths[split] = os.path.abspath(path)

    for split in ("train", "val", "test"):
        if split in paths:
            continue
        path = _flat_candidates(data_dir, split, fold_id)
        if path is None:
            path = _generic_find(data_dir, split, fold_id)
        if path is not None:
            paths[split] = os.path.abspath(path)

    return paths, (os.path.abspath(fold_dir) if fold_dir else "")


def discover_folds(data_dir, n_folds, requested_folds=None):
    """发现每折数据文件并做完整性检查。"""
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    if requested_folds:
        fold_ids = list(requested_folds)
    else:
        fold_ids = list(range(1, n_folds + 1))

    folds = []
    problems = []

    for fold_id in fold_ids:
        paths, fold_dir = resolve_fold_files(data_dir, fold_id)

        seen = {}
        for split, path in paths.items():
            key = os.path.abspath(path)
            if key in seen:
                problems.append(
                    f"fold{fold_id}: {split} 与 {seen[key]} 指向同一文件: {path}"
                )
            seen[key] = split

        if "train" not in paths:
            problems.append(f"fold{fold_id}: 缺少 train.csv")
        if "test" not in paths:
            problems.append(f"fold{fold_id}: 缺少 test.csv")
        if "val" not in paths:
            print(f"警告: fold{fold_id} 未找到 val.csv，训练时将从 train.csv 分层切分验证集。")

        folds.append({
            "fold": fold_id,
            "dir": fold_dir,
            **paths,
        })

    if problems:
        message = ["无法为以下折定位完整数据集:"]
        message.extend(f"  - {item}" for item in problems)
        message.append("")
        message.append("期望的目录示例:")
        message.append(f"  {data_dir}/fold1/train.csv")
        message.append(f"  {data_dir}/fold1/val.csv")
        message.append(f"  {data_dir}/fold1/test.csv")
        message.append("")
        message.append("当前数据目录内容:")
        try:
            for name in sorted(os.listdir(data_dir))[:40]:
                message.append(f"  {name}")
        except OSError:
            pass
        raise FileNotFoundError("\n".join(message))

    print(f"\n数据目录: {data_dir}")
    print(f"将运行 {len(folds)} 折: " +
          ", ".join(f"fold{spec['fold']}" for spec in folds))
    for spec in folds:
        print(f"  fold{spec['fold']}:")
        print(f"    train: {spec['train']}")
        print(f"    val:   {spec.get('val', '<缺失，将从 train 切分>')}")
        print(f"    test:  {spec['test']}")

    return folds


# ============================================================================
# 数据读取与预处理
# ============================================================================

def find_column(df, candidates):
    """按候选列名精确匹配，匹配不到时再做大小写不敏感匹配。"""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _build_info(path, labels, func_column, label_column, source, invalid_rows=0):
    """构造数据子集信息，便于日志和结果记录。"""
    labels = np.asarray(labels).flatten()
    n_vul = int((labels == 1).sum())
    n_non_vul = int((labels == 0).sum())
    return {
        "path": path,
        "rows": int(len(labels)),
        "vul": n_vul,
        "non_vul": n_non_vul,
        "func_column": func_column,
        "label_column": label_column,
        "source": source,
        "invalid_rows": int(invalid_rows),
    }


def load_split(path, func_column=None, label_column=None, split_name=""):
    """读取一个数据子集，返回 (code_texts, labels, info)。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    print(f"  读取 {split_name}: {path}")
    df = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    if len(df) == 0:
        raise EmptySplitError(f"{split_name} 数据集为空: {path}")

    resolved_func = func_column or find_column(df, FUNC_COL_CANDIDATES)
    resolved_label = label_column or find_column(df, LABEL_COL_CANDIDATES)

    if resolved_func is None:
        raise KeyError(
            f"{path} 中未找到函数代码列，候选列名为 {FUNC_COL_CANDIDATES}。\n"
            f"可用列: {list(df.columns)}"
        )
    if resolved_label is None:
        raise KeyError(
            f"{path} 中未找到标签列，候选列名为 {LABEL_COL_CANDIDATES}。\n"
            f"可用列: {list(df.columns)}"
        )
    if resolved_func not in df.columns:
        raise KeyError(
            f"指定的函数代码列 '{resolved_func}' 不存在于 {path}。\n"
            f"可用列: {list(df.columns)}"
        )
    if resolved_label not in df.columns:
        raise KeyError(
            f"指定的标签列 '{resolved_label}' 不存在于 {path}。\n"
            f"可用列: {list(df.columns)}"
        )

    raw_labels = df[resolved_label]
    labels = pd.to_numeric(raw_labels, errors="coerce")

    # 兼容 True/False 类型的标签
    if labels.isin([0, 1]).sum() == 0 and raw_labels.dtype == object:
        mapped = raw_labels.astype(str).str.strip().str.lower().map({
            "true": 1, "false": 0, "yes": 1, "no": 0,
        })
        if mapped.notna().any():
            labels = mapped

    valid_mask = labels.isin([0, 1])
    n_invalid = int((~valid_mask).sum())

    if n_invalid:
        examples = raw_labels.loc[~valid_mask].head(5).tolist()
        print(f"  警告: {n_invalid} 行标签不是 0/1，已丢弃。示例: {examples}")

    df = df.loc[valid_mask].reset_index(drop=True)
    labels = labels.loc[valid_mask].astype(int).reset_index(drop=True)

    if len(df) == 0:
        raise EmptySplitError(f"{split_name} 在过滤无效标签后为空: {path}")

    texts = df[resolved_func].fillna("").astype(str).to_numpy()
    labels = labels.to_numpy(dtype=np.int64)

    n_vul = int((labels == 1).sum())
    n_non_vul = int(len(labels) - n_vul)
    print(
        f"        {len(labels)} 条 | 漏洞 {n_vul} | 非漏洞 {n_non_vul} | "
        f"函数列: {resolved_func} | 标签列: {resolved_label}"
    )
    if n_vul == 0 or n_non_vul == 0:
        print("       警告: 该子集只包含一个类别，对应指标可能没有意义。")

    info = _build_info(
        path=path,
        labels=labels,
        func_column=resolved_func,
        label_column=resolved_label,
        source="file",
        invalid_rows=n_invalid,
    )
    return texts, labels, info


def prepare_fold_data(spec, args):
    """读取某一折的 train/val/test；缺少 val 时从 train 中切分。"""
    fold_id = spec["fold"]
    print(f"\n[Fold {fold_id}] 加载数据")

    train_texts, train_labels, train_info = load_split(
        spec["train"], args.func_column, args.label_column, f"fold{fold_id}/train"
    )

    val_texts = None
    val_labels = None
    val_info = None
    if spec.get("val"):
        try:
            val_texts, val_labels, val_info = load_split(
                spec["val"], args.func_column, args.label_column, f"fold{fold_id}/val"
            )
        except EmptySplitError as exc:
            print(f"  警告: {exc}")

    if val_texts is None or len(val_texts) == 0:
        print(
            f"  警告: fold{fold_id} 没有可用 val.csv，"
            f"将从 train.csv 分层切分 {args.val_ratio:.0%} 作为验证集。"
        )
        if len(train_labels) < 2:
            raise ValueError(
                f"fold{fold_id} 的训练集只有 {len(train_labels)} 条样本，"
                "无法再切分验证集。"
            )

        n_classes = len(np.unique(train_labels))
        n_val = int(round(len(train_labels) * args.val_ratio))
        n_val = max(1, min(n_val, len(train_labels) - 1))

        stratify = None
        if n_classes > 1 and n_val >= n_classes and (len(train_labels) - n_val) >= n_classes:
            counts = np.bincount(train_labels, minlength=2)
            positive_counts = counts[counts > 0]
            if len(positive_counts) > 0 and positive_counts.min() >= 2:
                stratify = train_labels

        if stratify is None:
            print("  提示: 验证集切分未使用分层（样本量或类别数不足）。")

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=n_val,
            random_state=args.seed + fold_id,
            stratify=stratify,
            shuffle=True,
        )

        func_col = train_info["func_column"]
        label_col = train_info["label_column"]
        train_info = _build_info(
            path=spec["train"],
            labels=train_labels,
            func_column=func_col,
            label_column=label_col,
            source="file (train part after auto val split)",
        )
        val_info = _build_info(
            path=f"<auto_split_from:{spec['train']}>",
            labels=val_labels,
            func_column=func_col,
            label_column=label_col,
            source="auto_split_from_train",
        )
        print(f"  自动切分后: train {len(train_labels)} | val {len(val_labels)}")

    test_texts, test_labels, test_info = load_split(
        spec["test"], args.func_column, args.label_column, f"fold{fold_id}/test"
    )

    print(
        f"  数据规模: train {len(train_labels)} | "
        f"val {len(val_labels)} | test {len(test_labels)}"
    )

    return {
        "train_texts": train_texts,
        "train_labels": train_labels,
        "val_texts": val_texts,
        "val_labels": val_labels,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "train_info": train_info,
        "val_info": val_info,
        "test_info": test_info,
    }


# ============================================================================
# Dataset / Model
# ============================================================================

class CodeDataset(Dataset):
    """函数级数据集。

    默认在训练前一次性完成分词（pre_tokenize=True），避免每个 epoch 重复
    分词。5 折 × 每折 30 epoch（默认）的情况下，这能显著减少 CPU 开销。
    """

    def __init__(self, texts, labels, tokenizer, max_func_length=512,
                 pre_tokenize=True, desc=""):
        if len(texts) != len(labels):
            raise ValueError(
                f"texts 与 labels 长度不一致: {len(texts)} vs {len(labels)}"
            )
        if len(texts) == 0:
            raise ValueError("数据集为空，无法构造 DataLoader。")

        self.tokenizer = tokenizer
        self.max_func_length = max_func_length
        self.pre_tokenize = pre_tokenize
        self.texts = list(texts)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.input_ids = None
        self.attention_mask = None

        if self.pre_tokenize:
            self.input_ids, self.attention_mask = self._tokenize_all(desc)
            # 预分词完成后释放文本和 tokenizer，降低 DataLoader 内存/序列化开销
            self.tokenizer = None
            self.texts = None

    def _tokenize_all(self, desc):
        batch_size = 1024
        total = len(self.texts)
        all_input_ids = []
        all_attention_mask = []

        for start in range(0, total, batch_size):
            batch_texts = self.texts[start:start + batch_size]
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_func_length,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="np",
            )
            all_input_ids.append(encoding["input_ids"])
            all_attention_mask.append(encoding["attention_mask"])
            print(
                f"\r  预分词 {desc}: {min(start + batch_size, total)}/{total}",
                end="",
                flush=True,
            )
        print()

        input_ids = np.concatenate(all_input_ids, axis=0).astype(np.int64)
        attention_mask = np.concatenate(all_attention_mask, axis=0).astype(np.int64)
        return torch.from_numpy(input_ids), torch.from_numpy(attention_mask)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.pre_tokenize:
            return {
                "func_input_ids": self.input_ids[idx],
                "func_attention_mask": self.attention_mask[idx],
                "func_label": self.labels[idx],
            }

        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_func_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
            return_attention_mask=True,
        )
        return {
            "func_input_ids": encoding["input_ids"].squeeze(0),
            "func_attention_mask": encoding["attention_mask"].squeeze(0),
            "func_label": self.labels[idx],
        }


def custom_collate(batch):
    return {
        "func_input_ids": torch.stack([sample["func_input_ids"] for sample in batch]),
        "func_attention_mask": torch.stack([sample["func_attention_mask"] for sample in batch]),
        "func_label": torch.stack([sample["func_label"] for sample in batch]),
    }


class VulDetectionModel(nn.Module):
    """仅函数级：UnixCoder encoder + attention-mask mean pooling + MLP。"""

    def __init__(self, pretrained_path, num_classes=2):
        super().__init__()
        self.unixcoder = AutoModel.from_pretrained(pretrained_path)

        for param in self.unixcoder.parameters():
            param.requires_grad = False

        # 与原始 Only_FunVD_Unixcoder_FineTune.py 一致：
        # 优先解冻最后 6 个 encoder layer；若结构不同则回退为全部解冻。
        if hasattr(self.unixcoder, "encoder") and hasattr(self.unixcoder.encoder, "layer"):
            for param in self.unixcoder.encoder.layer[-6:].parameters():
                param.requires_grad = True
        else:
            for param in self.unixcoder.parameters():
                param.requires_grad = True

        self.func_classifier = nn.Sequential(
            nn.Linear(self.unixcoder.config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, func_inputs):
        encoder_outputs = self.unixcoder(
            input_ids=func_inputs["input_ids"],
            attention_mask=func_inputs["attention_mask"],
        )

        last_hidden = encoder_outputs.last_hidden_state
        attention_mask = func_inputs["attention_mask"]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())

        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
        token_count = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_features = sum_hidden / torch.clamp(token_count, min=1)

        func_logits = self.func_classifier(pooled_features)
        return {"func_logits": func_logits}


# ============================================================================
# 指标、日志、训练与评估
# ============================================================================

def calculate_func_metrics(preds, labels):
    """计算函数级二分类指标。"""
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    metrics = {
        "acc": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }

    if len(labels) == 0:
        return metrics

    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total = tp + fp + tn + fn
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn
    metrics["acc"] = (tp + tn) / total if total else 0.0
    metrics["precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    metrics["f1"] = (
        2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        if (metrics["precision"] + metrics["recall"])
        else 0.0
    )
    return metrics


def write_metrics_log(path, phase, epoch, metrics, loss):
    entry = (
        f"{phase}: Epoch {epoch:03d}:\n"
        f"\t[Func] F1: {metrics['f1']:.4f} | Acc: {metrics['acc']:.4f} | "
        f"Pre: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}\n"
        f"\t[Confusion] TP: {metrics['tp']} | FP: {metrics['fp']} | "
        f"TN: {metrics['tn']} | FN: {metrics['fn']}\n"
        f"\tLoss: {loss:.4f}\n"
    )
    with open(path, "a", encoding="utf-8") as file:
        file.write(entry)


def print_metrics(phase, loss, metrics):
    print(
        f"{phase} Loss: {loss:.4f} | Func_F1: {metrics['f1']:.4f} | "
        f"Func_Acc: {metrics['acc']:.4f} | Func_Pre: {metrics['precision']:.4f} | "
        f"Func_Rec: {metrics['recall']:.4f} | "
        f"TP/FP/TN/FN: {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}"
    )


def save_top_model(current_f1, epoch, model, model_dir, top_models, max_saved_models):
    """保存验证集 F1 更高的模型，每折最多保留 max_saved_models 个。"""
    if current_f1 is None or np.isnan(current_f1):
        return False

    if len(top_models) < max_saved_models or current_f1 > top_models[-1]["f1"]:
        model_name = f"epoch_{epoch:03d}__f1_{current_f1:.4f}.pth"
        model_path = os.path.join(model_dir, model_name)
        torch.save(model.state_dict(), model_path)

        top_models.append({
            "f1": float(current_f1),
            "epoch": epoch,
            "path": model_path,
        })
        top_models.sort(key=lambda item: -item["f1"])

        while len(top_models) > max_saved_models:
            removed = top_models.pop()
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])
        return True
    return False


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer, fold_tag):
    model.train()
    criterion = nn.CrossEntropyLoss()
    current_step = epoch * len(dataloader)

    total_loss = 0.0
    func_preds = []
    func_labels = []

    pbar = tqdm(
        dataloader,
        desc=f"{fold_tag} Epoch {epoch + 1} [Train]",
        unit="batch",
    )
    for batch_idx, batch in enumerate(pbar):
        func_inputs = {
            "input_ids": batch["func_input_ids"].to(device),
            "attention_mask": batch["func_attention_mask"].to(device),
        }
        labels = batch["func_label"].to(device)

        outputs = model(func_inputs)
        loss = criterion(outputs["func_logits"], labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        writer.add_scalar(
            f"{fold_tag}/Train/Batch_Loss",
            loss.item(),
            current_step + batch_idx,
        )
        writer.add_scalar(
            f"{fold_tag}/Learning_Rate",
            current_lr,
            current_step + batch_idx,
        )

        batch_preds = torch.argmax(outputs["func_logits"], dim=-1).cpu().numpy()
        batch_labels = labels.cpu().numpy()
        func_preds.extend(batch_preds)
        func_labels.extend(batch_labels)

    pbar.close()

    avg_loss = total_loss / max(len(dataloader), 1)
    func_metrics = calculate_func_metrics(func_preds, func_labels)

    writer.add_scalars(
        f"{fold_tag}/Epoch/Train_Func_Metrics",
        {
            "Func_F1": func_metrics["f1"],
            "Func_Acc": func_metrics["acc"],
            "Func_Precision": func_metrics["precision"],
            "Func_Recall": func_metrics["recall"],
        },
        epoch,
    )
    return avg_loss, func_metrics


def evaluate(model, dataloader, device, epoch, writer, fold_tag, mode="val"):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    func_preds = []
    func_labels = []

    with torch.inference_mode():
        pbar = tqdm(
            dataloader,
            desc=f"{fold_tag} Epoch {epoch + 1} [{mode.upper()}]",
            unit="batch",
        )
        for batch in pbar:
            func_inputs = {
                "input_ids": batch["func_input_ids"].to(device),
                "attention_mask": batch["func_attention_mask"].to(device),
            }
            labels = batch["func_label"].to(device)

            outputs = model(func_inputs)
            loss = criterion(outputs["func_logits"], labels)

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            batch_preds = torch.argmax(outputs["func_logits"], dim=-1).cpu().numpy()
            batch_labels = labels.cpu().numpy()
            func_preds.extend(batch_preds)
            func_labels.extend(batch_labels)
        pbar.close()

    avg_loss = total_loss / max(len(dataloader), 1)
    func_metrics = calculate_func_metrics(func_preds, func_labels)

    writer.add_scalars(
        f"{fold_tag}/Epoch/{mode}_Func_Metrics",
        {
            "Func_F1": func_metrics["f1"],
            "Func_Acc": func_metrics["acc"],
            "Func_Precision": func_metrics["precision"],
            "Func_Recall": func_metrics["recall"],
        },
        epoch,
    )
    return avg_loss, func_metrics


# ============================================================================
# 汇总与结果保存
# ============================================================================

def records_to_per_fold_df(records):
    rows = []
    for record in sorted(records, key=lambda item: item["fold_order"]):
        rows.append({
            "fold": record["fold"],
            "fold_order": record["fold_order"],
            "best_epoch": record["best_epoch"],
            "best_epoch_1based": record["best_epoch_1based"],
            "train_size": record["train_size"],
            "val_size": record["val_size"],
            "test_size": record["test_size"],
            "test_f1": record["test"]["f1"],
            "test_accuracy": record["test"]["acc"],
            "test_precision": record["test"]["precision"],
            "test_recall": record["test"]["recall"],
            "test_tp": record["test"]["tp"],
            "test_fp": record["test"]["fp"],
            "test_tn": record["test"]["tn"],
            "test_fn": record["test"]["fn"],
            "val_f1": record["val"]["f1"],
            "val_accuracy": record["val"]["acc"],
            "val_precision": record["val"]["precision"],
            "val_recall": record["val"]["recall"],
            "train_seconds": record["train_seconds"],
            "tokenize_seconds": record["tokenize_seconds"],
            "total_seconds": record["total_seconds"],
            "peak_gpu_mem_mb": record["peak_gpu_mem_mb"],
            "val_source": record["val_source"],
            "model_path": record["model_path"],
        })
    return pd.DataFrame(rows)


def records_to_long_df(records):
    rows = []
    for record in sorted(records, key=lambda item: item["fold_order"]):
        for split in ("test", "val"):
            metrics = record[split]
            for key, label in METRIC_DEFS:
                rows.append({
                    "fold": record["fold"],
                    "fold_order": record["fold_order"],
                    "best_epoch": record["best_epoch"],
                    "split": split,
                    "metric": label,
                    "value": float(metrics[key]),
                })
    return pd.DataFrame(rows)


def compute_summary_df(records):
    """计算每项指标的 mean、std 和 95% 置信区间。"""
    rows = []
    for split in ("test", "val"):
        for key, label in METRIC_DEFS:
            values = np.array(
                [record[split][key] for record in records],
                dtype=float,
            )
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

            half = 0.0
            if len(values) > 1 and std > 0:
                try:
                    from scipy.stats import t as t_dist
                    half = (
                        float(t_dist.ppf(0.975, len(values) - 1))
                        * std / np.sqrt(len(values))
                    )
                except ImportError:
                    # scipy 不可用时回退到正态近似；n=5 时建议安装 scipy
                    half = 1.96 * std / np.sqrt(len(values))

            rows.append({
                "split": split,
                "metric": label,
                "n_folds": len(values),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "mean_std": f"{mean:.4f} ± {std:.4f}",
                "ci95_low": round(mean - half, 4),
                "ci95_high": round(mean + half, 4),
                "ci95": f"[{mean - half:.4f}, {mean + half:.4f}]",
                "min": round(float(np.min(values)), 4),
                "max": round(float(np.max(values)), 4),
                "min_max": f"[{np.min(values):.4f}, {np.max(values):.4f}]",
            })
    return pd.DataFrame(rows)


def save_results(records, result_dir, expected_n_folds=None, verbose=True):
    """保存每折明细、汇总 CSV 和可读文本。"""
    os.makedirs(result_dir, exist_ok=True)

    per_fold_df = records_to_per_fold_df(records)
    long_df = records_to_long_df(records)
    summary_df = compute_summary_df(records)

    per_fold_path = os.path.join(result_dir, "cv_per_fold.csv")
    long_path = os.path.join(result_dir, "cv_per_fold_long.csv")
    summary_path = os.path.join(result_dir, "cv_summary.csv")
    txt_path = os.path.join(result_dir, "cv_summary.txt")

    per_fold_df.to_csv(per_fold_path, index=False, encoding="utf-8-sig")
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 便于直接查看的每折四指标表
    if not per_fold_df.empty:
        simple_per_fold = per_fold_df[[
            "fold", "best_epoch_1based",
            "test_f1", "test_accuracy", "test_precision", "test_recall",
        ]].copy()
        simple_per_fold.columns = [
            "fold", "best_epoch_1based", "F1", "Accuracy", "Precision", "Recall",
        ]
        simple_per_fold.to_csv(
            os.path.join(result_dir, "per_fold_test_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # 最终测试集汇总（只保留 test，便于写论文/表格）
    if not summary_df.empty:
        test_summary = summary_df[summary_df["split"] == "test"].copy()
        if not test_summary.empty:
            test_summary = test_summary.drop(columns=["split"])
            test_summary.to_csv(
                os.path.join(result_dir, "final_test_summary.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("=" * 90 + "\n")
        file.write("UnixCoder 函数级漏洞检测交叉验证结果\n")
        file.write("=" * 90 + "\n")
        file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if expected_n_folds is not None:
            file.write(f"已完成折数: {len(records)} / {expected_n_folds}\n")
            if len(records) < expected_n_folds:
                file.write("注意: 这是中间结果，全部折完成后会自动覆盖本文件。\n")
        file.write("详细配置见 run_config.json；每折模型与训练日志见 fold*/ 子目录。\n\n")

        file.write("【每折测试集结果（主结果）】\n")
        file.write("-" * 90 + "\n")
        if per_fold_df.empty:
            file.write("（暂无）\n")
        else:
            display = per_fold_df[[
                "fold", "best_epoch_1based",
                "test_f1", "test_accuracy", "test_precision", "test_recall",
                "test_tp", "test_fp", "test_tn", "test_fn",
                "train_size", "val_size", "test_size",
            ]].copy()
            display.columns = [
                "Fold", "BestEpoch", "F1", "Accuracy", "Precision", "Recall",
                "TP", "FP", "TN", "FN", "Train", "Val", "Test",
            ]
            file.write(display.to_string(index=False) + "\n")

        file.write("\n【测试集指标汇总：mean、std、95% CI、Min-Max】\n")
        file.write("-" * 90 + "\n")
        if summary_df.empty:
            file.write("（暂无）\n")
        else:
            test_summary = summary_df[summary_df["split"] == "test"]
            if test_summary.empty:
                file.write("（暂无）\n")
            else:
                display = test_summary[[
                    "metric", "mean", "std", "ci95", "min_max", "n_folds",
                ]].copy()
                display.columns = [
                    "Metric", "Mean", "Std", "95% CI (t)", "Min-Max", "N",
                ]
                file.write(display.to_string(index=False) + "\n")

            val_summary = summary_df[summary_df["split"] == "val"]
            file.write("\n【验证集指标汇总（用于选择最佳 epoch，仅供参考）】\n")
            file.write("-" * 90 + "\n")
            if val_summary.empty:
                file.write("（暂无）\n")
            else:
                display = val_summary[[
                    "metric", "mean", "std", "ci95", "min_max", "n_folds",
                ]].copy()
                display.columns = [
                    "Metric", "Mean", "Std", "95% CI (t)", "Min-Max", "N",
                ]
                file.write(display.to_string(index=False) + "\n")

        if (
            not per_fold_df.empty
            and "val_source" in per_fold_df.columns
            and (per_fold_df["val_source"] != "file").any()
        ):
            file.write(
                "\n注意: 部分折缺少 val.csv，已从该折 train.csv 分层切分验证集。\n"
            )

        file.write("\n说明：\n")
        file.write("  1. 每折使用各自的 train.csv 训练、val.csv 选择最佳 epoch、test.csv 评估。\n")
        file.write("  2. 最佳 epoch 的判据为该折验证集 F1。\n")
        file.write("  3. std 为样本标准差（ddof=1）。\n")
        file.write("  4. 95% CI 使用 t 分布: mean ± t(0.975, n-1) * std / sqrt(n)。\n")
        file.write("  5. Min-Max 为各折指标最低值到最高值的区间。\n")

    if verbose:
        print(f"每折明细已保存到: {per_fold_path}")
        print(f"每折四指标已保存到: "
              f"{os.path.join(result_dir, 'per_fold_test_metrics.csv')}")
        print(f"每折长表已保存到: {long_path}")
        print(f"汇总结果已保存到: {summary_path}")
        print(f"最终测试集汇总已保存到: "
              f"{os.path.join(result_dir, 'final_test_summary.csv')}")
        print(f"可读汇总已保存到: {txt_path}")

    return per_fold_df, long_df, summary_df


def save_fold_metrics(fold_dir, record):
    """把单折的验证/测试指标单独保存到该折目录，便于逐折查看。"""
    rows = []
    for split in ("test", "val"):
        metrics = record[split]
        rows.append({
            "fold": record["fold"],
            "best_epoch": record["best_epoch"],
            "best_epoch_1based": record["best_epoch_1based"],
            "split": split,
            "f1": metrics["f1"],
            "accuracy": metrics["acc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
        })

    path = os.path.join(fold_dir, "metrics.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"该折指标已保存到: {path}")


def save_fold_discovery(folds, result_dir):
    rows = []
    for spec in folds:
        rows.append({
            "fold": spec["fold"],
            "fold_dir": spec.get("dir", ""),
            "train": spec.get("train", ""),
            "val": spec.get("val", ""),
            "test": spec.get("test", ""),
        })
    path = os.path.join(result_dir, "fold_discovery.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"折数据路径已保存到: {path}")


def save_run_config(args, folds, device, result_dir, run_timestamp, tensorboard_dir):
    config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_timestamp,
        "data_dir": os.path.abspath(args.data_dir),
        "result_dir": os.path.abspath(result_dir),
        "pretrained_path": os.path.abspath(args.pretrained_path),
        "tensorboard_dir": os.path.abspath(tensorboard_dir),
        "device": str(device),
        "seed": args.seed,
        "n_folds_requested": args.n_folds,
        "folds_requested": args.folds,
        "folds_discovered": folds,
        "hyperparameters": {
            "max_func_length": args.max_func_length,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio,
            "num_classes": args.num_classes,
            "max_saved_models": args.max_saved_models,
            "val_ratio_if_missing": args.val_ratio,
            "pre_tokenize": not args.no_pretokenize,
            "num_workers": args.num_workers,
        },
        "cli_args": vars(args),
    }
    path = os.path.join(result_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    print(f"运行配置已保存到: {path}")


def add_cost_row(cost_rows, fold_id, fold_order, phase, epoch, seconds,
                 samples_per_sec="", peak_gpu_mem_mb="", train_size=""):
    cost_rows.append({
        "fold": fold_id,
        "fold_order": fold_order,
        "phase": phase,
        "epoch": epoch,
        "time_sec": round(seconds, 2),
        "time_min": round(seconds / 60, 3),
        "samples_per_sec": samples_per_sec,
        "peak_gpu_mem_mb": peak_gpu_mem_mb,
        "train_size": train_size,
    })


def save_cost(cost_rows, result_dir, verbose=True):
    if not cost_rows:
        return
    path = os.path.join(result_dir, "training_cost.csv")
    pd.DataFrame(cost_rows).to_csv(path, index=False, encoding="utf-8-sig")
    if verbose:
        print(f"训练耗时已保存到: {path}")


def peak_gpu_mb(device):
    if device.type == "cuda" and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
    return ""


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 命令行参数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="UnixCoder 函数级漏洞检测 5 折交叉验证（直接使用已切分的每折数据）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="每折数据集根目录，默认包含 fold1/train.csv 等")
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR,
                        help="结果输出目录，不存在会自动创建")
    parser.add_argument("--pretrained_path", default=DEFAULT_PRETRAINED_PATH,
                        help="UnixCoder 预训练模型目录")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="折数，fold 编号为 1..n_folds")
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="只运行指定 fold 编号，例如 --folds 1 2；默认运行全部")
    parser.add_argument("--max_func_length", type=int, default=512,
                        help="函数代码最大 token 长度")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="学习率")
    parser.add_argument("--epochs", type=int, default=30,
                        help="每折训练 epoch 数")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="cosine 调度 warmup 比例")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="类别数；当前指标按二分类实现，必须为 2")
    parser.add_argument("--max_saved_models", type=int, default=1,
                        help="每折保留的最佳模型数量")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="缺少 val.csv 时，从 train.csv 切分验证集的比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader 工作进程数；预分词后 0 即可")
    parser.add_argument("--device", default=None,
                        help="例如 cuda:0 / cpu；默认自动选择")
    parser.add_argument("--label_column", default=None,
                        help="标签列名；默认自动识别 fun_label/target/vul 等")
    parser.add_argument("--func_column", default=None,
                        help="函数代码列名；默认自动识别 func_before 等")
    parser.add_argument("--no_pretokenize", action="store_true",
                        help="关闭预分词（更省内存但每个 epoch 都要重新分词，较慢）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只检查数据路径并打印每折数据规模，不训练")
    return parser.parse_args()


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()

    if args.n_folds <= 0:
        raise ValueError("--n_folds 必须大于 0")
    if args.folds:
        if len(set(args.folds)) != len(args.folds):
            raise ValueError("--folds 中存在重复的 fold 编号")
    if args.max_func_length <= 1:
        raise ValueError("--max_func_length 必须大于 1")
    if args.batch_size <= 0:
        raise ValueError("--batch_size 必须大于 0")
    if args.lr <= 0:
        raise ValueError("--lr 必须大于 0")
    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay 不能为负数")
    if not (0 <= args.warmup_ratio < 1):
        raise ValueError("--warmup_ratio 必须在 [0, 1) 内")
    if args.num_classes != 2:
        raise ValueError("当前指标按二分类实现，--num_classes 必须为 2")
    if args.max_saved_models <= 0:
        raise ValueError("--max_saved_models 必须大于 0")
    if not (0 < args.val_ratio < 1):
        raise ValueError("--val_ratio 必须在 (0, 1) 内")
    if args.num_workers < 0:
        raise ValueError("--num_workers 不能为负数")

    set_seed(args.seed)

    folds = discover_folds(
        args.data_dir,
        n_folds=args.n_folds,
        requested_folds=args.folds,
    )

    # 仅检查数据布局，不加载模型
    if args.dry_run:
        print("\n" + "=" * 90)
        print("DRY RUN: 检查每折数据")
        print("=" * 90)
        for order, spec in enumerate(folds, 1):
            print("\n" + "#" * 90)
            print(f"Fold {spec['fold']} ({order}/{len(folds)})")
            print("#" * 90)
            data = prepare_fold_data(spec, args)
            print(
                f"  列信息: func_column={data['train_info']['func_column']} | "
                f"label_column={data['train_info']['label_column']}"
            )
        print("\n[dry_run] 数据布局检查完成，未开始训练。")
        return

    result_dir = os.path.abspath(args.result_dir)
    if os.path.isdir(result_dir) and os.listdir(result_dir):
        print(
            f"\n提示: 结果目录已存在，将覆盖同名汇总文件；"
            f"旧的模型文件不会自动删除: {result_dir}"
        )
    os.makedirs(result_dir, exist_ok=True)

    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA 设备，但 torch.cuda.is_available() 为 False")
        device_index = device.index if device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA 设备 cuda:{device_index} 不存在，"
                f"当前可用 GPU 数量: {torch.cuda.device_count()}"
            )
    print(f"\n使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    if not os.path.isdir(args.pretrained_path):
        raise FileNotFoundError(
            f"UnixCoder 预训练模型目录不存在: {args.pretrained_path}"
        )

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_dir = os.path.join(result_dir, "tensorboard", run_timestamp)
    writer = SummaryWriter(tensorboard_dir)

    combined_log = os.path.join(result_dir, "training_log_all_folds.txt")
    with open(combined_log, "a", encoding="utf-8") as file:
        file.write("\n" + "=" * 90 + "\n")
        file.write(f"UnixCoder 5-fold CV run {run_timestamp}\n")
        file.write(f"data_dir: {os.path.abspath(args.data_dir)}\n")
        file.write(f"result_dir: {result_dir}\n")
        file.write(
            f"epochs: {args.epochs} | batch_size: {args.batch_size} | "
            f"lr: {args.lr} | device: {device}\n"
        )
        file.write("=" * 90 + "\n")

    save_run_config(
        args=args,
        folds=folds,
        device=device,
        result_dir=result_dir,
        run_timestamp=run_timestamp,
        tensorboard_dir=tensorboard_dir,
    )
    save_fold_discovery(folds, result_dir)

    print("加载 UnixCoder tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    records = []
    cost_rows = []

    try:
        for order, spec in enumerate(folds, 1):
            fold_id = spec["fold"]
            fold_start = time.time()

            print("\n" + "#" * 90)
            print(f"########## Fold {fold_id} ({order}/{len(folds)}) ##########")
            print("#" * 90)

            set_seed(args.seed + fold_id)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            fold_dir = os.path.join(result_dir, f"fold{fold_id}")
            model_dir = os.path.join(fold_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            fold_log = os.path.join(fold_dir, "training_log.txt")

            data = prepare_fold_data(spec, args)
            train_info = data["train_info"]
            val_info = data["val_info"]
            test_info = data["test_info"]

            with open(fold_log, "a", encoding="utf-8") as file:
                file.write("\n" + "=" * 80 + "\n")
                file.write(f"Fold {fold_id} ({order}/{len(folds)})\n")
                file.write(
                    f"train: {train_info['path']} | rows={train_info['rows']} | "
                    f"vul={train_info['vul']} | non_vul={train_info['non_vul']}\n"
                )
                file.write(
                    f"val:   {val_info['path']} | rows={val_info['rows']} | "
                    f"vul={val_info['vul']} | non_vul={val_info['non_vul']} | "
                    f"source={val_info['source']}\n"
                )
                file.write(
                    f"test:  {test_info['path']} | rows={test_info['rows']} | "
                    f"vul={test_info['vul']} | non_vul={test_info['non_vul']}\n"
                )
                file.write(
                    f"func_column: {train_info['func_column']} | "
                    f"label_column: {train_info['label_column']}\n"
                )
                file.write("=" * 80 + "\n")

            tokenize_start = time.time()
            train_dataset = CodeDataset(
                data["train_texts"],
                data["train_labels"],
                tokenizer,
                max_func_length=args.max_func_length,
                pre_tokenize=not args.no_pretokenize,
                desc=f"fold{fold_id}/train",
            )
            val_dataset = CodeDataset(
                data["val_texts"],
                data["val_labels"],
                tokenizer,
                max_func_length=args.max_func_length,
                pre_tokenize=not args.no_pretokenize,
                desc=f"fold{fold_id}/val",
            )
            test_dataset = CodeDataset(
                data["test_texts"],
                data["test_labels"],
                tokenizer,
                max_func_length=args.max_func_length,
                pre_tokenize=not args.no_pretokenize,
                desc=f"fold{fold_id}/test",
            )
            tokenize_seconds = time.time() - tokenize_start

            # CodeDataset 已经复制了需要的数据，释放原始文本数组
            del data

            generator = torch.Generator()
            generator.manual_seed(args.seed + fold_id)

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=custom_collate,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                generator=generator,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=custom_collate,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
            )

            print(
                f"数据集大小 | train {len(train_dataset)} | "
                f"val {len(val_dataset)} | test {len(test_dataset)} | "
                f"预分词耗时 {tokenize_seconds:.1f}s"
            )

            add_cost_row(
                cost_rows, fold_id, order, "tokenize", 0, tokenize_seconds,
                samples_per_sec=round(len(train_dataset) / tokenize_seconds, 2)
                if tokenize_seconds > 0 else "",
                train_size=len(train_dataset),
            )

            print("初始化模型...")
            model = VulDetectionModel(
                args.pretrained_path,
                num_classes=args.num_classes,
            ).to(device)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            num_trainable = sum(p.numel() for p in trainable_params)
            num_total = sum(p.numel() for p in model.parameters())
            print(
                f"可训练参数: {num_trainable:,} / {num_total:,} "
                f"({num_trainable / num_total * 100:.2f}%)"
            )
            if not trainable_params:
                raise RuntimeError("模型没有可训练参数，请检查冻结/解冻逻辑。")

            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            num_training_steps = args.epochs * len(train_loader)
            num_warmup_steps = int(args.warmup_ratio * num_training_steps)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            fold_tag = f"Fold{fold_id}"
            top_models = []
            best_val_f1 = -1.0
            best_val_epoch = -1
            train_start = time.time()

            for epoch in range(args.epochs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_start = time.time()

                print("\n========== 开始训练阶段 ==========\n")
                train_loss, train_metrics = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                    writer,
                    fold_tag,
                )
                write_metrics_log(
                    combined_log, f"Fold{fold_id}_Train",
                    epoch, train_metrics, train_loss,
                )
                write_metrics_log(
                    fold_log, "Train", epoch, train_metrics, train_loss,
                )

                print("\n========== 开始验证阶段 ==========\n")
                val_loss, val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    epoch,
                    writer,
                    fold_tag,
                    mode="val",
                )
                write_metrics_log(
                    combined_log, f"Fold{fold_id}_Val",
                    epoch, val_metrics, val_loss,
                )
                write_metrics_log(
                    fold_log, "Val", epoch, val_metrics, val_loss,
                )

                writer.add_scalars(
                    f"{fold_tag}/Loss_Train_vs_Val",
                    {"Train": train_loss, "Validation": val_loss},
                    epoch,
                )

                print(f"\nFold {fold_id} | Epoch {epoch + 1}/{args.epochs}")
                print_metrics("Train", train_loss, train_metrics)
                print_metrics("Val", val_loss, val_metrics)

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_val_epoch = epoch

                saved = save_top_model(
                    val_metrics["f1"],
                    epoch,
                    model,
                    model_dir,
                    top_models,
                    args.max_saved_models,
                )
                if saved:
                    print(f"已保存新的最佳模型: {top_models[0]['path']}")

                print("\n当前 Top 模型列表:")
                for rank, item in enumerate(top_models, 1):
                    print(
                        f"  第{rank}名 | Epoch {item['epoch'] + 1} | "
                        f"F1: {item['f1']:.4f} | {item['path']}"
                    )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_seconds = time.time() - epoch_start
                current_peak = peak_gpu_mb(device)
                mem_display = (
                    f"{current_peak} MB" if current_peak != "" else "N/A (CPU)"
                )
                add_cost_row(
                    cost_rows, fold_id, order, "epoch", epoch + 1,
                    epoch_seconds,
                    samples_per_sec=round(len(train_dataset) / epoch_seconds, 2)
                    if epoch_seconds > 0 else "",
                    peak_gpu_mem_mb=current_peak,
                    train_size=len(train_dataset),
                )
                writer.add_scalar(
                    f"{fold_tag}/Cost/Epoch_Time_Sec",
                    epoch_seconds,
                    epoch,
                )
                print(
                    f"[Cost] Fold {fold_id} Epoch {epoch + 1} 耗时 "
                    f"{epoch_seconds:.1f}s ({epoch_seconds / 60:.2f} min) | "
                    f"峰值显存 {mem_display}"
                )

            train_seconds = time.time() - train_start

            # 选择该折验证集 F1 最高的 epoch 对应的模型
            if top_models:
                best_model_path = top_models[0]["path"]
                best_epoch = top_models[0]["epoch"]
                print(f"\n加载该折最佳模型: {best_model_path}")
                model.load_state_dict(
                    torch.load(best_model_path, map_location="cpu", weights_only=True)
                )
                model.to(device)
            else:
                best_model_path = ""
                best_epoch = (
                    best_val_epoch if best_val_epoch >= 0 else args.epochs - 1
                )
                print("\n警告: 没有保存最佳模型，使用当前模型进行测试。")

            # 重新在验证集上评估，确保记录的验证指标与加载的模型一致
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                device,
                best_epoch,
                writer,
                fold_tag,
                mode="val",
            )
            print_metrics("Best-Val", val_loss, val_metrics)

            print(
                f"\n[Fold {fold_id}] 在该折测试集上评估 "
                f"(best epoch = {best_epoch + 1})..."
            )
            test_loss, test_metrics = evaluate(
                model,
                test_loader,
                device,
                best_epoch,
                writer,
                fold_tag,
                mode="test",
            )
            write_metrics_log(
                combined_log, f"Fold{fold_id}_Test",
                best_epoch, test_metrics, test_loss,
            )
            write_metrics_log(
                fold_log, "Test", best_epoch, test_metrics, test_loss,
            )
            print_metrics("Test", test_loss, test_metrics)

            total_seconds = time.time() - fold_start
            current_peak = peak_gpu_mb(device)

            add_cost_row(
                cost_rows, fold_id, order, "fold_total", args.epochs,
                train_seconds,
                samples_per_sec=round(
                    len(train_dataset) * args.epochs / train_seconds, 2
                ) if train_seconds > 0 else "",
                peak_gpu_mem_mb=current_peak,
                train_size=len(train_dataset),
            )

            record = {
                "fold": fold_id,
                "fold_order": order,
                "best_epoch": best_epoch,
                "best_epoch_1based": best_epoch + 1,
                "model_path": best_model_path,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "test_size": len(test_dataset),
                "val": dict(val_metrics),
                "test": dict(test_metrics),
                "train_seconds": round(train_seconds, 2),
                "tokenize_seconds": round(tokenize_seconds, 2),
                "total_seconds": round(total_seconds, 2),
                "peak_gpu_mem_mb": current_peak,
                "val_source": val_info["source"],
                "test_source": test_info["source"],
            }
            records.append(record)
            save_fold_metrics(fold_dir, record)

            # 每完成一折就写一次结果，避免意外中断丢失已完成折的结果
            save_results(
                records,
                result_dir,
                expected_n_folds=len(folds),
                verbose=False,
            )
            save_cost(cost_rows, result_dir, verbose=False)
            writer.flush()

            print(
                f"\nFold {fold_id} 完成 | Test F1: {test_metrics['f1']:.4f} | "
                f"Acc: {test_metrics['acc']:.4f} | "
                f"Pre: {test_metrics['precision']:.4f} | "
                f"Rec: {test_metrics['recall']:.4f}"
            )

            del model, optimizer, scheduler
            del train_loader, val_loader, test_loader
            del train_dataset, val_dataset, test_dataset
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        try:
            save_cost(cost_rows, result_dir, verbose=False)
            if records:
                save_results(
                    records,
                    result_dir,
                    expected_n_folds=len(folds),
                    verbose=False,
                )
        finally:
            writer.close()

    per_fold_df, _, summary_df = save_results(
        records,
        result_dir,
        expected_n_folds=len(folds),
        verbose=True,
    )
    save_cost(cost_rows, result_dir, verbose=True)

    print("\n" + "=" * 90)
    print(f"全部 {len(records)} 折完成")
    print("=" * 90)

    if not per_fold_df.empty:
        display = per_fold_df[[
            "fold", "best_epoch_1based",
            "test_f1", "test_accuracy", "test_precision", "test_recall",
        ]].copy()
        display.columns = [
            "Fold", "BestEpoch", "F1", "Accuracy", "Precision", "Recall",
        ]
        print("\n【每折测试集结果】")
        print(display.to_string(index=False))

    if not summary_df.empty:
        test_summary = summary_df[summary_df["split"] == "test"]
        if not test_summary.empty:
            display = test_summary[[
                "metric", "mean", "std", "ci95", "min_max", "n_folds",
            ]].copy()
            display.columns = [
                "Metric", "Mean", "Std", "95% CI (t)", "Min-Max", "N",
            ]
            print("\n【测试集指标汇总】")
            print(display.to_string(index=False))

    print(f"\n全部结果已保存到: {result_dir}")
    print(f"TensorBoard 日志: {tensorboard_dir}")


if __name__ == "__main__":
    main()
