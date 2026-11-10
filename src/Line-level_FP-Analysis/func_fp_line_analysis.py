# -*- coding: utf-8 -*-
"""
=============================================================================
函数级误报（False Positive）样本的行级误报统计分析
=============================================================================
用途：
    对 Fun_Line_VD_Trans+PosEncoding.py 训练好的模型，**只重新执行测试环节**
    （不训练、不验证），测试集与主实验完全一致：
        /data/prqu/prqu_files/data/Most_Information_Dataset/test_balanced.csv

    重点回答一个审稿人常见问题：
        "函数级被判为有漏洞（但实际无漏洞）的误报样本，
         在行级定位阶段又被误报了多少行？"

    即：筛选出函数级 FP（真值 0 / 预测 1）的样本，逐个统计其有效代码行中
        被行级分类器预测为 1 的行数，并输出明细到 Excel。

输出（Excel，多 sheet）：
    Summary            —— 整体统计（函数级混淆矩阵、FP 样本数、行级误报分布等）
    FP_Samples         —— 每个函数级 FP 样本一行：该样本行级误报了多少行
    Line_FP_Detail     —— 每个被行级误报的具体代码行（含代码文本与概率）
    Distribution       —— 行级误报行数的区间分布
    Column_Notes       —— 各列含义说明

依赖：
    pandas / numpy / torch / transformers / tqdm / openpyxl
    （若缺少 openpyxl，会自动降级为写出多个 CSV 文件并给出提示）

用法：
    # 1) 最简：指定一个 checkpoint（推荐，避免选错模型）
    python func_fp_line_analysis.py \
        --checkpoint /data/prqu/prqu_files/save_file/models_Trans+PosEncoding_Cross-project-dataset/epoch_042__f1_0.4623.pth

    # 2) 不指定 checkpoint 时，自动扫描候选目录并挑选文件名中 F1 最高的 .pth
    python func_fp_line_analysis.py

    # 3) 指定测试集 / 输出位置 / 设备
    python func_fp_line_analysis.py --checkpoint xxx.pth \
        --test_csv /path/test_balanced.csv \
        --out_xlsx /path/result.xlsx --device cuda:1
=============================================================================
"""

import os
import re
import json
import glob
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset


"""路径参数及超参设置（与 Fun_Line_VD_Trans+PosEncoding.py 保持一致）"""
CONFIG = {
    "line_bert_path": "/data/prqu/prqu_files/pretrained_models/CodeT5_base",
    "func_bert_path": "/data/prqu/prqu_files/pretrained_models/CodeT5_base",
    "max_func_length": 512,
    "max_line_length": 32,
    "max_lines": 48,
    "num_classes": 2,
    "batch_size": 16,

    # 测试集：与主实验完全相同
    "test_csv": "/data/prqu/prqu_files/data/Most_Information_Dataset/test_balanced.csv",

    # 未显式指定 --checkpoint 时的候选模型目录（按顺序扫描，取文件名 F1 最高者）
    "candidate_model_dirs": [
        "/data/prqu/prqu_files/save_file/models_Trans+PosEncoding_Cross-project-dataset",
        "/data/prqu/prqu_files/save_file/models_metrics_modified_Trans+PosEncoding_NewDateset",
        "/data/prqu/prqu_files/save_file/cv_5fold_Trans+Pos_balanced",
    ],

    # 结果输出目录
    "out_dir": "/data/prqu/prqu_files/save_file/FP_Analysis",
}


"""数据集处理（与主实验完全一致，保证输入编码相同）"""
class CodeDataset(Dataset):
    def __init__(self, df, max_lines=48, max_func_length=512, max_line_length=32):
        self.df = df
        self.max_lines = max_lines
        self.max_func_length = max_func_length
        self.max_line_length = max_line_length
        self.tokenizer = RobertaTokenizer.from_pretrained(CONFIG['line_bert_path'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        func_before = self.df.iloc[idx]['func_before']
        func_label = self.df.iloc[idx]['fun_label']
        lines = json.loads(self.df.iloc[idx]['lines'])[:self.max_lines]
        line_labels = json.loads(self.df.iloc[idx]['line_labels'])[:self.max_lines]

        # 函数级编码
        func_encoding = self.tokenizer(
            func_before,
            max_length=self.max_func_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,
            return_attention_mask=True
        )

        # 行级编码
        line_embeddings = []
        valid_masks = []
        for line in lines:
            if not line or line.strip() == "":
                line = " "

            encoding = self.tokenizer(
                line,
                max_length=self.max_line_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False,
                return_attention_mask=True
            )

            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)

            if input_ids.size(0) != self.max_line_length:
                if input_ids.size(0) < self.max_line_length:
                    pad_length = self.max_line_length - input_ids.size(0)
                    input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
                else:
                    input_ids = input_ids[:self.max_line_length]
                    attention_mask = attention_mask[:self.max_line_length]

            line_embeddings.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })
            valid_masks.append(1)

        # 填充空行
        padding_length = self.max_lines - len(lines)
        for _ in range(padding_length):
            line_embeddings.append({
                'input_ids': torch.zeros(self.max_line_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_line_length, dtype=torch.long)
            })
            valid_masks.append(0)
            line_labels.append(-100)

        return {
            'func_input': {
                'func_input_ids': func_encoding['input_ids'].squeeze(0),
                'func_attention_mask': func_encoding['attention_mask'].squeeze(0)
            },
            'func_label': torch.tensor(func_label, dtype=torch.long),
            'line_inputs': line_embeddings,
            'line_labels': torch.LongTensor(line_labels),
            'valid_mask': torch.BoolTensor(valid_masks)
        }


"""模型类（与主实验完全一致，否则无法加载 checkpoint）"""
class VulDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(CONFIG['line_bert_path'])

        # CodeT5-1: 行级特征提取器
        self.codeT51 = T5ForConditionalGeneration.from_pretrained(CONFIG['line_bert_path'])
        for param in self.codeT51.parameters():
            param.requires_grad = False

        # CodeT5-2: 函数级特征提取器
        self.codeT52 = T5ForConditionalGeneration.from_pretrained(CONFIG['func_bert_path'])
        for param in self.codeT52.encoder.block[-6:].parameters():
            param.requires_grad = True

        # 维度适配器
        self.dim_adapter = nn.Linear(
            self.codeT51.config.d_model,
            self.codeT52.config.d_model
        )

        # 可学习的相对位置编码层
        self.position_embedding = nn.Embedding(
            CONFIG['max_lines'],
            self.codeT52.config.d_model
        )
        self.position_dropout = nn.Dropout(0.1)

        # 行间结构建模Transformer层
        self.line_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.codeT52.config.d_model,
                nhead=8,
                dim_feedforward=self.codeT52.config.d_model * 2,
                dropout=0.1,
                activation='relu',
                batch_first=True,
                norm_first=False
            ),
            num_layers=3,
            enable_nested_tensor=False
        )

        # 行级语义特征融合层（行级特征 + 结构特征）
        self.line_semantic_fusion = nn.Sequential(
            nn.Linear(self.codeT52.config.d_model * 2, self.codeT52.config.d_model),
            nn.LayerNorm(self.codeT52.config.d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 函数级分类头
        self.func_classifier = nn.Sequential(
            nn.Linear(self.codeT52.config.d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, CONFIG['num_classes'])
        )

        # 行级分类头
        self.line_classifier = nn.Sequential(
            nn.Linear(self.codeT52.config.d_model * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, CONFIG['num_classes'])
        )

    def forward(self, func_inputs, line_inputs, valid_mask):
        batch_size = func_inputs['input_ids'].size(0)
        max_lines = len(line_inputs)
        device = func_inputs['input_ids'].device

        # Stage 1: 函数级全局特征
        global_output = self.codeT52.encoder(
            input_ids=func_inputs['input_ids'],
            attention_mask=func_inputs['attention_mask']
        )
        last_hidden = global_output.last_hidden_state
        attention_mask = func_inputs['attention_mask']

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
        token_count = torch.sum(attention_mask, dim=1, keepdim=True)
        global_features = sum_hidden / torch.clamp(token_count, min=1)

        func_logits = self.func_classifier(global_features)

        # Stage 2: 逐行特征提取 + 位置编码
        line_features_list = []
        for line_idx in range(max_lines):
            line_data = line_inputs[line_idx]
            outputs = self.codeT51.encoder(
                input_ids=line_data['input_ids'],
                attention_mask=line_data['attention_mask']
            )
            line_hidden = outputs.last_hidden_state
            line_mask = line_data['attention_mask']

            line_mask_expanded = line_mask.unsqueeze(-1).expand(line_hidden.size())
            line_sum_hidden = torch.sum(line_hidden * line_mask_expanded, dim=1)
            line_token_count = torch.sum(line_mask, dim=1, keepdim=True)
            line_avg = line_sum_hidden / torch.clamp(line_token_count, min=1)

            adapted_line_features = self.dim_adapter(line_avg)
            line_features_list.append(adapted_line_features)

        all_line_features = torch.stack(line_features_list, dim=1)

        position_ids = torch.arange(max_lines, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embedding(position_ids)
        position_embeddings = self.position_dropout(position_embeddings)

        all_line_features_with_pos = all_line_features + position_embeddings

        valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(all_line_features_with_pos)
        all_line_features_with_pos = all_line_features_with_pos * valid_mask_expanded.float()

        # Stage 3: 行间结构建模
        line_structural_features = self.line_transformer(all_line_features_with_pos)

        valid_mask_expanded = valid_mask.unsqueeze(-1).expand_as(line_structural_features)
        line_structural_features = line_structural_features * valid_mask_expanded.float()

        # Stage 4: 行级语义特征融合
        line_semantic_combined = torch.cat([all_line_features_with_pos, line_structural_features], dim=-1)
        enhanced_line_features = self.line_semantic_fusion(line_semantic_combined)

        # Stage 5: 与全局特征融合后做行级分类
        global_features_expanded = global_features.unsqueeze(1).expand(-1, max_lines, -1)
        final_fused_features = torch.cat([enhanced_line_features, global_features_expanded], dim=-1)
        line_logits = self.line_classifier(final_fused_features)

        line_probs = torch.softmax(line_logits, dim=-1)[:, :, 1]
        line_probs = line_probs * valid_mask.float()

        return {
            'func_logits': func_logits,
            'line_logits': line_logits,
            'line_probs': line_probs
        }


"""collate 函数（与主实验完全一致）"""
def custom_collate(batch):
    reorganized = defaultdict(list)
    func_input_ids = []
    func_attention_mask = []
    func_labels = []

    for sample in batch:
        func_input_ids.append(sample['func_input']['func_input_ids'])
        func_attention_mask.append(sample['func_input']['func_attention_mask'])
        func_labels.append(sample['func_label'])

        for line_idx, line_data in enumerate(sample['line_inputs']):
            reorganized[line_idx].append({
                'input_ids': line_data['input_ids'],
                'attention_mask': line_data['attention_mask']
            })

    collated = {
        'func_input_ids': torch.stack(func_input_ids),
        'func_attention_mask': torch.stack(func_attention_mask),
        'func_label': torch.stack(func_labels).squeeze(),
        'line_inputs': [],
        'valid_mask': torch.stack([s['valid_mask'] for s in batch]),
        'line_labels': torch.stack([s['line_labels'] for s in batch])
    }

    for line_idx in range(len(reorganized)):
        collated['line_inputs'].append({
            'input_ids': torch.stack([x['input_ids'] for x in reorganized[line_idx]]),
            'attention_mask': torch.stack([x['attention_mask'] for x in reorganized[line_idx]])
        })

    return collated


# =============================================================================
# checkpoint 选择
# =============================================================================
def _parse_f1(path):
    """从 epoch_044__f1_0.4569.pth 之类的文件名中解析 F1"""
    m = re.search(r'f1[_-]?([0-9]*\.?[0-9]+)', os.path.basename(path), re.IGNORECASE)
    return float(m.group(1)) if m else -1.0


def select_checkpoint(explicit_path, model_dir):
    """返回 (checkpoint_path, 说明文本)"""
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {explicit_path}")
        return explicit_path, "由 --checkpoint 显式指定"

    dirs = [model_dir] if model_dir else CONFIG['candidate_model_dirs']
    searched = []
    for d in dirs:
        if not d or not os.path.isdir(d):
            searched.append(f"{d}（不存在，跳过）")
            continue
        files = glob.glob(os.path.join(d, '**', '*.pth'), recursive=True)
        if not files:
            searched.append(f"{d}（无 .pth 文件）")
            continue
        best = max(files, key=_parse_f1)
        searched.append(f"{d}（{len(files)} 个 .pth，选中 F1={_parse_f1(best):.4f}）")
        return best, "自动扫描选取（文件名 F1 最高）：" + " | ".join(searched)

    raise FileNotFoundError(
        "未能定位到 checkpoint。请在候选目录中放入 .pth，或显式传入 --checkpoint。\n"
        "已搜索：\n  " + "\n  ".join(searched)
    )


# =============================================================================
# 推理：只跑测试环节
# =============================================================================
@torch.no_grad()
def collect_predictions(model, loader, device, total_samples):
    """逐样本收集函数级预测、行级预测/概率/标签/有效掩码"""
    model.eval()
    records = []
    offset = 0

    pbar = tqdm(loader, desc="测试推理", unit="batch")
    for batch in pbar:
        bsz = batch['func_input_ids'].size(0)

        func_inputs = {
            'input_ids': batch['func_input_ids'].to(device, non_blocking=True),
            'attention_mask': batch['func_attention_mask'].to(device, non_blocking=True)
        }
        line_inputs = [
            {
                'input_ids': line['input_ids'].to(device, non_blocking=True),
                'attention_mask': line['attention_mask'].to(device, non_blocking=True)
            }
            for line in batch['line_inputs']
        ]
        valid_mask = batch['valid_mask'].to(device, non_blocking=True)

        outputs = model(func_inputs=func_inputs, line_inputs=line_inputs, valid_mask=valid_mask)

        func_logits = outputs['func_logits']
        func_prob = torch.softmax(func_logits.float(), dim=-1)[:, 1].cpu().numpy()
        func_pred = torch.argmax(func_logits, dim=-1).cpu().numpy()

        line_logits = outputs['line_logits']
        # 注意：这里用未掩码的 softmax 概率，掩码由 label != -100 完成
        line_prob = torch.softmax(line_logits.float(), dim=-1)[:, :, 1].cpu().numpy()
        line_pred = torch.argmax(line_logits, dim=-1).cpu().numpy()

        line_label = batch['line_labels'].cpu().numpy()
        vmask = batch['valid_mask'].cpu().numpy().astype(bool)

        for b in range(bsz):
            records.append({
                'sample_index': offset + b,
                'func_prob': float(func_prob[b]),
                'func_pred': int(func_pred[b]),
                'line_prob': line_prob[b],
                'line_pred': line_pred[b],
                'line_label': line_label[b],
                'valid_mask': vmask[b],
            })

        offset += bsz
        pbar.set_postfix({'已处理': f"{offset}/{total_samples}"})

    pbar.close()
    return records


# =============================================================================
# 统计：函数级 FP 样本的行级误报
# =============================================================================
def analyze(records, df, max_lines, test_csv=None, ckpt_path=None, ckpt_note=None, device=None):
    """
    返回:
        summary_rows   —— Summary sheet 的 (指标, 数值) 列表
        fp_sample_rows —— 每个函数级 FP 样本一行
        fp_line_rows   —— 每个被行级误报的代码行一行
        dist_rows      —— 行级误报行数区间分布
        stats          —— 供控制台打印的字典
    """
    tp = fp = tn = fn = 0
    n_truncated = 0

    fp_sample_rows = []
    fp_line_rows = []

    for rec in records:
        idx = rec['sample_index']
        row = df.iloc[idx]

        true_label = int(row['fun_label'])
        pred_label = rec['func_pred']

        # 函数级混淆矩阵
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        else:
            fn += 1

        # 只关心函数级误报样本：真值 0、预测 1
        if not (true_label == 0 and pred_label == 1):
            continue

        lines = json.loads(row['lines'])
        if len(lines) > max_lines:
            n_truncated += 1

        # 与 calculate_line_metrics 完全一致的有效行口径：label != -100
        valid = rec['line_label'] != -100
        n_lines = int(valid.sum())

        line_pred_pos = valid & (rec['line_pred'] == 1)
        n_pred_pos = int(line_pred_pos.sum())

        # 严格意义（真值 0 且预测 1）的行级误报；对真正的无漏洞函数二者应相等
        line_true_fp = line_pred_pos & (rec['line_label'] == 0)
        n_line_fp = int(line_true_fp.sum())

        probs = rec['line_prob'][valid]
        max_prob = float(probs.max()) if probs.size else 0.0
        mean_prob = float(probs.mean()) if probs.size else 0.0

        pred_pos_idx = np.where(line_true_fp)[0].tolist()

        fp_sample_rows.append({
            'sample_index': idx,
            'func_id': row.get('func_id', idx),
            'CWE_id': row.get('CWE_id', ''),
            'func_prob': round(rec['func_prob'], 6),
            'n_lines': n_lines,
            'n_line_pred_pos': n_pred_pos,
            'n_line_fp': n_line_fp,
            'line_fp_ratio': round(n_line_fp / n_lines, 6) if n_lines else 0.0,
            'max_line_prob': round(max_prob, 6),
            'mean_line_prob': round(mean_prob, 6),
            'has_line_fp': 1 if n_line_fp > 0 else 0,
            'pred_pos_line_idx_0based': ';'.join(str(i) for i in pred_pos_idx[:200]),
        })

        for j in pred_pos_idx:
            fp_line_rows.append({
                'sample_index': idx,
                'func_id': row.get('func_id', idx),
                'CWE_id': row.get('CWE_id', ''),
                'line_index_0based': int(j),
                'line_index_1based': int(j) + 1,
                'line_text': str(lines[j]) if j < len(lines) else '',
                'line_prob': round(float(rec['line_prob'][j]), 6),
                'line_label': int(rec['line_label'][j]),
            })

    # ---------------- 汇总 ----------------
    n_samples = len(records)
    n_fp = len(fp_sample_rows)

    def _conf_metrics(tp_, fp_, tn_, fn_):
        acc = (tp_ + tn_) / max(tp_ + fp_ + tn_ + fn_, 1)
        precision = tp_ / max(tp_ + fp_, 1)
        recall = tp_ / max(tp_ + fn_, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return acc, precision, recall, f1

    func_acc, func_pre, func_rec, func_f1 = _conf_metrics(tp, fp, tn, fn)

    if n_fp > 0:
        counts = np.array([r['n_line_fp'] for r in fp_sample_rows], dtype=float)
        line_counts = np.array([r['n_lines'] for r in fp_sample_rows], dtype=float)
        ratios = np.array([r['line_fp_ratio'] for r in fp_sample_rows], dtype=float)
        n_with_line_fp = int((counts > 0).sum())
        n_without_line_fp = int((counts == 0).sum())
        total_line_fp = int(counts.sum())
        total_lines = int(line_counts.sum())
    else:
        counts = np.array([], dtype=float)
        line_counts = np.array([], dtype=float)
        ratios = np.array([], dtype=float)
        n_with_line_fp = n_without_line_fp = total_line_fp = total_lines = 0

    summary_rows = [
        ('生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        ('测试集', test_csv if test_csv else CONFIG['test_csv']),
        ('checkpoint', ckpt_path if ckpt_path else ''),
        ('checkpoint 选择方式', ckpt_note if ckpt_note else ''),
        ('device', str(device) if device is not None else ''),
        ('测试样本总数', n_samples),
        ('最大行数 max_lines', max_lines),
        ('函数样本行数被截断的数量', n_truncated),
        ('', ''),
        ('【函数级混淆矩阵】', ''),
        ('TP（真 1 / 预测 1）', tp),
        ('FP（真 0 / 预测 1）', fp),
        ('TN（真 0 / 预测 0）', tn),
        ('FN（真 1 / 预测 0）', fn),
        ('函数级 Accuracy', round(func_acc, 6)),
        ('函数级 Precision', round(func_pre, 6)),
        ('函数级 Recall', round(func_rec, 6)),
        ('函数级 F1', round(func_f1, 6)),
        ('', ''),
        ('【函数级误报（FP）样本的行级表现】', ''),
        ('函数级 FP 样本数', n_fp),
        ('其中：行级至少误报 1 行的样本数', n_with_line_fp),
        ('其中：行级完全未误报（0 行）的样本数', n_without_line_fp),
        ('行级存在误报的 FP 样本占比', round(n_with_line_fp / n_fp, 6) if n_fp else 0.0),
        ('FP 样本的行级误报总行数', total_line_fp),
        ('FP 样本的有效代码行总数', total_lines),
        ('行级误报行数占 FP 样本总行数比例', round(total_line_fp / total_lines, 6) if total_lines else 0.0),
        ('', ''),
        ('【每个 FP 样本的行级误报行数统计】', ''),
        ('均值', round(float(counts.mean()), 4) if counts.size else 0.0),
        ('中位数', round(float(np.median(counts)), 4) if counts.size else 0.0),
        ('标准差', round(float(counts.std(ddof=1)), 4) if counts.size > 1 else 0.0),
        ('最小值', int(counts.min()) if counts.size else 0),
        ('最大值', int(counts.max()) if counts.size else 0),
        ('P25', round(float(np.percentile(counts, 25)), 4) if counts.size else 0.0),
        ('P75', round(float(np.percentile(counts, 75)), 4) if counts.size else 0.0),
        ('P90', round(float(np.percentile(counts, 90)), 4) if counts.size else 0.0),
        ('P95', round(float(np.percentile(counts, 95)), 4) if counts.size else 0.0),
        ('', ''),
        ('【行级误报比例（误报行数 / 该函数有效行数）】', ''),
        ('均值', round(float(ratios.mean()), 6) if ratios.size else 0.0),
        ('中位数', round(float(np.median(ratios)), 6) if ratios.size else 0.0),
        ('最大值', round(float(ratios.max()), 6) if ratios.size else 0.0),
    ]

    # ---------------- 区间分布 ----------------
    bins = [-1, 0, 1, 2, 3, 5, 10, 20, float('inf')]
    labels = ['0 行', '1 行', '2 行', '3 行', '4-5 行', '6-10 行', '11-20 行', '>20 行']
    dist_rows = []
    if counts.size:
        binned = pd.cut(counts, bins=bins, labels=labels)
        vc = binned.value_counts().reindex(labels, fill_value=0)
        for lb in labels:
            c = int(vc[lb])
            dist_rows.append({
                '行级误报行数区间': lb,
                '样本数': c,
                '占 FP 样本比例': round(c / n_fp, 6) if n_fp else 0.0,
            })
    else:
        for lb in labels:
            dist_rows.append({'行级误报行数区间': lb, '样本数': 0, '占 FP 样本比例': 0.0})

    stats = {
        'n_samples': n_samples,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'func_acc': func_acc, 'func_pre': func_pre, 'func_rec': func_rec, 'func_f1': func_f1,
        'n_fp': n_fp,
        'n_with_line_fp': n_with_line_fp,
        'n_without_line_fp': n_without_line_fp,
        'total_line_fp': total_line_fp,
        'mean_line_fp': float(counts.mean()) if counts.size else 0.0,
        'median_line_fp': float(np.median(counts)) if counts.size else 0.0,
        'max_line_fp': int(counts.max()) if counts.size else 0,
    }

    return summary_rows, fp_sample_rows, fp_line_rows, dist_rows, stats


# =============================================================================
# 输出
# =============================================================================
COLUMN_NOTES = [
    ('sheet', '列名', '说明'),
    ('Summary', '指标 / 数值', '整体统计：函数级混淆矩阵、FP 样本数、行级误报行数统计量'),
    ('FP_Samples', 'sample_index', '该样本在测试集中的行号（0 开始，与 test_csv 行序一致）'),
    ('FP_Samples', 'func_id', '数据集中的函数唯一标识'),
    ('FP_Samples', 'CWE_id', '漏洞类型编号'),
    ('FP_Samples', 'func_prob', '函数级分类器输出的 P(有漏洞)'),
    ('FP_Samples', 'n_lines', '该函数的有效代码行数（≤ max_lines=48，与行级指标口径一致）'),
    ('FP_Samples', 'n_line_pred_pos', '行级预测为 1 的行数（有效行范围内）'),
    ('FP_Samples', 'n_line_fp', '行级误报行数（预测 1 且真值 0）—— 核心指标'),
    ('FP_Samples', 'line_fp_ratio', 'n_line_fp / n_lines'),
    ('FP_Samples', 'max_line_prob', '该函数所有有效行中最高的行级漏洞概率'),
    ('FP_Samples', 'mean_line_prob', '该函数所有有效行的平均行级漏洞概率'),
    ('FP_Samples', 'has_line_fp', '是否至少误报 1 行（1 是 / 0 否）'),
    ('FP_Samples', 'pred_pos_line_idx_0based', '被误报的行下标（0 开始，分号分隔，最多 200 个）'),
    ('Line_FP_Detail', 'line_index_1based', '被误报行在函数中的行号（1 开始，便于对照代码）'),
    ('Line_FP_Detail', 'line_text', '该行的源码文本（已按 tokenizer 前处理，空行为空白字符）'),
    ('Line_FP_Detail', 'line_prob', '该行的行级漏洞概率'),
    ('Distribution', '行级误报行数区间', '按每个 FP 样本的 n_line_fp 分桶统计样本数'),
]


def write_outputs(xlsx_path, sheets):
    """写 Excel；若 openpyxl 缺失则降级为 CSV"""
    try:
        import openpyxl  # noqa: F401
        from openpyxl import load_workbook
    except ImportError:
        csv_dir = os.path.splitext(xlsx_path)[0] + '_csv'
        os.makedirs(csv_dir, exist_ok=True)
        for name, df in sheets.items():
            df.to_csv(os.path.join(csv_dir, f"{name}.csv"), index=False, encoding='utf-8-sig')
        print("\n[警告] 未安装 openpyxl，已改为输出 CSV：\n  " + csv_dir)
        print("        如需 Excel，请先执行： pip install openpyxl")
        return False

    os.makedirs(os.path.dirname(os.path.abspath(xlsx_path)) or '.', exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    # 简单的排版：冻结首行、自适应列宽
    wb = load_workbook(xlsx_path)
    for ws in wb.worksheets:
        ws.freeze_panes = 'A2'
        for col in ws.columns:
            longest = max((len(str(c.value)) if c.value is not None else 0) for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(12, longest * 1.1), 60)
    wb.save(xlsx_path)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='函数级误报样本的行级误报统计分析（只跑测试环节）'
    )
    parser.add_argument('--checkpoint', default=None,
                        help='模型权重 .pth 路径；不指定则自动在候选目录中选 F1 最高者')
    parser.add_argument('--model_dir', default=None,
                        help='未指定 --checkpoint 时，在该目录（递归）中查找 .pth')
    parser.add_argument('--test_csv', default=CONFIG['test_csv'],
                        help=f'测试集路径，默认 {CONFIG["test_csv"]}')
    parser.add_argument('--out_xlsx', default=None,
                        help='输出 Excel 路径；默认 <out_dir>/func_fp_line_analysis_<ckpt>.xlsx')
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--device', default=None, help="如 cuda:0 / cpu，默认自动选择")
    parser.add_argument('--max_samples', type=int, default=None,
                        help='仅取前 N 个测试样本（用于快速冒烟测试）')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 1) 定位 checkpoint
    ckpt_path, ckpt_note = select_checkpoint(args.checkpoint, args.model_dir)

    # 2) 读取测试集（顺序必须与 DataLoader 一致，故 shuffle=False）
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    if args.max_samples is not None:
        test_df = test_df.iloc[:args.max_samples].reset_index(drop=True)
    print(f"测试集: {args.test_csv}，样本数: {len(test_df)}")

    test_dataset = CodeDataset(
        test_df,
        max_lines=CONFIG['max_lines'],
        max_func_length=CONFIG['max_func_length'],
        max_line_length=CONFIG['max_line_length']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )

    # 3) 加载模型
    print(f"使用模型: {ckpt_path}\n  （{ckpt_note}）")
    model = VulDetectionModel()
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 4) 只跑测试推理
    records = collect_predictions(model, test_loader, device, len(test_df))
    if len(records) != len(test_df):
        raise RuntimeError(
            f"预测样本数({len(records)})与测试集行数({len(test_df)})不一致，"
            f"样本下标将错位，已中止。请检查 collate 与 batch 逻辑。"
        )

    # 5) 统计
    summary_rows, fp_sample_rows, fp_line_rows, dist_rows, stats = analyze(
        records, test_df, CONFIG['max_lines'],
        test_csv=args.test_csv, ckpt_path=ckpt_path, ckpt_note=ckpt_note, device=device
    )

    # 6) 组装输出
    df_summary = pd.DataFrame(summary_rows, columns=['指标', '数值'])

    df_fp_samples = pd.DataFrame(fp_sample_rows)
    df_fp_lines = pd.DataFrame(fp_line_rows)
    df_dist = pd.DataFrame(dist_rows)
    df_notes = pd.DataFrame(COLUMN_NOTES[1:], columns=list(COLUMN_NOTES[0]))

    # 排序：误报行数多的排前面，便于人工核查
    if len(df_fp_samples):
        df_fp_samples = df_fp_samples.sort_values(
            by=['n_line_fp', 'max_line_prob'], ascending=[False, False]
        ).reset_index(drop=True)
    if len(df_fp_lines):
        df_fp_lines = df_fp_lines.sort_values(
            by=['sample_index', 'line_index_0based']
        ).reset_index(drop=True)

    sheets = {
        'Summary': df_summary,
        'FP_Samples': df_fp_samples,
        'Line_FP_Detail': df_fp_lines,
        'Distribution': df_dist,
        'Column_Notes': df_notes,
    }

    # 7) 输出文件名
    if args.out_xlsx:
        out_xlsx = args.out_xlsx
    else:
        ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_xlsx = os.path.join(CONFIG['out_dir'], f'func_fp_line_analysis_{ckpt_stem}_{stamp}.xlsx')

    ok = write_outputs(out_xlsx, sheets)

    # 8) 控制台汇总
    print("\n" + "=" * 70)
    print("函数级误报（FP）样本的行级误报统计")
    print("=" * 70)
    print(f"测试样本数      : {stats['n_samples']}")
    print(f"函数级 TP/FP/TN/FN: {stats['tp']}/{stats['fp']}/{stats['tn']}/{stats['fn']}")
    print(f"函数级 Acc/P/R/F1 : {stats['func_acc']:.4f} / {stats['func_pre']:.4f} / "
          f"{stats['func_rec']:.4f} / {stats['func_f1']:.4f}")
    print(f"函数级 FP 样本数 : {stats['n_fp']}")
    print(f"  其中行级≥1 误报: {stats['n_with_line_fp']}")
    print(f"  其中行级 0 误报: {stats['n_without_line_fp']}")
    print(f"行级误报总行数   : {stats['total_line_fp']}")
    print(f"每个 FP 样本行级误报行数: 均值 {stats['mean_line_fp']:.2f} | "
          f"中位数 {stats['median_line_fp']:.1f} | 最大 {stats['max_line_fp']}")
    print("-" * 70)
    print("行级误报行数分布:")
    for r in dist_rows:
        print(f"  {r['行级误报行数区间']:>10s}: {r['样本数']:>5d}  ({r['占 FP 样本比例']:.2%})")
    print("=" * 70)
    print(f"结果已保存: {out_xlsx}" if ok else f"结果已保存(CSV): {out_xlsx}")
    print(f"模型权重  : {ckpt_path}")


if __name__ == '__main__':
    main()
