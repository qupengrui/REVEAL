import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup  # 学习率调度器，使用带预热的余弦退火策略
from transformers import RobertaTokenizer, T5ForConditionalGeneration  # CodeT5使用RobertaTokenizer + T5架构
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter

device_ids = [0, 1]  # 可自定义需要使用的GPU编号
"""路径参数及超参设置"""
CONFIG = {
    "line_bert_path": "/data/prqu/prqu_files/pretrained_models/CodeT5_base",
    "func_bert_path": "/data/prqu/prqu_files/pretrained_models/CodeT5_base",
    "max_func_length": 512,
    "max_line_length": 32,
    "max_lines": 48,
    "num_classes": 2,
    "batch_size": 16,
    "lr": 1e-5,
    "epochs": 50,
    "device": torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"),

    "log_dir": "/data/prqu/prqu_files/dir_logs_test",  # TensorBoard日志路径
    "log_file": "/data/prqu/prqu_files/save_file/training_logs_CodeT5_Attention.txt",  # 训练日志路径
    "save_file": "/data/prqu/prqu_files/save_file",  # 结果保存目录
    "model_dir": "/data/prqu/prqu_files/save_file/models_metrics_modified_CodeT5_Attention",
    "max_saved_models": 1  # 最大保存模型数量
}

"""数据集处理"""
class CodeDataset(Dataset):
    def __init__(self, df, max_lines=64, max_func_length=512, max_line_length=32):
        self.df = df
        self.max_lines = max_lines
        self.max_func_length = max_func_length
        self.max_line_length = max_line_length
        self.tokenizer = RobertaTokenizer.from_pretrained(CONFIG['line_bert_path'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 解析JSON格式的行数据和标签
        func_before = self.df.iloc[idx]['func_before']
        func_label = self.df.iloc[idx]['fun_label']
        lines = json.loads(self.df.iloc[idx]['lines'])[:self.max_lines]
        line_labels = json.loads(self.df.iloc[idx]['line_labels'])[:self.max_lines]  # 函数级编码 - 不使用特殊token
        func_encoding = self.tokenizer(
            func_before,
            max_length=self.max_func_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,  # 不添加特殊token
            return_attention_mask=True
        )
        # 行级编码 - 不使用特殊token
        line_embeddings = []
        valid_masks = []
        for line in lines:
            # 处理空行或空字符串
            if not line or line.strip() == "":
                line = " "  # 用空格替代空字符串，确保tokenizer能处理

            encoding = self.tokenizer(
                line,
                max_length=self.max_line_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_special_tokens=False,  # 不添加特殊token
                return_attention_mask=True
            )

            # 确保 tensor 维度正确
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)

            # 如果维度不对，进行修正
            if input_ids.dim() == 0:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)

            # 确保长度为 max_line_length
            if input_ids.size(0) != self.max_line_length:
                if input_ids.size(0) < self.max_line_length:
                    # 填充到指定长度
                    pad_length = self.max_line_length - input_ids.size(0)
                    input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long)])
                else:
                    # 截断到指定长度
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
            line_labels.append(-100)  # 填充行标签

        return {
            'func_input': {
                'func_input_ids': func_encoding['input_ids'].squeeze(0),
                'func_attention_mask': func_encoding['attention_mask'].squeeze(0)
            },
            'func_label': torch.tensor(func_label, dtype=torch.long),
            'line_inputs': line_embeddings,  # List[Dict] 每行包含完整token信息
            'line_labels': torch.LongTensor(line_labels),
            'valid_mask': torch.BoolTensor(valid_masks)
        }


"""交叉注意力机制模块"""
class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        # Query, Key, Value投影层
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Dropout和LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, query, key, value, mask=None):
        batch_size, max_lines, _ = key.size()
        
        # 线性投影
        Q = self.query_projection(query)  # [batch, 1, hidden_size]
        K = self.key_projection(key)      # [batch, max_lines, hidden_size]
        V = self.value_projection(value)  # [batch, max_lines, hidden_size]
        
        # 重塑为多头注意力格式
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, 1, head_dim]
        K = K.view(batch_size, max_lines, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, max_lines, head_dim]
        V = V.view(batch_size, max_lines, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, max_lines, head_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [batch, num_heads, 1, max_lines]
        
        # 应用掩码（将填充行的注意力分数设为负无穷）
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, max_lines]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, num_heads, 1, max_lines]
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        attended_output = torch.matmul(attention_weights, V)  # [batch, num_heads, 1, head_dim]
        
        # 重塑并拼接多头结果
        attended_output = attended_output.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size)  # [batch, 1, hidden_size]
        
        # 输出投影
        attended_output = self.output_projection(attended_output)  # [batch, 1, hidden_size]
        
        # 扩展到所有行位置，并与原始行级特征进行残差连接
        attended_output_expanded = attended_output.expand(-1, max_lines, -1)  # [batch, max_lines, hidden_size]
        
        # 残差连接 + LayerNorm
        fused_features = self.layer_norm(value + attended_output_expanded)  # [batch, max_lines, hidden_size]
        
        return fused_features


""""模型类"""
class VulDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载tokenizer - CodeT5使用RobertaTokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(CONFIG['line_bert_path'])

        # CodeT5-1: 行级特征提取器，处理单行代码
        self.codeT51 = T5ForConditionalGeneration.from_pretrained(CONFIG['line_bert_path'])
        for param in self.codeT51.parameters():
            param.requires_grad = False  # 冻结参数

        # CodeT5-2: 函数级特征提取器
        self.codeT52 = T5ForConditionalGeneration.from_pretrained(CONFIG['func_bert_path'])
        # 只解冻后六层进行微调
        for param in self.codeT52.encoder.block[-6:].parameters():
            param.requires_grad = True

        # 维度适配器：统一行级和函数级特征维度
        self.dim_adapter = nn.Linear(
            self.codeT51.config.d_model,
            self.codeT52.config.d_model
        )

        # 交叉注意力模块：函数级特征作为Query，行级特征作为Key/Value
        self.cross_attention = CrossAttentionModule(
            hidden_size=self.codeT52.config.d_model,
            num_heads=8,
            dropout=0.1
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

        # 行级分类头（使用交叉注意力融合后的特征）
        self.line_classifier = nn.Sequential(
            nn.Linear(self.codeT52.config.d_model, 128),  # 使用融合后的特征维度
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, CONFIG['num_classes'])
        )

    def forward(self, func_inputs, line_inputs, valid_mask):
        """
        Args:
            func_inputs: 函数级输入 {'input_ids': [batch, func_seq_len], 'attention_mask': [batch, func_seq_len]}
            line_inputs: 行级输入列表 [{'input_ids': [batch, line_seq_len], 'attention_mask': [batch, line_seq_len]}]
            valid_mask: 有效行掩码 [batch, max_lines]
        """
        batch_size = func_inputs['input_ids'].size(0)
        max_lines = len(line_inputs)
        device = func_inputs['input_ids'].device

        # Stage 1: 使用 CodeT5-2 提取函数级全局特征
        global_output = self.codeT52.encoder(
            input_ids=func_inputs['input_ids'],
            attention_mask=func_inputs['attention_mask']
        )

        # 使用加权平均池化获取全局特征（替代CLS token）
        last_hidden = global_output.last_hidden_state  # [batch, seq_len, hidden]
        attention_mask = func_inputs['attention_mask']  # [batch, seq_len]

        # 计算加权平均（仅考虑非填充token）
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)  # [batch, hidden]
        token_count = torch.sum(attention_mask, dim=1, keepdim=True)  # [batch, 1]
        global_features = sum_hidden / torch.clamp(token_count, min=1)  # [batch, hidden]

        # 函数级分类
        func_logits = self.func_classifier(global_features)

        # Stage 2: 使用 CodeT5-1 逐行提取行级特征
        line_features_list = []
        for line_idx in range(max_lines):
            line_data = line_inputs[line_idx]
            outputs = self.codeT51.encoder(
                input_ids=line_data['input_ids'],
                attention_mask=line_data['attention_mask']
            )
            # 同样使用加权平均池化获取行级特征
            line_hidden = outputs.last_hidden_state  # [batch, line_seq_len, hidden]
            line_mask = line_data['attention_mask']  # [batch, line_seq_len]

            line_mask_expanded = line_mask.unsqueeze(-1).expand(line_hidden.size())
            line_sum_hidden = torch.sum(line_hidden * line_mask_expanded, dim=1)  # [batch, hidden]
            line_token_count = torch.sum(line_mask, dim=1, keepdim=True)  # [batch, 1]
            line_avg = line_sum_hidden / torch.clamp(line_token_count, min=1)  # [batch, hidden]

            # 维度适配
            adapted_line_features = self.dim_adapter(line_avg)  # [batch, hidden]
            line_features_list.append(adapted_line_features)

        # 堆叠所有行特征 [batch, max_lines, hidden]
        all_line_features = torch.stack(line_features_list, dim=1)

        # Stage 3: 函数级特征作为Query，行级特征作为Key和Value，使用交叉注意力进行特征融合
        query = global_features.unsqueeze(1)  # [batch, 1, hidden] - 函数级特征作为Query
        key = all_line_features  # [batch, max_lines, hidden] - 行级特征作为Key
        value = all_line_features  # [batch, max_lines, hidden] - 行级特征作为Value
        
        # 应用交叉注意力机制，得到融合后的行级特征
        fused_features = self.cross_attention(
            query=query,
            key=key,
            value=value,
            mask=valid_mask  # [batch, max_lines]
        )  # [batch, max_lines, hidden]

        # 行级分类
        line_logits = self.line_classifier(fused_features)  # [batch, max_lines, num_classes]

        # 计算行级概率并应用掩码
        line_probs = torch.softmax(line_logits, dim=-1)[:, :, 1]  # [batch, max_lines]
        line_probs = line_probs * valid_mask.float()  # 将填充行的概率置零

        return {
            'func_logits': func_logits,
            'line_logits': line_logits,
            'line_probs': line_probs
        }


"""重写collate函数"""
def custom_collate(batch):
    reorganized = defaultdict(list)
    func_input_ids = []
    func_attention_mask = []
    func_labels = []  # 收集func_label

    for sample in batch:
        func_input_ids.append(sample['func_input']['func_input_ids'])
        func_attention_mask.append(sample['func_input']['func_attention_mask'])
        func_labels.append(sample['func_label'])  # 收集func_label

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
        # 调试：检查tensor维度
        tensor_shapes = [x['input_ids'].shape for x in reorganized[line_idx]]
        if not all(shape == tensor_shapes[0] for shape in tensor_shapes):
            print(f"警告：第{line_idx}行的tensor维度不一致: {tensor_shapes}")
            # 修正维度不一致的问题
            max_len = max(shape[0] if len(shape) > 0 else 0 for shape in tensor_shapes)
            if max_len == 0:
                max_len = 32  # 默认长度

            corrected_tensors = []
            for x in reorganized[line_idx]:
                input_ids = x['input_ids']
                attention_mask = x['attention_mask']

                # 确保维度正确
                if input_ids.dim() == 0:
                    input_ids = torch.zeros(max_len, dtype=torch.long)
                    attention_mask = torch.zeros(max_len, dtype=torch.long)
                elif input_ids.size(0) != max_len:
                    if input_ids.size(0) < max_len:
                        pad_len = max_len - input_ids.size(0)
                        input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
                        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
                    else:
                        input_ids = input_ids[:max_len]
                        attention_mask = attention_mask[:max_len]

                corrected_tensors.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })

            line_batch = {
                'input_ids': torch.stack([x['input_ids'] for x in corrected_tensors]),
                'attention_mask': torch.stack([x['attention_mask'] for x in corrected_tensors])
            }
        else:
            line_batch = {
                'input_ids': torch.stack([x['input_ids'] for x in reorganized[line_idx]]),
                'attention_mask': torch.stack([x['attention_mask'] for x in reorganized[line_idx]])
            }
        collated['line_inputs'].append(line_batch)

    return collated


"""计算函数级指标——F1、acc、precision、recall"""
def calculate_func_metrics(preds, labels):
    preds = np.array(preds).flatten()
    labels = np.array(labels).flatten()

    metrics = {
        'acc': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }

    if len(labels) == 0:
        return metrics

    # 计算混淆矩阵元素
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    TN = np.sum((preds == 0) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    # 指标计算
    metrics['acc'] = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) else 0
    metrics['precision'] = TP / (TP + FP) if (TP + FP) else 0
    metrics['recall'] = TP / (TP + FN) if (TP + FN) else 0
    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if \
        (metrics['precision'] + metrics['recall']) else 0

    return metrics


"""计算行级指标——额外添加ROCAUC、PRAUC（修正版本）"""
def calculate_line_metrics(preds_2d, probs_2d, labels_2d, ignore_index=-100):
    """
    修正版本的行级指标计算函数
    Args:
        preds_2d: List[numpy.ndarray] - 每个元素是一个样本的行级预测 [max_lines]
        probs_2d: List[numpy.ndarray] - 每个元素是一个样本的行级概率 [max_lines]
        labels_2d: List[numpy.ndarray] - 每个元素是一个样本的行级标签 [max_lines]
        ignore_index: int - 需要忽略的标签值（通常为填充值-100）
    """
    # 初始化收集列表
    all_preds = []
    all_probs = []
    all_labels = []

    # 逐样本处理，先过滤再聚合
    for sample_preds, sample_probs, sample_labels in zip(preds_2d, probs_2d, labels_2d):
        # 确保输入是numpy数组
        sample_preds = np.array(sample_preds)
        sample_probs = np.array(sample_probs)
        sample_labels = np.array(sample_labels)

        # 样本级掩码：过滤无效行
        valid_mask = (sample_labels != ignore_index)

        # 提取有效的预测、概率和标签
        valid_preds = sample_preds[valid_mask]
        valid_probs = sample_probs[valid_mask]
        valid_labels = sample_labels[valid_mask]

        # 添加到全局列表
        all_preds.extend(valid_preds)
        all_probs.extend(valid_probs)
        all_labels.extend(valid_labels)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = {
        'acc': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'IoU': 0.0,
        'ROC-AUC': 0.0,
        'PR-AUC': 0.0,
        'Top-5%Acc': 0.0,
        'Top-10%Acc': 0.0
    }

    if len(all_labels) == 0:
        print("警告: 没有有效的行级标签数据")
        return metrics

    # 计算混淆矩阵元素
    TP = np.sum((all_preds == 1) & (all_labels == 1))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    # # 调试信息（可选，训练时可注释掉）
    # total_samples = len(all_labels)
    # pos_samples = np.sum(all_labels == 1)
    # neg_samples = np.sum(all_labels == 0)
    # print(f"行级数据统计: 总样本={total_samples}, 正样本={pos_samples}, 负样本={neg_samples}")
    # print(f"混淆矩阵: TP={TP}, FP={FP}, TN={TN}, FN={FN}")

    # 基础指标计算
    total = TP + FP + TN + FN
    metrics['acc'] = (TP + TN) / total if total > 0 else 0.0
    metrics['precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    metrics['recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1分数
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0

    # IoU计算
    metrics['IoU'] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    # AUC指标计算
    try:
        # 检查是否有足够的正负样本
        if len(np.unique(all_labels)) < 2:
            print("警告: 标签中只有一个类别，无法计算AUC")
            metrics['ROC-AUC'] = 0.0
            metrics['PR-AUC'] = 0.0
        else:
            metrics['ROC-AUC'] = roc_auc_score(all_labels, all_probs)
            metrics['PR-AUC'] = average_precision_score(all_labels, all_probs)
    except ValueError as e:
        print(f"AUC计算出错: {e}")
        metrics['ROC-AUC'] = 0.0
        metrics['PR-AUC'] = 0.0

    # 计算Top-K% Acc
    if len(all_labels) > 0:
        # 确保有正样本，否则指标无意义
        total_vulnerable_lines = np.sum(all_labels == 1)
        if total_vulnerable_lines > 0:
            # 2. 将(概率, 标签)组合在一起，然后按概率降序排序
            combined = list(zip(all_probs, all_labels))
            # 按概率降序排序
            combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)

            # 3. 提取排序后的标签
            sorted_labels = [label for _, label in combined_sorted]

            # 4. 计算需要检查的行数
            total_valid_lines = len(sorted_labels)
            top5_cutoff = int(total_valid_lines * 0.05)
            top10_cutoff = int(total_valid_lines * 0.10)

            # 5. 计算指标
            # Top5% Acc: 前5%的行中，有多少是真正的漏洞行 (Recall)
            metrics['Top-5%Acc'] = np.sum(sorted_labels[:top5_cutoff]) / total_vulnerable_lines
            # Top10% Acc: 前10%的行中，有多少是真正的漏洞行 (Recall)
            metrics['Top-10%Acc'] = np.sum(sorted_labels[:top10_cutoff]) / total_vulnerable_lines
        else:
            print("警告: 没有真实的漏洞行，无法计算Top-% Acc")
            metrics['Top-5%Acc'] = 0.0
            metrics['Top-10%Acc'] = 0.0
    else:
        print("警告: 没有有效的行级数据")
        metrics['Top-5%Acc'] = 0.0
        metrics['Top-10%Acc'] = 0.0

    return metrics


""""打印“样本结构化”的行级预测信息"""
def print_line_debug_info(line_preds, line_labels, ignore_index=-100):
    # 行级标签和预测的全局统计
    global_label_counts = defaultdict(int)
    global_pred_counts = defaultdict(int)

    for pred, label in zip(line_preds, line_labels):
        if len(label) == 0:
            continue

        # 转换为numpy数组以便于处理
        label = np.array(label)
        pred = np.array(pred)

        # 创建有效标签的掩码（不等于ignore_index的位置）
        valid_mask = (label != ignore_index)

        # 如果没有有效标签，跳过这个样本
        if not np.any(valid_mask):
            continue

        # 只保留有效的标签和对应的预测
        valid_labels = label[valid_mask]
        valid_preds = pred[valid_mask]

        # 统计有效标签的分布
        unique_labels, label_counts = np.unique(valid_labels, return_counts=True)
        unique_preds, pred_counts = np.unique(valid_preds, return_counts=True)

        # 聚合到全局统计
        for lbl, cnt in zip(unique_labels, label_counts):
            global_label_counts[lbl] += cnt
        for prd, cnt in zip(unique_preds, pred_counts):
            global_pred_counts[prd] += cnt

    # 只打印行级标签和预测的分布信息
    print(f"行级标签分布: {dict(global_label_counts)}")
    print(f"行级预测分布: {dict(global_pred_counts)}")


"""写入训练日志至文件"""
def write_logfile(phase, epoch, func_metrics, line_metrics, loss):
    log_entry = (
        f"{phase}:Epoch {epoch:03d}:\n"
        f"\t[Func] Acc: {func_metrics['acc']:.4f} | F1: {func_metrics['f1']:.4f} | Pre: {func_metrics['precision']:.4f}"
        f" | Rec: {func_metrics['recall']:.4f}\n"
        f"\t[Line] Acc: {line_metrics['acc']:.4f} | F1: {line_metrics['f1']:.4f} | Pre: {line_metrics['precision']:.4f}"
        f" | Rec: {line_metrics['recall']:.4f} | IoU: {line_metrics['IoU']:.4f} | ROC-AUC: {line_metrics['ROC-AUC']:.4f}"
        f" | PR-AUC: {line_metrics['PR-AUC']:.4f}\n"
        f"\t[Top-K] Top-5%Acc: {line_metrics['Top-5%Acc']:.4f} | Top-10%Acc: {line_metrics['Top-10%Acc']:.4f}\n"
        f"\tLoss: {loss:.4f}\n"
    )
    with open(CONFIG["log_file"], "a") as f:
        f.write(log_entry)


"""打印级指标信息"""
def print_metrics(phase, loss, fun_metrics, line_metrics):
    print(f"{phase} Loss: {loss:.4f} | Func_Acc: {fun_metrics['acc']:.4f} | Func_Pre: {fun_metrics['precision']:.4f} | "
          f"Func_Rec: {fun_metrics['recall']:.4f} | Func_F1: {fun_metrics['f1']:.4f}")
    print(f"\t\tLine_Acc: {line_metrics['acc']:.4f} | Line_Pre: {line_metrics['precision']:.4f} | Line_Rec: "
          f"{line_metrics['recall']:.4f} | Line_F1: {line_metrics['f1']:.4f} | Line_IoU: {line_metrics['IoU']:.4f} | "
          f"Line_ROCAUC: {line_metrics['ROC-AUC']:.4f} | Line_PRAUC: {line_metrics['PR-AUC']:.4f}")
    print(f"\t\tTop-5%Acc: {line_metrics['Top-5%Acc']:.4f} | Top-10%Acc: {line_metrics['Top-10%Acc']:.4f}")


"""保存当前模型到跟踪列表（如果进入前5名）"""
def save_top_models(current_f1, epoch, model):
    global top_models

    # 如果列表未满或当前F1值高于最低记录
    if len(top_models) < CONFIG["max_saved_models"] or current_f1 > top_models[-1]["f1"]:
        # 生成唯一模型文件名
        model_name = f"epoch_{epoch:03d}__f1_{current_f1:.4f}.pth"
        model_path = os.path.join(CONFIG["model_dir"], model_name)

        # 保存模型参数
        torch.save(model.state_dict(), model_path)

        # 添加新记录
        top_models.append({
            "f1": current_f1,
            "epoch": epoch,
            "path": model_path
        })

        # 按F1值降序排序
        top_models.sort(key=lambda x: -x["f1"])

        # 删除超出数量的最差模型
        if len(top_models) > CONFIG["max_saved_models"]:
            removed = top_models.pop()
            if os.path.exists(removed["path"]):
                os.remove(removed["path"])


"""动态计算行级的类别权重"""
def calculate_pos_weight(labels):
    # 过滤无效标签（填充值-100）
    valid_mask = (labels != -100)
    valid_labels = labels[valid_mask]

    # 统计正负样本数量
    pos_count = (valid_labels == 1).sum().float()
    neg_count = (valid_labels == 0).sum().float()

    # 计算动态权重（负样本数/正样本数）
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0  # 防止除以零
    return pos_weight


"""加权损失函数计算"""
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.6):
        super().__init__()
        self.alpha = alpha  # 函数级损失权重

    def forward(self, outputs, labels):
        # 解包标签
        func_labels = labels['func'].squeeze(-1)
        line_labels = labels['line']
        valid_mask = labels['mask']

        # 函数级损失
        func_loss = F.cross_entropy(outputs['func_logits'], func_labels)

        # 行级损失
        pos_weight = calculate_pos_weight(line_labels)  # 计算行级的正样本比例

        line_logits = outputs['line_logits'].view(-1, 2)  # [batch*lines, 2]
        line_labels = labels['line'].view(-1)  # [batch*lines]
        valid_mask = (line_labels != -100)

        line_loss = F.cross_entropy(
            line_logits[valid_mask],
            line_labels[valid_mask],
            weight=torch.tensor([1.0, pos_weight], device=line_logits.device)
        )

        total_loss = self.alpha * func_loss + (1 - self.alpha) * line_loss
        # 加权总损失
        return total_loss


""""训练"""
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer):
    model.train()
    criterion = MultiTaskLoss(alpha=0.6)
    current_step = epoch * len(dataloader)  # 计算全局步数起点

    total_loss = 0.0
    # 初始化预测和标签收集列表
    func_preds, func_labels = [], []
    line_preds, line_probs, line_labels = [], [], []

    # 初始化进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]", unit="batch")

    for batch_idx, batch in enumerate(pbar):
        # 转换数据到GPU
        func_inputs = {
            'input_ids': batch['func_input_ids'].to(device),
            'attention_mask': batch['func_attention_mask'].to(device)
        }
        inputs = {
            'func_inputs': func_inputs,
            'line_inputs': [
                {
                    'input_ids': line['input_ids'].to(device),
                    'attention_mask': line['attention_mask'].to(device)  # attention_mask：作用于每个代码行的内部，处理“token的有效性”
                }
                for line in batch['line_inputs']
            ],
            'valid_mask': batch['valid_mask'].to(device)  # valid_mask：作用于整个代码行的级别，处理“行的有效性”
        }
        labels = {
            'func': batch['func_label'].to(device),
            'line': batch['line_labels'].to(device),
            'mask': batch['valid_mask'].to(device)
        }

        # 前向传播
        logits = model(**inputs)

        # 计算损失
        batch_total_loss = criterion(logits, labels)

        # 累积损失
        total_loss += batch_total_loss.item()

        # 反向传播
        optimizer.zero_grad()
        batch_total_loss.backward()
        optimizer.step()
        scheduler.step()

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, current_step + batch_idx)

        # 收集函数级预测结果
        func_valid_preds = torch.argmax(logits['func_logits'], dim=-1).cpu().numpy()
        func_valid_labels = labels['func'].cpu().numpy()
        func_preds.extend(func_valid_preds)
        func_labels.extend(func_valid_labels)

        # 收集行级预测结果
        batch_line_preds = torch.argmax(logits['line_logits'], dim=-1).cpu().numpy()  # [B, L]
        batch_line_probs = logits['line_probs'].cpu().detach().numpy()  # [B, L]
        batch_line_labels = labels['line'].cpu().numpy()  # [B, L]
        batch_valid_masks = labels['mask'].cpu().numpy()  # [B, L]

        for bidx in range(batch_valid_masks.shape[0]):
            line_pred = batch_line_preds[bidx]
            line_prob = batch_line_probs[bidx]
            line_label = batch_line_labels[bidx]

            # 按样本存储
            line_preds.append(line_pred)
            line_probs.append(line_prob)
            line_labels.append(line_label)

        # # 打印函数级、行级预测结果和标签
        # print("函数级预测：", func_preds)         # [batch_size]
        # print("函数级标签：", func_labels)        # [batch_size]
        # print("行级预测：", line_preds)          # [batch_size, valid_lines]，valid_lines一种有阶段，一种无截断
        # print("行级标签：", line_labels)         # [batch_size, valid_lines]
        # print("行级概率：", line_probs)          # [batch_size, valid_lines]
        # break

        # 更新进度条
        pbar.set_postfix({'loss': f"{batch_total_loss.item():.4f}"})

        # 记录每batch损失
        writer.add_scalar('Train/Batch_Total_Loss', batch_total_loss.item(), epoch * len(dataloader) + batch_idx)

        # 打印函数级、行级标签，确保数据集中存在正样本
        print("\n函数有效标签分布：", np.unique(func_valid_labels, return_counts=True))
        print("函数预测分布：", np.unique(func_valid_preds, return_counts=True))
        print_line_debug_info(batch_line_preds, batch_line_labels)

        # if batch_idx >= 10:
        #     break

    # 关闭训练进度条
    pbar.close()

    # 计算指标
    avg_total_loss = total_loss / len(dataloader)
    func_metrics = calculate_func_metrics(func_preds, func_labels)
    line_metrics = calculate_line_metrics(line_preds, line_probs, line_labels)

    # 记录TensorBoard
    writer.add_scalars(f'Epoch/Train_Fun_Metrics',
                       {'Func_F1': func_metrics['f1'],
                        'Func_Acc': func_metrics['acc'],
                        'Func_Precision': func_metrics['precision'],
                        'Func_Recall': func_metrics['recall']}, epoch)
    writer.add_scalars(f'Epoch/Train_Line_Metrics',
                       {'Line_F1': line_metrics['f1'],
                        'Line_Acc': line_metrics['acc'],
                        'Line_Precision': line_metrics['precision'],
                        'Line_Recall': line_metrics['recall'],
                        'IoU': line_metrics['IoU'],
                        'ROC-AUC': line_metrics['ROC-AUC'],
                        'PR-AUC': line_metrics['PR-AUC'],
                        'Top-5%Acc': line_metrics['Top-5%Acc'],
                        'Top-10%Acc': line_metrics['Top-10%Acc']}, epoch)

    return avg_total_loss, func_metrics, line_metrics


"""验证"""
def evaluate(model, dataloader, device, epoch, writer, mode='val'):
    model.eval()
    criterion = MultiTaskLoss(alpha=0.6)

    total_loss = 0.0
    # 初始化预测和标签收集列表
    func_preds, func_labels = [], []
    line_preds, line_probs, line_labels = [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [{mode.upper()}]", unit="batch")

        for batch_idx, batch in enumerate(pbar):
            # 转换输入到GPU
            func_inputs = {
                'input_ids': batch['func_input_ids'].to(device),
                'attention_mask': batch['func_attention_mask'].to(device)
            }
            inputs = {
                'func_inputs': func_inputs,
                'line_inputs': [
                    {
                        'input_ids': line['input_ids'].to(device),
                        'attention_mask': line['attention_mask'].to(device)
                    }
                    for line in batch['line_inputs']
                ],
                'valid_mask': batch['valid_mask'].to(device)
            }
            labels = {
                'func': batch['func_label'].to(device),
                'line': batch['line_labels'].to(device),
                'mask': batch['valid_mask'].to(device)
            }

            # 前向传播
            logits = model(**inputs)

            # 计算损失
            batch_total_loss = criterion(logits, labels)

            # 累积损失
            total_loss += batch_total_loss.item()
            pbar.set_postfix({'loss': f"{batch_total_loss.item():.4f}"})

            # 收集函数级预测结果
            func_valid_preds = torch.argmax(logits['func_logits'], dim=-1).cpu().numpy()
            func_valid_labels = labels['func'].cpu().numpy()
            func_preds.extend(func_valid_preds)
            func_labels.extend(func_valid_labels)

            # 收集行级预测结果
            batch_line_preds = torch.argmax(logits['line_logits'], dim=-1).cpu().numpy()  # [B, L]
            batch_line_probs = logits['line_probs'].cpu().detach().numpy()  # [B, L]
            batch_line_labels = labels['line'].cpu().numpy()  # [B, L]
            batch_valid_masks = labels['mask'].cpu().numpy()  # [B, L]

            for bidx in range(batch_valid_masks.shape[0]):
                line_pred = batch_line_preds[bidx]
                line_prob = batch_line_probs[bidx]
                line_label = batch_line_labels[bidx]

                # 按样本存储
                line_preds.append(line_pred)
                line_probs.append(line_prob)
                line_labels.append(line_label)

            # 打印函数级、行级标签，确保数据集中存在正样本
            print("\n函数有效标签分布：", np.unique(func_valid_labels, return_counts=True))
            print("函数预测分布：", np.unique(func_valid_preds, return_counts=True))
            print_line_debug_info(batch_line_preds, batch_line_labels)

            # if batch_idx >= 10:
            #     break

        # 关闭验证进度条
        pbar.close()

    # 计算指标
    avg_total_loss = total_loss / len(dataloader)
    func_metrics = calculate_func_metrics(func_preds, func_labels)
    line_metrics = calculate_line_metrics(line_preds, line_probs, line_labels)

    # 记录TensorBoard
    writer.add_scalars(f'Epoch/{mode}_Fun_Metrics',
                       {'Func_F1': func_metrics['f1'],
                        'Func_Acc': func_metrics['acc'],
                        'Func_Precision': func_metrics['precision'],
                        'Func_Recall': func_metrics['recall']}, epoch)
    writer.add_scalars(f'Epoch/{mode}_Line_Metrics',
                       {'Line_F1': line_metrics['f1'],
                        'Line_Acc': line_metrics['acc'],
                        'Line_Precision': line_metrics['precision'],
                        'Line_Recall': line_metrics['recall'],
                        'IoU': line_metrics['IoU'],
                        'ROC-AUC': line_metrics['ROC-AUC'],
                        'PR-AUC': line_metrics['PR-AUC'],
                        'Top-5%Acc': line_metrics['Top-5%Acc'],
                        'Top-10%Acc': line_metrics['Top-10%Acc']}, epoch)

    return avg_total_loss, func_metrics, line_metrics


"""测试模块"""
def test(model, test_loader, device, writer):
    # # 1、训练、验证、测试一体化时
    # 加载最佳模型
    # if len(top_models) > 0:
    #     best_model_path = top_models[0]["path"]  # 使用F1最高的模型
    #     print(f"加载最佳模型: {best_model_path}")
    #     model.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=True))
    #     model.to(CONFIG['device'])
    # else:
    #     print("警告: 没有找到保存的最佳模型，使用当前模型进行测试")

    # 2、单独检测已保存模型的效果时
    best_model_path = "/data/prqu/prqu_files/save_file/models_metrics_modified_CodeT5_Attention/epoch_024__f1_0.4035.pth"  # 使用F1最高的模型
    model.load_state_dict(torch.load(best_model_path, map_location='cpu', weights_only=True))
    model.to(CONFIG['device'])

    test_loss, test_func_metrics, test_line_metrics = evaluate(model, test_loader, device, 0, writer, mode='test')

    # 写入测试结果到日志文件
    write_logfile("Test_Metrics", 0, func_metrics=test_func_metrics, line_metrics=test_line_metrics, loss=test_loss)
    print_metrics('Test:', test_loss, test_func_metrics, test_line_metrics)

if __name__ == "__main__":
    # 初始化TensorBoard
    writer = SummaryWriter(CONFIG['log_dir'])
    # 初始化模型跟踪列表
    top_models = []  # 元素格式: {'f1': float, 'epoch': int, 'path': str}

    # 早停策略变量
    best_val_loss = float('inf')
    patience_epoch = 5  # 损失未减小的最大的epoch数量
    no_improve_epoch = 0  # 损失未减小的epoch数量

    train_df = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/train_balanced.csv")
    val_df = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/val_balanced.csv")

    # # 数据分布检查
    # def analyze_labels(df):
    #     total_lines = 0
    #     vuln_lines = 0
    #     for idx in tqdm(range(len(df))):
    #         labels = json.loads(df.iloc[idx]['line_labels'])
    #     total_lines += len(labels)
    #     vuln_lines += sum(labels)
    #
    #     print(f"总代码行: {total_lines}")
    #     print(f"漏洞行占比: {vuln_lines / total_lines:.4%}")
    #     print(f"负样本占比: {(total_lines - vuln_lines) / total_lines:.4%}")
    #
    # # 在main函数中调用
    # print("\n训练集标签分布:")
    # analyze_labels(train_df)
    # print("\n验证集标签分布:")
    # analyze_labels(val_df)

    # 创建数据集
    train_dataset = CodeDataset(train_df, max_lines=CONFIG['max_lines'])
    # sample = train_dataset[0]       # 检查预处理后的标签，非-100值
    # print("处理后的标签:", [x.item() for x in sample['labels'] if x != -100])

    val_dataset = CodeDataset(val_df, max_lines=CONFIG['max_lines'])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=custom_collate)
    # for batch_idx, batch in enumerate(train_loader):
    #     # 查看每个batch的数据样本
    #     print(batch)
    #     break

    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], collate_fn=custom_collate)

    # 初始化模型
    model = VulDetectionModel().to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.1)  # 1、添加L2正则化，防止过拟合；
    # 2、出现过拟合，尝试增大权重衰减：0.01-->0.1
    # 计算总训练步数（总训练批次数）和预热步数
    num_training_steps = CONFIG['epochs'] * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)  # 10%的预热
    # 创建学习率调度器——使用带预热的余弦退火策略
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # # 模型结构验证
    # print("模型结构验证:")
    # dummy_batch = next(iter(train_loader))
    # with torch.no_grad():
    #     logits = model([{k: v.to(CONFIG['device']) for k, v in line.items()} for line in dummy_batch['line_inputs']])
    #     print(f"输入维度: {dummy_batch['line_inputs'][0]['input_ids'].shape}")
    #     print(f"输出维度: {logits.shape} (应匹配 [batch, lines, 2])")

    # # 训练循环
    # for epoch in range(CONFIG['epochs']):
    #     """训练阶段"""
    #     print("\n==========开始训练阶段============\n")
    #     train_loss, train_func_metrics, train_line_metrics = train_epoch(model, train_loader, optimizer, scheduler,
    #                                                                      CONFIG['device'], epoch, writer)
    #     write_logfile("Train_Metrics", epoch, func_metrics=train_func_metrics, line_metrics=train_line_metrics,
    #                   loss=train_loss)
    #
    #     """验证阶段"""
    #     print("\n==========开始验证阶段============\n")
    #     val_loss, val_func_metrics, val_line_metrics = evaluate(model, val_loader, CONFIG['device'], epoch, writer)
    #     write_logfile("Val_Metrics", epoch, func_metrics=val_func_metrics, line_metrics=val_line_metrics,
    #                   loss=val_loss)
    #
    #     # 训练和验证损失
    #     writer.add_scalars('Loss/Train_vs_Val',
    #                        {'Train': train_loss,
    #                         'Validation': val_loss}, epoch)
    #     # 打印进度及指标信息
    #     print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
    #     print_metrics('Train:', train_loss, train_func_metrics, train_line_metrics)
    #     print_metrics('Val:', val_loss, val_func_metrics, val_line_metrics)
    #
    #     # # 若验证损失在多个连续的 epoch 不见减少，则实施早停
    #     # if val_loss < best_val_loss:
    #     #     best_val_loss = val_loss
    #     #     no_improve_epoch = 0
    #     # else:
    #     #     no_improve_epoch += 1
    #     #     if no_improve_epoch >= patience_epoch:
    #     #         print("Early stopping!")
    #     #         break
    #
    #     # 保存当前模型到跟踪列表（如果符合条件）
    #     save_top_models(val_line_metrics['f1'], epoch, model)
    #
    #     # 打印当前top模型信息
    #     print(f"\n当前Top模型列表：")
    #     for i, m in enumerate(top_models, 1):
    #         print(f"第{i}名 | Epoch {m['epoch']} | F1: {m['f1']:.4f}")

    # 训练完成后，添加测试阶段
    print("\n" + "=" * 60)
    print("训练完成! 开始测试阶段...")
    print("=" * 60)

    test_df = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/test_balanced.csv")
    test_dataset = CodeDataset(test_df, max_lines=CONFIG['max_lines'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], collate_fn=custom_collate)

    test(model, test_loader, CONFIG['device'], writer)

    writer.close()
