import csv
import json
import pandas as pd
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import ast  # 用于解析列表字符串

# 1、查看漏洞行的存储类型，进行行定位
# df = pd.read_csv("/data/prqu/prqu_files/Bigvul_dataset/test.csv", encoding='utf-8')
# data = df.iloc[142, 38]
# print(type(data))
# print(data)
# print(data[0])


# # 2、测试数据集用json存储的列信息，以及数据分布检查
# def analyze_labels(df):
#     total_lines = 0
#     vuln_lines = 0
#     for idx in tqdm(range(len(df))):
#         line_code = json.loads(df.iloc[idx]['lines'])
#         print("\nLine_Code:", line_code)
#         print("\n代码行数：", len(line_code))
#         labels = json.loads(df.iloc[idx]['line_labels'])
#         print("labels:", labels)
#         print("\nlabels长度：", len(labels))
#         total_lines += len(labels)
#         vuln_lines += sum(labels)
#
#     print(f"总代码行: {total_lines}")
#     print(f"漏洞行占比: {vuln_lines / total_lines:.4%}")
#     print(f"负样本占比: {(total_lines - vuln_lines) / total_lines:.4%}")
#
# print("\n训练集标签分布:")
# train_df = pd.read_csv("/root/shared-nvme/prqu_files/new/Fun_Lines_dataset/train_balanced.csv")
# analyze_labels(train_df)


# # 3、加载预训练的 CodeBert 模型和 tokenizer，打印每行代码的[CLS]向量
# model_name = "/data/prqu/prqu_files/CodeBert"
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaModel.from_pretrained(model_name)
#
# # 设置模型为评估模式（不进行训练）
# model.eval()
#
#
# def extract_cls_vector(code_lines):
#     """
#     提取每行代码的 [CLS] 向量
#     :param code_lines: 代码行的列表
#     :return: 每行代码的 [CLS] 向量组成的列表
#     """
#     cls_vectors = []
#
#     for line in code_lines:
#         # 对代码行进行编码（将其转化为token id）
#         inputs = tokenizer(line, return_tensors='pt', padding=True, truncation=True, max_length=512)
#
#         # 将输入传递给模型并获取输出
#         with torch.no_grad():  # 不计算梯度
#             outputs = model(**inputs)
#
#         # 获取 [CLS] 向量，它是输出的第一个token的向量
#         cls_vector = outputs.last_hidden_state[0, 0].numpy()  # [batch_size, seq_len, hidden_size]
#
#         # 将 [CLS] 向量加入到结果列表中
#         cls_vectors.append(cls_vector)
#
#     return cls_vectors
#
# # 示例：提取几行代码的 [CLS] 向量
# code_lines = [
#     'def add(a, b): return a + b',
#     'def subtract(a, b): return a - b',
#     'def multiply(a, b): return a * b'
# ]
# cls_vectors = extract_cls_vector(code_lines)
# # 打印提取的 [CLS] 向量
# for i, vector in enumerate(cls_vectors):
#     print(f"Code line {i + 1} CLS vector:", vector)


# # 4、加载查看CodeBert的模型配置
# bert_config = RobertaConfig.from_pretrained("/data/prqu/prqu_files/CodeBert")
# print(bert_config)      # 查看配置内容

# 5、查看数据集漏洞非漏洞个数
df = pd.read_csv('/data/prqu/prqu_files/new/Fun_Lines_dataset/val_balanced.csv')
count = df['fun_label'].value_counts()
print("漏洞个数：", count.get(1, 0))  # 1代表漏洞
print("非漏洞个数：", count.get(0, 0))  # 0代表非漏洞

# # 6、统计数据集中样本的平均长度
# df = pd.read_csv("/data/prqu/prqu_files/new/Fun_Lines_dataset/train_balanced.csv")
#
# # 初始化变量，用于计算总行数和总字符数
# total_lines = 0
# total_chars = 0
#
# # 遍历每一行数据
# for index, row in df.iterrows():
#     # 获取代码样本
#     code_sample = row['func_before']
#
#     # 获取行标签（可能是列表形式的字符串）
#     line_labels = ast.literal_eval(row['line_labels'])  # 用 literal_eval 安全地转换为列表
#
#     # 按行分割代码样本
#     code_lines = code_sample.split('\n')
#
#     # 统计当前代码样本的行数和字符数
#     total_lines += len(code_lines)
#     total_chars += sum(len(line) for line in code_lines)
#
# # 计算平均行长度
# average_line_length = total_chars / total_lines if total_lines > 0 else 0
#
# # 输出结果
# print(f"平均行长度：{average_line_length:.2f}")
