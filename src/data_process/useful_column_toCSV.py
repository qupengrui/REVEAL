import pandas as pd
from difflib import SequenceMatcher
import csv
import json
import os

# # 2、测试写入新文件的数据
# train = pd.read_csv(
#          "/home/pod/shared-nvme/prqu_files/big-vul_dataset/data/train_balance.csv",
#          encoding='utf-8')
# print(train['vul'].value_counts())
# val = pd.read_csv(
#          "/home/pod/shared-nvme/prqu_files/big-vul_dataset/data/val.csv",
#          encoding='utf-8')
# print(val['vul'].value_counts())
# test = pd.read_csv(
#          "/data/prqu/prqu_files/data/Most_Information_Dataset/train_balanced.csv",
#          encoding='utf-8')
# print(test['fun_label'].value_counts())


# 3、函数+标签、行代码+标签保存在同一文件
# 读取原始数据集
df = pd.read_csv("/data/prqu/prqu_files/Bigvul_dataset/test.csv")

# 自动识别CWE列名（兼容不同数据集命名）
cwe_column = None
for candidate in ['CWE_id', 'CWE_ID', 'cwe_id', 'CWE', 'cwe', 'CWE ID', 'cwe id']:
    if candidate in df.columns:
        cwe_column = candidate
        break

if cwe_column is None:
    for col in df.columns:
        if 'cwe' in str(col).lower():
            cwe_column = col
            break

if cwe_column is None:
    print("警告: 未找到CWE列，输出中的CWE_id将为空")
else:
    print(f"找到CWE列: {cwe_column}")

new_rows = []

for idx, row in df.iterrows():
    # 提取基础信息（假设原始数据使用'index'作为函数ID，若实际情况不同请修改）
    func_id = idx  # 若原始数据有专门ID列，改为 row['your_id_column']
    func_before = row['func_before']
    func_after = row['func_after']  # 新增：修复后的函数代码
    vul = row['target']
    cwe_id = row[cwe_column] if cwe_column is not None else ""
    flaw_line = row['flaw_line']  # 新增：漏洞行内容
    flaw_line_index = row['flaw_line_index']  # 新增：漏洞行位置

    # 分割函数代码为行列表
    code_lines = [line.rstrip('\n') for line in func_before.split('\n')]  # 保留原始换行符

    # 初始化行标签（默认全0）
    line_labels = [0] * len(code_lines)

    # 处理漏洞行（若存在漏洞）
    if pd.notna(row['flaw_line_index']) and row['flaw_line_index'].strip():
        for line_str in row['flaw_line_index'].split(','):
            line_str = line_str.strip()
            if line_str:
                try:
                    # 将1-based行号转为0-based索引
                    line_num = int(line_str)
                    if 0 <= line_num < len(code_lines):
                        line_labels[line_num] = 1
                except ValueError:
                    pass  # 忽略无效行号格式

    # 构建新数据行
    new_rows.append({
        "func_id": func_id,
        "func_after": func_after,
        "func_before": func_before,
        "fun_label": vul,
        "CWE_id": cwe_id,
        "flaw_line": flaw_line,
        "flaw_line_index": flaw_line_index,
        "lines": json.dumps(code_lines, ensure_ascii=False),
        "line_labels": json.dumps(line_labels, ensure_ascii=False)
    })

# 创建新DataFrame并保持原始顺序
new_df = pd.DataFrame(new_rows)

# 保存结果到CSV（确保列顺序正确）
new_df[["func_id", "CWE_id", "func_after", "func_before", "fun_label", "flaw_line", "flaw_line_index", "lines", "line_labels"]].to_csv(
    "/data/prqu/prqu_files/data/Most_Information_Dataset/test_with_cwe.csv", index=False, encoding='utf-8'
)

# 检验提取数量
df_Initial_dataset = pd.read_csv("/data/prqu/prqu_files/Bigvul_dataset/test.csv", encoding='utf-8')
df_processed = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/test_with_cwe.csv", encoding='utf-8')
print(df_Initial_dataset['target'].value_counts())
print(df_processed['fun_label'].value_counts())

# 4、对训练样本进行下采样，漏洞非漏洞比例为1:1
# 加载数据集
df = pd.read_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/test_with_cwe.csv")

# 筛选出漏洞样本 (fun_label = 1) 和非漏洞样本 (fun_label = 0)
train_vul = df[df['fun_label'] == 1]
train_non_vul = df[df['fun_label'] == 0]

# 随机抽取与漏洞样本数量相等的非漏洞样本
train_non_vul_sampled = train_non_vul.sample(n=len(train_vul), random_state=42)

# 合并漏洞样本和抽样的非漏洞样本
train_balanced = pd.concat([train_vul, train_non_vul_sampled])

# 打乱数据集顺序
train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存新的平衡数据集为CSV文件
train_balanced.to_csv("/data/prqu/prqu_files/data/Most_Information_Dataset/test_with_cwe_balanced.csv", index=False)

# 输出平衡后的标签分布，确保1:1的比例
print(train_balanced['fun_label'].value_counts())



# # 6、随机抽取五个样本
# input_csv = r"D:\Python_Line-level_Vulnerability-detection\data\Most_Information_Dataset\test_with_cwe_balanced.csv"
# output_csv = r"D:\Python_Line-level_Vulnerability-detection\data\test_10samples\test_5samples.csv"

# df_balanced = pd.read_csv(input_csv, encoding='utf-8')
# sample_n = min(5, len(df_balanced))

# # 固定随机种子，保证可复现
# df_5samples = df_balanced.sample(n=sample_n, random_state=42).reset_index(drop=True)

# os.makedirs(os.path.dirname(output_csv), exist_ok=True)
# df_5samples.to_csv(output_csv, index=False, encoding='utf-8')

# print(f"已随机抽取 {sample_n} 个样本并保存到: {output_csv}")














