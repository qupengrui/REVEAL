import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup    # 学习率调度器，使用带预热的余弦退火策略
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.nn as nn

"""1、测试CodeBERT功能"""
# CodeBert_model_path = "/data/prqu/prqu_files/CodeBert"
#
# # 补充部分：测试CodeBert模型是否加载成功
# # 1、检查 Tokenizer：
# tokenizer = RobertaTokenizer.from_pretrained(CodeBert_model_path)
# # 验证 Tokenizer 是否成功加载
# print("Tokenizer loaded successfully!")
# # 测试分词
# test_code = """
# static char* get_rootdir(pid_t pid)
# {
#     char buf[sizeof("/proc/%lu/root") + sizeof(long)*3];
#     sprintf(buf, "/proc/%lu/root", (long)pid);
#     return malloc_readlink(buf);
# }
# """
# tokens = tokenizer.tokenize(test_code)
# print("Token:", tokens)
# # # 测试编码
# # encoded = tokenizer.encode(test_code, add_special_tokens=True)
# # print("Encoded IDs:", encoded)
# # # 测试解码
# # decoded = tokenizer.decode(encoded)
# # print("Decoded IDs:", decoded)
# # # # 2、检查Model：
# # # tokenizer = RobertaTokenizer.from_pretrained(CodeBert_model_path)
# # # model = RobertaModel.from_pretrained(CodeBert_model_path)
# # # # 验证 model 是否加载成功
# # # print("Model loaded successfully!")
# # # # 测试前向传播
# # # test_code = "def hello_world():\n    print('Hello, World!')"
# # # inputs = tokenizer(test_code, return_tensors="pt", padding=True, truncation=True)
# # # outputs = model(**inputs)
# # # # 输出模型的结果
# # # print("Model output shape:", outputs.last_hidden_state.shape)
# # # # 3、查看模型配置：
# # # tokenizer = RobertaTokenizer.from_pretrained(CodeBert_model_path)
# # # model = RobertaModel.from_pretrained(CodeBert_model_path)
# # # # 查看 tokenizer 和 model 的配置信息
# # # print("Tokenizer vocab size:", tokenizer.vocab_size)
# # # print("Model configuration:", model.config)
# # # # 4、测试推理能力：
# # # # 加载 CodeBert 模型用于代码分类
# # # tokenizer = RobertaTokenizer.from_pretrained(CodeBert_model_path)
# # # model = RobertaForSequenceClassification.from_pretrained(CodeBert_model_path, num_labels=2)
# # # # 测试输入
# # # test_code = "def add(a, b): return a + b"
# # # inputs = tokenizer(test_code, return_tensors="pt", padding=True, truncation=True)
# # # # 推理
# # # outputs = model(**inputs)
# # # logits = outputs.logits
# # # # 检查推理结果
# # # print("Logits:", logits)
# # # predicted_class = torch.argmax(logits, dim=1).item()
# # # print("Predicted class:", predicted_class)
#
#
# # a = [[1,2,3],
# #      [1,2,3,4,5,6]]
# # print(np.array(a).flatten())


"""2、测试 CodeT5 的功能"""
def test_tokenizer(tokenizer):
    # 测试文本
    test_code = "def calculate_sum(a, b):\n    return a + b  # 加法运算"

    # 1. 基本分词测试
    tokens = tokenizer.tokenize(test_code)
    print("\n1. 分词结果:")
    print(tokens)

    # 2. 编码/解码测试
    encoded = tokenizer.encode(test_code, return_tensors="pt")
    decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
    print("\n2. 编码->解码一致性:")
    print(f"原始代码: {test_code}")
    print("编码IDs:", encoded)
    print(f"解码结果: {decoded}")
    print(f"一致性检查: {'通过' if test_code == decoded else '失败'}")

    # 3. 特殊标记处理
    print("\n3. 特殊标记处理:")
    print(f"填充标记: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"未知标记: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"掩码标记: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    print(f"额外ID标记: <extra_id_0> (ID: {tokenizer.convert_tokens_to_ids('<extra_id_0>')})")


def test_model(model):
    # 1. 模型配置信息
    config = model.config
    print("1. 模型配置:")
    print(f"模型类型: {config.model_type}")
    print(f"词汇表大小: {config.vocab_size}")
    print(f"隐藏层大小: {config.d_model}")
    print(f"注意力头数: {config.num_heads}")
    print(f"前馈网络维度: {config.d_ff}")
    print(f"解码器层数: {config.num_layers}")

    # 2. 模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n2. 模型参数:")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 3. 设备信息
    device = next(model.parameters()).device
    print(f"\n3. 模型所在设备: {device}")


def test_inference(tokenizer, model):
    """测试模型推理能力"""
    print("\n" + "=" * 50)
    print("推理能力测试")
    print("=" * 50)

    # 1. 代码填充测试
    print("1. 代码填充测试:")
    text = "def greet(user): print(f'hello <extra_id_0>!')"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=20)
    print(f"输入: {text}")
    print(f"输出: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")

    # 2. 代码补全测试
    print("\n2. 代码补全测试:")
    text = "def factorial(n):"
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=50)
    print(f"输入: {text}")
    print(f"补全结果: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")

    # 3. 代码翻译测试
    print("\n3. 代码翻译测试 (Python 到 JavaScript):")
    text = "def add(a, b): return a + b"
    input_ids = tokenizer(f"Translate Python to JavaScript: {text}", return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=30)
    print(f"输入: {text}")
    print(f"翻译结果: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")

    # 4. 代码注释生成
    print("\n4. 代码注释生成:")
    text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)"
    input_ids = tokenizer(f"Generate comment for this code: {text}", return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=100)
    print(f"输入: {text}")
    print(f"注释结果: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")

    # 5. 错误代码修复
    print("\n5. 错误代码修复:")
    text = "def divide(a, b):\n    result = a / b\n    return results"  # 故意拼写错误
    input_ids = tokenizer(f"Fix the bug in this code: {text}", return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=50)
    print(f"输入: {text}")
    print(f"修复结果: {tokenizer.decode(generated_ids[0], skip_special_tokens=True)}")


def main():
    # 初始化模型和分词器
    tokenizer = RobertaTokenizer.from_pretrained("/data/prqu/prqu_files/CodeT5")
    model = T5ForConditionalGeneration.from_pretrained("/data/prqu/prqu_files/CodeT5")

    # 1、测试分词
    # test_tokenizer(tokenizer)

    # 2、打印模型参数以及配置信息
    # test_model(model)
    # 3、测试模型推理能力
    # test_inference(tokenizer, model)


if __name__ == "__main__":
    main()