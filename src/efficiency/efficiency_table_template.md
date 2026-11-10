# REVEAL 效率与消耗指标 —— 结果表格模板

> 用途：回应审稿意见 **R1-6**（模型规模、训练成本、推理时间、逐行独立编码的开销）与 **R2-7**（参数量、训练时间、推理成本、内存消耗）。
>
> 数字来源：运行 `src/efficiency/measure_efficiency.py`（自动生成 `efficiency_table.md`），
> 以及训练时由训练脚本自动写入的 `training_cost_*.tsv`。
>
> 下表 `【待填】` 处粘贴实测值。测量环境（GPU 型号/显存、CUDA、PyTorch 版本、batch size）务必一并写明，
> 否则不同机器的数字不可比。

---

## 表 A：模型规模（Model Size / Complexity）

| 指标 | 数值 | 单位 | 说明 |
|---|---|---|---|
| 总参数量 | 【待填】 | params | 含已实例化但前向未调用的 decoder |
| 可训练参数量 | 【待填】 | params | 仅函数级编码器后 6 层 + 新增模块 |
| 冻结参数量 | 【待填】 | params | 行级编码器全部冻结 |
| 实际参与前向的参数量 | 【待填】 | params | forward hook 实测，剔除未调用的 decoder |
| 参数量（FP32） | 【待填】 | MB | 参数量 × 4 bytes |
| checkpoint 文件大小 | 【待填】 | MB | 磁盘占用 |
| 单函数前向 FLOPs | 【待填】 | FLOPs | batch = 【待填】 |

**子模块参数分布**（`measure_efficiency.py` 会逐项输出）

| 子模块 | 参数量 | 可训练 |
|---|---|---|
| codeT51（行级编码器，冻结） | 【待填】 | 0 |
| codeT52（函数级编码器，部分解冻） | 【待填】 | 【待填】 |
| dim_adapter | 【待填】 | 【待填】 |
| position_embedding | 【待填】 | 【待填】 |
| line_transformer | 【待填】 | 【待填】 |
| line_semantic_fusion | 【待填】 | 【待填】 |
| func_classifier | 【待填】 | 【待填】 |
| line_classifier | 【待填】 | 【待填】 |

---

## 表 B：训练成本（Training Cost）

| 指标 | 数值 | 单位 | 说明 |
|---|---|---|---|
| 训练集样本数 | 【待填】 | samples | |
| batch size | 16 | — | 与论文设置一致 |
| 每轮训练步数 | 【待填】 | steps | |
| 单步训练耗时 | 【待填】 | ms/step | 含前向 + 反向 + 优化器 |
| 单轮训练耗时 | 【待填】 | min/epoch | |
| 总训练时长 | 【待填】 | GPU-hours | epochs = 【待填】，外推值 |
| 训练吞吐 | 【待填】 | samples/s | |
| 训练峰值显存 | 【待填】 | MB | `max_memory_allocated` |
| 训练峰值显存（含缓存） | 【待填】 | MB | `max_memory_reserved` |

---

## 表 C：推理成本（Inference Cost）

| batch size | 单函数延迟（中位） | 单函数延迟 p95 | 吞吐 | 推理峰值显存 |
|---|---|---|---|---|
| 1 | 【待填】 ms | 【待填】 ms | 【待填】 函数/s | 【待填】 MB |
| 8 | 【待填】 ms | 【待填】 ms | 【待填】 函数/s | 【待填】 MB |
| 16 | 【待填】 ms | 【待填】 ms | 【待填】 函数/s | 【待填】 MB |

---

## 表 D：逐行独立编码的计算开销（审稿人重点关注）

这是 R1-6 中 "the computational overhead of independently encoding each code line" 的直接回答。

| 指标 | 数值 | 单位 | 说明 |
|---|---|---|---|
| 逐行独立编码耗时 | 【待填】 | ms | 当前实现：`max_lines` 行逐行各调用一次编码器 |
| 全部行合并为单次编码耗时 | 【待填】 | ms | 把所有行拼成一个 batch，只调用一次编码器 |
| 独立编码相对合并编码的倍数 | 【待填】 | × | 越大说明逐行编码开销越显著 |
| 行编码占单次前向总耗时的比例 | 【待填】 | % | 见表 E |

---

## 表 E：单次前向的模块耗时拆分

| 模块 | 耗时 | 占比 |
|---|---|---|
| 行编码器 codeT51.encoder | 【待填】 ms | 【待填】 % |
| 函数编码器 codeT52.encoder | 【待填】 ms | 【待填】 % |
| 行间建模 line_transformer | 【待填】 ms | 【待填】 % |
| 行级融合 line_semantic_fusion | 【待填】 ms | 【待填】 % |
| 维度适配 dim_adapter | 【待填】 ms | 【待填】 % |
| 位置编码 position_embedding | 【待填】 ms | 【待填】 % |
| 函数分类头 func_classifier | 【待填】 ms | 【待填】 % |
| 行级分类头 line_classifier | 【待填】 ms | 【待填】 % |

---

## 测量环境（必须填写，否则数字不可复现）

| 项 | 值 |
|---|---|
| GPU 型号 | 【待填】 |
| GPU 显存 | 【待填】 GB |
| CUDA / PyTorch 版本 | 【待填】 |
| 预热次数 / 计时重复次数 | 【待填】 |
| 计时方式 | `torch.cuda.synchronize()` 前后取 `perf_counter` 差值，取中位数 |

---

## 撰写回复时的两点提醒

1. **逐行编码的开销要如实说明，并给出缓解措施。**
   REVEAL 确实对每一行独立调用一次编码器（`max_lines` 次），这是真实开销。
   表 D 的"倍数"列就是量化这个开销的关键证据。建议在回复中同时说明：
   该开销换来的是行级定位能力，而函数级单次编码的基线（表 C 对比）虽然更快但不提供行级输出。
   若把表 D 的"合并编码"实现出来，可以同时报告"优化后"的延迟，这会显著增强说服力。

2. **参数量要区分"总参数"与"实际使用参数"。**
   模型实例化了两个 `T5ForConditionalGeneration`（每个都自带 decoder），
   但前向中只调用 `.encoder`，decoder 参数属于"已实例化但未使用"。
   只看总参数量会高估模型规模，只看使用参数量又与实际显存占用不符。
   建议两个数都报，并注明差异原因 —— 这也能体现分析严谨性。
