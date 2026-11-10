import argparse
import gc
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import defaultdict

import pandas as pd
import torch

"""REVEAL 效率/消耗指标测量脚本。

针对两位审稿人的意见：
  R1-6: 需报告模型规模、训练成本、推理时间，以及"逐行独立编码"带来的计算开销
  R2-7: 需报告参数量、训练时间、推理成本、内存消耗

本脚本复用训练脚本中的模型定义（不重复实现，避免与训练代码不一致），测量：
  1) 模型规模   : 参数量(总/可训练/冻结)、各子模块参数、参数量MB、checkpoint体积
  2) 计算量     : 单样本 FLOPs、前向-used 参数量(实际参与前向的参数)
  3) 推理成本   : 不同 batch 下的单函数延迟(p50/p95)、吞吐、峰值显存
  4) 逐行编码开销: "逐行独立编码" vs "全部行合并为一次编码" 的耗时对比  ← 审稿人重点关注
  5) 模块耗时拆分: 行编码器 / 函数编码器 / 行间Transformer / 融合层 / 分类头
  6) 训练成本   : 单步耗时、外推单轮与总训练时长、GPU小时、峰值显存、吞吐

用法（在服务器上运行）:
    python measure_efficiency.py
    python measure_efficiency.py --checkpoint /path/to/epoch_009__f1_0.xxxx.pth
    python measure_efficiency.py --test_csv  /data/prqu/prqu_files/data/Most_Information_UnbalancedDataset/test.csv \
                                 --train_csv /data/prqu/prqu_files/data/Most_Information_UnbalancedDataset/train.csv

输出（--out_dir 下）:
    efficiency_results.csv    长表，含 category/metric/value/unit/note
    efficiency_table.md       可直接粘进论文的 Markdown 表格
"""

DEFAULT_MODEL_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'Fun_Line_VD_Trans+Pos_dataset_split_With_Project.py')


# ----------------------------------------------------------------------------
# 载入训练脚本（按路径动态加载，文件名含 '+' 无法用常规 import）
# ----------------------------------------------------------------------------
def load_training_module(script_path, module_name='reveal_train_module'):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"未找到训练脚本: {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# 输入构造：复用训练脚本自带的 CodeDataset / custom_collate
# ----------------------------------------------------------------------------
def build_synthetic_df(n, max_lines):
    """构造与真实样本同形状的合成数据。

    由于 func/line 均被填充到固定长度（max_func_length / max_line_length），
    合成数据与真实数据的前向计算量一致，适合测耗时与显存。
    """
    rows = []
    body = ['void f(int a) {', '  int x = a;', '  if (x) { x++; }', '  return;', '}']
    for i in range(n):
        lines = (body * 20)[:max_lines]
        rows.append({
            'func_before': '\n'.join(lines),
            'fun_label': 0,
            'lines': json.dumps(lines),
            'line_labels': json.dumps([0] * max_lines),
        })
    return pd.DataFrame(rows)


def build_batch(mod, df, max_lines, max_func_length, max_line_length):
    ds = mod.CodeDataset(
        df, max_lines=max_lines,
        max_func_length=max_func_length, max_line_length=max_line_length)
    samples = [ds[i] for i in range(len(ds))]
    return mod.custom_collate(samples)


def to_inputs(batch, device):
    """与训练脚本 train_epoch 中的 inputs 构造保持一致。"""
    return {
        'func_inputs': {
            'input_ids': batch['func_input_ids'].to(device),
            'attention_mask': batch['func_attention_mask'].to(device),
        },
        'line_inputs': [
            {'input_ids': l['input_ids'].to(device),
             'attention_mask': l['attention_mask'].to(device)}
            for l in batch['line_inputs']
        ],
        'valid_mask': batch['valid_mask'].to(device),
    }


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# 1) 模型规模
# ----------------------------------------------------------------------------
def count_params(module):
    return sum(p.numel() for p in module.parameters())


def measure_complexity(model):
    total = count_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    per_module = {}
    for name, child in model.named_children():
        per_module[name] = {
            'params': count_params(child),
            'trainable': sum(p.numel() for p in child.parameters() if p.requires_grad),
        }

    # T5ForConditionalGeneration 会同时实例化 encoder 和 decoder，
    # 但本项目只调用 .encoder —— 因此 decoder 参数属于"已实例化但未使用"。
    unused_decoder = 0
    for enc_name in ('codeT51', 'codeT52'):
        sub = getattr(model, enc_name, None)
        if sub is not None and hasattr(sub, 'decoder'):
            unused_decoder += count_params(sub.decoder)

    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': total - trainable,
        'param_memory_mb': total * 4 / 1024 ** 2,          # fp32
        'unused_decoder_params': unused_decoder,
        'per_module': per_module,
    }


def measure_forward_used_params(model, forward_fn):
    """统计真正参与前向计算的参数量。

    方法：对每个子模块挂 forward hook，跑一次前向后收集被触发过的模块，
    再累加这些模块自身持有的参数（recurse=False，避免重复计数）。
    """
    executed = set()
    handles = [m.register_forward_hook(lambda mod, i, o: executed.add(id(mod)))
               for m in model.modules()]
    try:
        forward_fn()
    finally:
        for h in handles:
            h.remove()

    used = 0
    for m in model.modules():
        if id(m) in executed:
            used += sum(p.numel() for p in m.parameters(recurse=False))
    return used


def measure_flops(model, inputs, device):
    """用 torch>=2.1 的 FlopCounterMode 统计单次前向 FLOPs。"""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    try:
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            model(**inputs)
            sync(device)
        return counter.get_total_flops()
    except Exception as e:                                   # noqa: BLE001
        print(f"  警告: FLOPs 统计失败({e})，已跳过")
        return None


# ----------------------------------------------------------------------------
# 2) 推理耗时 / 显存
# ----------------------------------------------------------------------------
def time_forward(model, inputs, device, warmup, iters):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)
        sync(device)
        times = []
        for _ in range(iters):
            sync(device)
            t0 = time.perf_counter()
            model(**inputs)
            sync(device)
            times.append(time.perf_counter() - t0)
    return times


def measure_inference_memory(model, inputs, device):
    if device.type != 'cuda':
        return None, None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(**inputs)
        sync(device)
    return (torch.cuda.max_memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_reserved() / 1024 ** 2)


# ----------------------------------------------------------------------------
# 3) 逐行独立编码 vs 合并编码 —— 直接回应审稿人关注的开销
# ----------------------------------------------------------------------------
def time_line_encoding(model, inputs, device, warmup, iters):
    """对比两种行编码方式的耗时。

    independent: 当前实现 —— 对 max_lines 行逐行调用一次编码器
    batched    : 把所有行拼成一个 batch，只调用一次编码器
    """
    line_inputs = inputs['line_inputs']
    encoder = model.codeT51.encoder

    ids = torch.cat([l['input_ids'] for l in line_inputs], dim=0)
    mask = torch.cat([l['attention_mask'] for l in line_inputs], dim=0)

    def run_independent():
        for l in line_inputs:
            encoder(input_ids=l['input_ids'], attention_mask=l['attention_mask'])

    def run_batched():
        encoder(input_ids=ids, attention_mask=mask)

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            run_independent()
            run_batched()
        sync(device)

        def timed(fn):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            return time.perf_counter() - t0

        ind = [timed(run_independent) for _ in range(iters)]
        bat = [timed(run_batched) for _ in range(iters)]

    return {
        'independent_ms': statistics.median(ind) * 1000,
        'batched_ms': statistics.median(bat) * 1000,
        'num_lines': len(line_inputs),
    }


# ----------------------------------------------------------------------------
# 4) 模块耗时拆分
# ----------------------------------------------------------------------------
def time_submodules(model, inputs, device, iters):
    """按顶层子模块统计前向耗时占比。

    注意：模型内部是 self.codeT51.encoder(...) 而非 self.codeT51(...)，
    所以必须 hook encoder 子模块本身，hook codeT51 不会触发。
    """
    targets = [
        ('行编码器 codeT51.encoder', model.codeT51.encoder),
        ('函数编码器 codeT52.encoder', model.codeT52.encoder),
        ('维度适配 dim_adapter', model.dim_adapter),
        ('位置编码 position_embedding', model.position_embedding),
        ('行间建模 line_transformer', model.line_transformer),
        ('行级融合 line_semantic_fusion', model.line_semantic_fusion),
        ('函数分类头 func_classifier', model.func_classifier),
        ('行级分类头 line_classifier', model.line_classifier),
    ]

    accum = defaultdict(float)
    stack = defaultdict(list)

    def make_hooks(label):
        def pre(mod, inp):
            sync(device)
            stack[label].append(time.perf_counter())

        def post(mod, inp, out):
            sync(device)
            accum[label] += time.perf_counter() - stack[label].pop()

        return pre, post

    handles = []
    for label, module in targets:
        pre, post = make_hooks(label)
        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    model.eval()
    try:
        with torch.no_grad():
            model(**inputs)                      # warmup
            for d in accum:
                accum[d] = 0.0
            for _ in range(iters):
                model(**inputs)
    finally:
        for h in handles:
            h.remove()

    total = sum(accum.values())
    return [{'module': k, 'ms': v / iters * 1000,
             'share': (v / total * 100) if total > 0 else 0.0}
            for k, v in accum.items()]


# ----------------------------------------------------------------------------
# 5) 训练成本
# ----------------------------------------------------------------------------
def measure_training_step(mod, model, batch, device, steps, out):
    """测量若干训练步的耗时与峰值显存。"""
    criterion = mod.MultiTaskLoss(
        alpha=mod.CONFIG['alpha'],
        learnable_alpha=mod.CONFIG['learnable_alpha']).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=mod.CONFIG['lr'], weight_decay=0.1)

    inputs = to_inputs(batch, device)
    labels = {
        'func': batch['func_label'].to(device),
        'line': batch['line_labels'].to(device),
        'mask': batch['valid_mask'].to(device),
    }
    bs = inputs['func_inputs']['input_ids'].size(0)

    model.train()
    criterion.train()

    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 预热（含一次反向，触发 cuDNN 算法选择）
    for _ in range(2):
        optimizer.zero_grad()
        logits = model(**inputs)
        loss, _, _, _ = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    sync(device)

    times = []
    for _ in range(steps):
        sync(device)
        t0 = time.perf_counter()
        optimizer.zero_grad()
        logits = model(**inputs)
        loss, _, _, _ = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        sync(device)
        times.append(time.perf_counter() - t0)

    step_time = statistics.median(times)
    steps_per_epoch = out.get('steps_per_epoch')

    result = {
        'ms_per_step': step_time * 1000,
        'samples_per_sec': bs / step_time,
        'batch_size': bs,
    }
    if steps_per_epoch:
        result['sec_per_epoch'] = step_time * steps_per_epoch
        result['total_train_hours'] = step_time * steps_per_epoch * mod.CONFIG['epochs'] / 3600
    if device.type == 'cuda':
        result['peak_alloc_mb'] = torch.cuda.max_memory_allocated() / 1024 ** 2
        result['peak_reserved_mb'] = torch.cuda.max_memory_reserved() / 1024 ** 2
    return result


# ----------------------------------------------------------------------------
# 输出
# ----------------------------------------------------------------------------
def add(rows, category, metric, value, unit='', note=''):
    rows.append({'category': category, 'metric': metric,
                 'value': value, 'unit': unit, 'note': note})


def fmt(v):
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        return f'{v:,.4f}' if abs(v) < 1 else f'{v:,.2f}'
    return f'{v:,}' if isinstance(v, int) else str(v)


def write_markdown(rows, path, args):
    lines = ['# REVEAL 效率与消耗指标', '']
    lines.append(f'- 模型脚本: `{os.path.basename(args.model_script)}`')
    lines.append(f'- checkpoint: `{args.checkpoint or "未加载（使用随机初始化权重，规模/耗时不受影响）"}`')
    lines.append(f'- batch_size: {args.batch_sizes}，warmup={args.warmup}，iters={args.iters}，'
                 f'train_steps={args.train_steps}')
    lines.append(f'- 设备: {args.device}')
    lines.append('')
    lines.append('| 类别 | 指标 | 数值 | 单位 | 说明 |')
    lines.append('|------|------|------|------|------|')
    for r in rows:
        lines.append(f"| {r['category']} | {r['metric']} | {fmt(r['value'])} | "
                     f"{r['unit']} | {r['note']} |")
    lines.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='测量 REVEAL 的效率与消耗指标')
    p.add_argument('--model_script', default=DEFAULT_MODEL_SCRIPT,
                   help='训练脚本路径（用于复用模型定义）')
    p.add_argument('--checkpoint', default=None,
                   help='已训练模型 .pth 路径；不提供则用随机初始化权重（不影响规模/耗时测量）')
    p.add_argument('--test_csv', default=None,
                   help='真实测试集 CSV；不提供则用同形状合成数据')
    p.add_argument('--train_csv', default=None,
                   help='真实训练集 CSV；用于估算每轮步数。不提供则仅报告单步耗时')
    p.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 8, 16],
                   help='推理测量的 batch size 列表 (默认: 1 8 16)')
    p.add_argument('--warmup', type=int, default=5, help='预热次数 (默认: 5)')
    p.add_argument('--iters', type=int, default=20, help='计时重复次数 (默认: 20)')
    p.add_argument('--train_steps', type=int, default=20, help='训练步基准的步数 (默认: 20)')
    p.add_argument('--device', default=None, help='覆盖设备，如 cuda:0 / cpu')
    p.add_argument('--out_dir',
                   default='/data/prqu/prqu_files/save_file/efficiency_analysis',
                   help='输出目录')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('载入训练脚本（复用模型定义）...')
    mod = load_training_module(args.model_script)
    cfg = mod.CONFIG

    device = torch.device(args.device) if args.device else cfg['device']
    print(f'设备: {device}')

    print('构建模型...')
    model = mod.VulDetectionModel()
    if args.checkpoint:
        print(f'加载 checkpoint: {args.checkpoint}')
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.to(device)
    model.eval()

    rows = []

    # ---- 1) 模型规模 ----
    print('测量模型规模...')
    cplx = measure_complexity(model)
    add(rows, '模型规模', '总参数量', cplx['total_params'], 'params', '含已实例化但未使用的 decoder')
    add(rows, '模型规模', '可训练参数量', cplx['trainable_params'], 'params', '')
    add(rows, '模型规模', '冻结参数量', cplx['frozen_params'], 'params', '')
    add(rows, '模型规模', '参数量(FP32)', cplx['param_memory_mb'], 'MB', '')
    add(rows, '模型规模', '未参与前向的 decoder 参数', cplx['unused_decoder_params'], 'params',
        'T5ForConditionalGeneration 的 decoder 被实例化但前向中未调用')
    if args.checkpoint and os.path.exists(args.checkpoint):
        add(rows, '模型规模', 'checkpoint 文件大小',
            os.path.getsize(args.checkpoint) / 1024 ** 2, 'MB', os.path.basename(args.checkpoint))
    for name, st in cplx['per_module'].items():
        add(rows, '子模块参数', name, st['params'], 'params', f"可训练 {st['trainable']:,}")

    # ---- 2) 前向实际使用参数量 & FLOPs ----
    probe_df = build_synthetic_df(max(args.batch_sizes), cfg['max_lines'])
    probe_batch = build_batch(mod, probe_df, cfg['max_lines'],
                              cfg['max_func_length'], cfg['max_line_length'])
    probe_inputs = to_inputs(probe_batch, device)

    print('测量前向实际使用参数量...')
    used = measure_forward_used_params(model, lambda: model(**probe_inputs))
    add(rows, '模型规模', '实际参与前向的参数量', used, 'params',
        '由 forward hook 统计，剔除未调用的 decoder')

    print('测量 FLOPs...')
    flops = measure_flops(model, probe_inputs, device)
    if flops:
        add(rows, '计算量', '单函数前向 FLOPs', flops, 'FLOPs',
            f'batch={probe_inputs["func_inputs"]["input_ids"].size(0)}')

    # ---- 3) 推理成本 ----
    for bs in args.batch_sizes:
        print(f'测量推理成本 (batch_size={bs})...')
        df = build_synthetic_df(bs, cfg['max_lines'])
        batch = build_batch(mod, df, cfg['max_lines'],
                            cfg['max_func_length'], cfg['max_line_length'])
        inputs = to_inputs(batch, device)

        times = time_forward(model, inputs, device, args.warmup, args.iters)
        med = statistics.median(times)
        p95 = sorted(times)[int(len(times) * 0.95) - 1] if len(times) > 1 else med
        add(rows, '推理成本', f'单函数延迟 (bs={bs})', med / bs * 1000, 'ms/函数',
            f'batch={bs} 中位延迟 {med*1000:.2f} ms')
        add(rows, '推理成本', f'单函数延迟 p95 (bs={bs})', p95 / bs * 1000, 'ms/函数', '')
        add(rows, '推理成本', f'吞吐 (bs={bs})', bs / med, '函数/s', '')

        peak, resv = measure_inference_memory(model, inputs, device)
        if peak:
            add(rows, '内存消耗', f'推理峰值显存 (bs={bs})', peak, 'MB', 'max_memory_allocated')
            add(rows, '内存消耗', f'推理峰值显存(含缓存) (bs={bs})', resv, 'MB', 'max_memory_reserved')

    # ---- 4) 逐行编码开销 ----
    print('测量逐行独立编码开销...')
    le = time_line_encoding(model, probe_inputs, device, args.warmup, args.iters)
    add(rows, '逐行编码开销', f'逐行独立编码耗时 ({le["num_lines"]} 行)',
        le['independent_ms'], 'ms', '当前实现：每行一次编码器调用')
    add(rows, '逐行编码开销', '合并为单次编码耗时',
        le['batched_ms'], 'ms', '把所有行拼成一个 batch 只调用一次')
    if le['batched_ms'] > 0:
        add(rows, '逐行编码开销', '独立编码相对合并编码的倍数',
            le['independent_ms'] / le['batched_ms'], 'x', '越大说明逐行编码的开销越显著')

    # ---- 5) 模块耗时拆分 ----
    print('测量模块耗时拆分...')
    for r in time_submodules(model, probe_inputs, device, args.iters):
        add(rows, '模块耗时拆分', r['module'], r['ms'], 'ms', f"占比 {r['share']:.1f}%")

    # ---- 6) 训练成本 ----
    print('测量训练成本...')
    steps_per_epoch = None
    if args.train_csv and os.path.exists(args.train_csv):
        with open(args.train_csv, 'r', encoding='utf-8') as f:
            n_train = sum(1 for _ in f) - 1
        steps_per_epoch = n_train // cfg['batch_size']
        add(rows, '训练成本', '训练集样本数', n_train, 'samples', os.path.basename(args.train_csv))
        add(rows, '训练成本', '每轮训练步数', steps_per_epoch, 'steps', f"batch_size={cfg['batch_size']}")

    train_df = build_synthetic_df(cfg['batch_size'], cfg['max_lines'])
    train_batch = build_batch(mod, train_df, cfg['max_lines'],
                              cfg['max_func_length'], cfg['max_line_length'])
    tr = measure_training_step(mod, model, train_batch, device,
                               args.train_steps, {'steps_per_epoch': steps_per_epoch})

    add(rows, '训练成本', '单步训练耗时', tr['ms_per_step'], 'ms/step',
        f"batch_size={tr['batch_size']}，含前向+反向+优化器")
    add(rows, '训练成本', '训练吞吐', tr['samples_per_sec'], 'samples/s', '')
    if 'sec_per_epoch' in tr:
        add(rows, '训练成本', '单轮训练耗时', tr['sec_per_epoch'] / 60, 'min/epoch',
            f"epochs={cfg['epochs']}")
        add(rows, '训练成本', '总训练时长', tr['total_train_hours'], 'GPU-hours',
            f"epochs={cfg['epochs']}，外推值")
    if 'peak_alloc_mb' in tr:
        add(rows, '内存消耗', '训练峰值显存', tr['peak_alloc_mb'], 'MB', 'max_memory_allocated')
        add(rows, '内存消耗', '训练峰值显存(含缓存)', tr['peak_reserved_mb'], 'MB',
            'max_memory_reserved')

    # ---- 输出 ----
    res_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, 'efficiency_results.csv')
    md_path = os.path.join(args.out_dir, 'efficiency_table.md')
    res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    write_markdown(rows, md_path, args)

    print('\n' + '=' * 70)
    print(res_df.to_string(index=False))
    print('=' * 70)
    print(f'结果已保存到:\n  {csv_path}\n  {md_path}')


if __name__ == '__main__':
    main()
