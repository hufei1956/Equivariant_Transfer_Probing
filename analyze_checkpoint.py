import os
import io
import math
import torch


def human_bytes(n):
    return f"{n/1_000_000:.2f} MB"


def human_count(n):
    return f"{n:,}"


def round_pct(x):
    return f"{x*100:.1f}%"


def save_size(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


def tensor_bytes(t):
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


def param_count_and_bytes(state_dict):
    total_numel = 0
    total_bytes = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total_numel += v.numel()
            total_bytes += tensor_bytes(v)
        elif isinstance(v, (list, tuple)):
            for t in v:
                if isinstance(t, torch.Tensor):
                    total_numel += t.numel()
                    total_bytes += tensor_bytes(t)
    return total_numel, total_bytes

def _classify_type(name):
    n = name.lower()
    if n.startswith("fc.") or ".fc." in n:
        return "fc"
    if "bn" in n or "batchnorm" in n:
        return "bn"
    if "conv" in n or n.endswith("weights") or ".weights" in n:
        return "conv"
    return "other"

def _classify_stage(name):
    n = name.lower()
    if n.startswith("conv1.") or n.startswith("bn1.") or n.startswith("relu1.") or n.startswith("maxpool."):
        return "stem"
    for k in ["layer1", "layer2", "layer3", "layer4"]:
        if k in n:
            return k
    if n.startswith("gpool."):
        return "gpool"
    if n.startswith("fc."):
        return "head"
    if "downsample" in n:
        return "downsample"
    return "other"

def breakdown_stats(state_dict):
    type_stats = {}
    stage_stats = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            b = tensor_bytes(v)
            t = _classify_type(k)
            s = _classify_stage(k)
            type_stats[t] = type_stats.get(t, 0) + b
            stage_stats[s] = stage_stats.get(s, 0) + b
    return type_stats, stage_stats


def detect_model_type(name, ckpt):
    low = name.lower()
    if "escnn" in low:
        if "so2f" in low:
            return "ESCNN-" + low.split("best_escnn_")[1].split("_")[0].upper()
        elif "_c4_" in low:
            return "ESCNN-C4"
        elif "_c8_" in low:
            return "ESCNN-C8"
        elif "_c16_" in low:
            return "ESCNN-C16"
        elif "_d4_" in low:
            return "ESCNN-D4"
        elif "_d8_" in low:
            return "ESCNN-D8"
        return "ESCNN"
    if "resnet18" in low:
        return "ResNet18"
    if isinstance(ckpt, dict) and "group_type" in ckpt:
        g = str(ckpt["group_type"]).upper()
        return f"ESCNN-{g}"
    return "Model"


def analyze_file(path):
    name = os.path.basename(path)
    file_size = os.path.getsize(path)
    ckpt = torch.load(path, map_location="cpu")
    model_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    optimizer_state = ckpt["optimizer_state"] if isinstance(ckpt, dict) and "optimizer_state" in ckpt else None
    classes = ckpt["classes"] if isinstance(ckpt, dict) and "classes" in ckpt else None
    base_channels = ckpt["base_channels"] if isinstance(ckpt, dict) and "base_channels" in ckpt else None
    group_type = ckpt["group_type"] if isinstance(ckpt, dict) and "group_type" in ckpt else None
    max_frequency = ckpt["max_frequency"] if isinstance(ckpt, dict) and "max_frequency" in ckpt else None
    model_type = detect_model_type(name, ckpt)

    num_params, param_bytes = param_count_and_bytes(model_state if isinstance(model_state, dict) else {})
    ser_model_bytes = save_size(model_state)
    ser_optim_bytes = save_size(optimizer_state) if optimizer_state is not None else 0
    other_bytes = max(0, file_size - ser_model_bytes - ser_optim_bytes)

    part_total = ser_model_bytes + ser_optim_bytes + other_bytes
    pct_model = ser_model_bytes / part_total if part_total > 0 else 0.0
    pct_optim = ser_optim_bytes / part_total if part_total > 0 else 0.0
    pct_other = other_bytes / part_total if part_total > 0 else 0.0

    cls_count = len(classes) if isinstance(classes, (list, tuple)) else None
    if cls_count is None and isinstance(model_state, dict):
        for k, v in model_state.items():
            if isinstance(v, torch.Tensor) and k.endswith("fc.weight"):
                cls_count = v.shape[0]
                break
    type_stats, stage_stats = breakdown_stats(model_state if isinstance(model_state, dict) else {})

    lines = []
    lines.append("=" * 79)
    lines.append(f"文件: {name}")
    lines.append("=" * 79)
    lines.append("")
    lines.append("【基本信息】")
    lines.append(f"  模型类型: {model_type}")
    lines.append(f"  文件大小: {human_bytes(file_size)}")
    lines.append(f"  类别数: {cls_count if cls_count is not None else '未知'}")
    lines.append(f"  基础通道数: {base_channels if base_channels is not None else '未知'}")
    if group_type is not None:
        lines.append(f"  对称群: {str(group_type).upper()}")
    if max_frequency is not None:
        lines.append(f"  SO2 最大频率: {max_frequency}")
    lines.append("")
    lines.append("【参数统计】")
    lines.append(f"  模型参数量: {human_count(num_params)} ({num_params/1_000_000:.2f}M)")
    lines.append(f"  模型参数大小: {human_bytes(param_bytes)}")
    lines.append(f"  优化器状态: {human_bytes(ser_optim_bytes)}")
    lines.append(f"  其他开销: {human_bytes(other_bytes)}")
    lines.append("")
    lines.append("【文件组成】")
    lines.append(f"  模型参数: {human_bytes(ser_model_bytes)} ({round_pct(pct_model)})")
    lines.append(f"  优化器:   {human_bytes(ser_optim_bytes)} ({round_pct(pct_optim)})")
    lines.append(f"  其他:      {human_bytes(other_bytes)} ({round_pct(pct_other)})")
    lines.append("  " + "─" * 29)
    lines.append(f"  总计:     {human_bytes(file_size)} (100.0%)")
    lines.append("")
    lines.append("【层级统计】")
    lines.append("  按类型:")
    total_b = max(1, param_bytes)
    for t in ["conv", "bn", "fc", "other"]:
        b = type_stats.get(t, 0)
        lines.append(f"    - {t}: {human_bytes(b)} ({round_pct(b/total_b)})")
    lines.append("  按 stage:")
    for s in ["stem", "layer1", "layer2", "layer3", "layer4", "downsample", "gpool", "head", "other"]:
        b = stage_stats.get(s, 0)
        if b > 0:
            lines.append(f"    - {s}: {human_bytes(b)} ({round_pct(b/total_b)})")
    lines.append("")
    return "\n".join(lines)


def main():
    root = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.isdir(root):
        print("未找到 checkpoints 目录")
        return
    files = [f for f in os.listdir(root) if f.lower().endswith(".pth")]
    if not files:
        print("checkpoints 目录下没有 .pth 文件")
        return
    for fn in files:
        path = os.path.join(root, fn)
        report = analyze_file(path)
        out_name = os.path.splitext(fn)[0] + ".txt"
        out_path = os.path.join(root, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()
