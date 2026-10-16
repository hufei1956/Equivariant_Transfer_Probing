# 在 WSL 中同时使用两张 RTX 5090 的方法

## 1. 为什么原来的程序只用了一张卡

PyTorch 默认只用第一张 GPU。原脚本里 `device = torch.device("cuda")` 等价于 `cuda:0`，第二张卡（`cuda:1`）完全空闲。
这不是 WSL 的限制：`nvidia-smi` 能看到两张卡，PyTorch 也能看到（`torch.cuda.device_count()` 返回 2），只是程序没有去用它。

另外，原脚本的 `num_workers=0`（这是 Windows 下的建议）在 WSL/Linux 下没有必要。图片解码和数据增强只在一个 CPU 核上串行执行，GPU 大部分时间在等数据。这台机器有 32 个 CPU 线程，把它改成 4–8 通常会快不少。

## 2. 这次实验是怎么用两张卡的：多个独立进程并行（推荐）

这次没有把一个模型拆到两张卡上，而是**同时运行多个互不相关的训练进程**，每个进程只看得到一张卡：

```bash
# 终端 1：在 GPU 0 上训练
CUDA_VISIBLE_DEVICES=0 python train.py --dataset-name trashnet  --seed 42 &
# 终端 2：在 GPU 1 上训练
CUDA_VISIBLE_DEVICES=1 python train.py --dataset-name realwaste --seed 42 &
```

`CUDA_VISIBLE_DEVICES=1` 会让该进程只看到第二张卡，并把它当成 `cuda:0`，所以**代码一行都不用改**。

这种方式适合论文实验：多个种子、多个模型、两个数据集，本质上就是大量相互独立的训练任务。它能做到几乎线性的加速，也最稳定。

这次的探针模型都比较小，一个进程远远用不满一张 5090，所以每张卡同时跑 4 个进程，两张卡共 8 个。GPU 利用率约 90%，170 个 run 大约 8 小时跑完。
每张卡同时跑几个，要看显存和利用率：先起 2 个，用 `nvidia-smi` 看利用率还没到 90% 并且显存有余量，再往上加。

### 自动排队：`r1exp/launch.py`

手工开几个终端容易乱，`r1_revision/r1exp/launch.py` 就是一个简单的排队器。它把任务列表分配到 `--gpus 0,1`、每卡 `--per-gpu N` 个槽位；某个任务结束就启动下一个；已完成的任务（结果 json 已存在）会自动跳过，所以可以随时中断、随时续跑。
要给自己的其他项目用，只需把其中构造 `cmd = [...]` 的那几行改成你自己的训练命令。核心逻辑如下：

```python
import os, subprocess, time
jobs = [["python", "train.py", "--seed", str(s), "--dataset-name", d]
        for s in (42, 123, 2024) for d in ("trashnet", "realwaste")]
slots = [(g, k) for g in ("0", "1") for k in range(2)]      # 2 张卡 × 每卡 2 个
running = {}
while jobs or running:
    for slot in slots:
        if slot not in running and jobs:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot[0])
            running[slot] = subprocess.Popen(jobs.pop(0), env=env)
    time.sleep(10)
    running = {s: p for s, p in running.items() if p.poll() is None}
```

后台运行、关掉终端后不中断：

```bash
nohup setsid python -m r1exp.launch --tiers 1 --gpus 0,1 --per-gpu 4 >> launcher.log 2>&1 &
```

（注意：不要执行 `wsl --shutdown`，也不要让 Windows 睡眠，否则 WSL 里的所有进程都会被终止。）

## 3. 一个训练任务同时用两张卡：DistributedDataParallel (DDP)

**只有单个模型训练太慢**（比如大模型、大数据集，或者 YOLO 检测训练）时才需要这种方式。每张卡各持一份模型副本，各自处理一半的 batch，反向传播时同步梯度。

对 `train.py` 需要改这几处：

```python
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group(backend="nccl")                 # 1) 初始化
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

model = build_model(...).to(device)
model = DDP(model, device_ids=[local_rank])             # 2) 包装模型
# BN 很多时可选：model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)（在 DDP 包装之前做）

sampler = DistributedSampler(train_set, shuffle=True)   # 3) 每张卡拿不同的数据
train_loader = DataLoader(train_set, batch_size=16,     #    注意：这是"每张卡"的 batch
                          sampler=sampler, num_workers=4, pin_memory=True)

for epoch in range(epochs):
    sampler.set_epoch(epoch)                            # 4) 每个 epoch 重新打乱
    train_one_epoch(model, train_loader, ...)
    if dist.get_rank() == 0:                            # 5) 只在主进程上验证/保存/打印
        torch.save(model.module.state_dict(), "best.pth")   # 保存 .module，便于单卡加载

dist.destroy_process_group()
```

启动方式（不再用 `python train.py`）：

```bash
torchrun --nproc_per_node=2 train.py --epochs 100
```

注意事项：
- **总 batch = 每卡 batch × 2。** 如果想和单卡结果可比，要么每卡 batch 减半，要么按总 batch 等比例放大学习率（linear scaling）。论文里对比用的实验，建议保持和单卡一致的有效 batch。
- **WSL2 下的 NCCL：** 两张卡之间通常没有 P2P 直连。若训练卡在第一步不动，先设 `export NCCL_P2P_DISABLE=1` 再启动；仍有问题可加 `export NCCL_IB_DISABLE=1`，并用 `NCCL_DEBUG=INFO` 查看日志。`nvidia-smi topo -m` 可以查看两卡之间的连接方式。
- **ESCNN 等变网络**是普通的 `nn.Module`，可以直接用 DDP 包装。
- **Ultralytics YOLO**已内置多卡支持，不用改代码：`model.train(data=..., device=[0, 1])`，或命令行 `yolo train ... device=0,1`。
- **`nn.DataParallel`**（一行 `model = nn.DataParallel(model)`）写法最简单，但单进程、效率低，也不适合 ESCNN，不推荐。

## 4. 怎么选

| 情况 | 推荐做法 |
|---|---|
| 多个种子 / 多个模型 / 多个数据集的对比实验（论文里绝大多数情况） | **第 2 种**：每卡跑独立进程，`CUDA_VISIBLE_DEVICES` 指定卡，必要时每卡多个进程 |
| 只有一个很大、很慢的训练任务 | **第 3 种**：DDP + `torchrun` |
| YOLO 检测训练 | `device=[0,1]` |

## 5. 这台机器上另外发现的两个问题

1. **numpy 与 torch 不兼容。** 在 `p310torch27` 环境里，`torchvision.transforms.ToTensor()` 在多进程 DataLoader 中会报错 `expected np.ndarray (got numpy.ndarray)`。原来的 `dataset.py` 用 `CustomToTensor` 绕过了这个错误，但它用 `list(pic.getdata())` 逐像素转换，非常慢。`r1exp/data.py` 里的 `PILToFloatTensor` 用 `torch.frombuffer` 转换，不经过 numpy，速度快，可以直接替换。根本的解决办法是重装一个和 torch 2.7 匹配的 numpy（`pip list | grep -i numpy` 先检查是否同时存在两份 numpy）。
2. **GPU 0 同时负责显示输出**（桌面占用约 1.2 GB 显存）。任务数为奇数时，优先把多出的那个放到 GPU 1。
