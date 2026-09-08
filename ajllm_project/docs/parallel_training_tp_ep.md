# TP / EP 并行训练设计记录（历史）

> 当前已实现的接口、限制和验证状态见 [parallel_training.md](parallel_training.md)。本文保留实现前的原始设计推导，其中的“拟实现”描述均为历史记录，不能作为可运行配置说明。

本文记录当时的设计目标：让现有 `TransformerLM` 通过 wrap 获得 Tensor Parallelism（TP）和 Expert Parallelism（EP），并保留原模型与 Triton 实现。实现后的实际接口和命令以主文档为准。

## 1. 结论与实现边界

可以提供 `model = wrap_parallel(model, context, config)` 的使用方式，无需用户用 Megatron 的并行层重新搭建模型。但 TP/EP 需要改变层内部的计算与通信，不能只把现有 FSDP 的参数 all-gather hooks 换个名字。

- FSDP：计算时恢复当前层的完整权重，主要切分参数、梯度与优化器状态的存储。
- TP：计算时只使用权重的一部分，通过激活及其梯度通信合成完整运算。
- EP：每卡只拥有部分专家，把 token 发给专家所在的卡，计算后再送回来源卡。

拟采用“统一外层 wrapper + 项目专用模块转换规则 + 普通 Tensor 上的可微通信”。原始构模配置、残差结构、路由数学定义、`forward(input_ids) -> logits` 均保留。必要时用并行适配模块替换已构建模型中的 Attention、SwiGLU、MoE；原始单卡类仍可独立使用。

训练循环中的前向、loss、backward 和 AdamW 更新公式保持原有用法，但数据采样、梯度同步与裁剪、AMP 溢出处理、保存恢复必须接入并行上下文。这是一次训练基础设施适配，不能承诺只改一行 wrap 就让当前所有外围逻辑自动正确。

实现顺序：通信与梯度基线 → Dense TP → Torch EP → MoE TP×EP → 本地 Triton 专家 → 与现有 FSDP 组合。前几个阶段不开放尚未验证的组合；最终目标包含 TP、EP、TP×EP，以及它们与数据并行/FSDP 的组合。

首轮不纳入：Pipeline Parallel、Context/Sequence Parallel、词表并行交叉熵、KV head 复制分片、不等量专家分配、token capacity/drop、通信计算重叠、跨拓扑优化器重分片。后续可以在同一接口下增加。

## 2. 当前代码审查结果

以下是源码事实与对本方案的影响，不是对既有多卡正确性的测试结论。

| 文件 / 对象 | 当前行为 | 设计影响 |
| --- | --- | --- |
| `src/ajllm/training/parallel/fsdp.py` | `FullyShardedDataParallel` 使用默认全局进程组，前向 gather，反向自定义 autograd reduce-scatter 并求平均 | 所有通信函数需支持显式 group；不能让 TP/EP 继续隐式使用 WORLD |
| 同上 | 匹配自定义 `Linear`，含其子类 `VariableGroupedLinear`；非分片参数在 `finish_gradient_synchronization()` 中平均 | 必须按逻辑参数归属区分同步范围 |
| 同上 | FSDP 默认 checkpoint；此模式 gather 在 checkpointed forward 内，前后 gather hooks 不启用 | 组合时需要唯一的权重恢复和 checkpoint 管理者 |
| `modeling/layers.py` | `Linear` 是自定义 `nn.Module`，权重 `[out, in]`；Embedding 同样自定义 | 不能假定原生 PyTorch TP 样式可直接匹配 |
| `modeling/attention.py` | 用全局 `num_heads`、`num_kv_heads` reshape；输出重新 view 为 `self.d_model` | 仅切权重会导致 reshape 错误，需要本地 head 元数据及输出宽度适配 |
| 同上 | Q/K Norm 的权重为 `[head_dim]`，在所有 heads 上共享 | TP 后各卡是不同 head 的梯度贡献，要 SUM，不能按普通复制参数处理 |
| `modeling/activations.py` | Dense SwiGLU 为 gate/up/down 三个投影 | gate/up 切输出维，down 切输入维 |
| 同上 | Top-1 grouped 权重为 `[experts, 2*d_ff, d_model]` 和 `[experts, d_model, d_ff]` | EP 切第 0 维；TP 要分别切 gate 与 up，不能直接等分融合维 |
| `modeling/moe.py` | Top-1 用 argmax、归一化权重恒为 1；Top-K>1 用选中概率再次归一化 | 保持原路由定义，不能引入概率加权 Top-1 |
| 同上 | Top-1 router 主损失不经过 argmax 求导；训练信号来自 auxiliary loss | 验证 router 梯度时必须纳入辅助损失 |
| `modeling/transformer.py` | tied head 直接读取 `token_embeddings.weight`；辅助损失用 `isinstance(..., MoELayer)` 收集 | 首轮复制词表权重；EP adapter 需继承 `MoELayer` 或由 wrapper 显式收集 |
| `training/pretrainer.py` | 只识别 FSDP wrapper；全局梯度平方和直接 WORLD SUM | 需统一 wrapper 协议，裁剪时对复制参数去重 |
| `workflows/pretrain.py` | sampler 按 WORLD 切数据，seed 为 `seed + rank`；多进程且不启用 FSDP 时会报错 | TP 同组数据与随机状态必须一致；入口需支持新的并行模式 |
| `training/checkpoint.py` | FSDP 完整模型导出与各 rank optimizer 文件；恢复只检查 world size | 新格式需保存完整拓扑、分片映射与 RNG/AMP 状态 |
| `training/evaluation.py` | 全局 SUM loss 总量及 token 数 | TP 复制样本不能重复计数；EP/DP 不同样本需要计入 |

在当前 checkout 中，没有独立 DP/DDP wrapper，预训练入口还明确拒绝非 FSDP 多进程模式。因此这里沿用已有 FSDP，另补充混合并行需要的“复制参数梯度归约”，不假设仓库已经接入可直接组合的 DDP。

## 3. 为什么选择项目内 wrapper

PyTorch 自身也支持对已构建模型按计划调用 `parallelize_module`，无需先用另一套模型类构模。但 2.11 的 `ColwiseParallel` / `RowwiseParallel` 文档列出的支持层为 `nn.Linear` 与 `nn.Embedding`，且明确提醒分片输出会要求调整后续 shape 操作。[PyTorch 2.11 TP 文档](https://docs.pytorch.org/docs/2.11/distributed.tensor.parallel.html)

本项目的自定义 Linear、三维专家权重和 Triton autograd 接口，都需要额外规则。因此首轮直接实现少量可审查的规则，避免同时引入 DTensor 布局传播与自定义内核适配。今后可以增加原生 TP 后端，但不在首轮维护两套实现。

wrapper 的自动化范围是“当前已知的 ajLLM 模块”。未知模块、重复 wrap、已被不兼容 FSDP 改写的模型应立即报错，不能静默跳过后继续宣称全模型已并行化。

## 4. 并行拓扑与数据语义

### 4.1 坐标与进程组

定义 `D = dp_size`、`P = ep_size`、`T = tp_size`，总进程数 `W = D * P * T`。`N` 表示专家总数，避免与 EP size 混用。

使用坐标 `(d, e, t)`，映射为：

```text
global_rank = ((d * P) + e) * T + t
```

| 组 | 固定坐标 | 变化坐标 | 用途 |
| --- | --- | --- | --- |
| TP | d, e | t | 同一数据 batch 的张量并行 |
| EP | d, t | e | token dispatch/combine；非专家参数跨 e 同步 |
| DP | e, t | d | 相同专家归属及 TP 分片在不同数据副本间同步或 FSDP |
| Dense replica | t | d, e | 不使用 FSDP 时，同一 Dense TP 分片的复制梯度平均 |
| Model | d | e, t | 一份逻辑模型的不同专家及张量分片 |
| WORLD | 无 | d, e, t | 启动检查、统一停止、AMP 溢出共识等控制操作 |

所有 rank 按同一确定性顺序创建全部组，包括不属于自身的组；业务通信总是显式传入对应 group。PyTorch 对组创建及多组 collective 顺序有要求，实现时按 2.11 文档检查。[PyTorch distributed 文档](https://docs.pytorch.org/docs/2.11/distributed.html)

### 4.2 batch 分配

同一 `(d,e)` 下所有 TP ranks 处理相同 `input_ids` 和 `labels`。不同 `d`、不同 `e` 处理不同 batch；EP 卡并不是整模型副本，但它们的非专家部分分别处理自己的数据，专家部分交换计算。

```text
data_world_size = D * P
data_rank = d * P + e
effective_batch_size = local_batch_size * gradient_accumulation_steps * D * P
```

TP 不扩大有效 batch。sampler 应使用上述 data rank。首轮可让 TP 各卡以相同 sampler/seed 加载相同数据；为支持以后随机数据增强，统一由 TP leader 广播 batch，并将相同长度/步数作为必要约束。不能只广播输入而让 labels 各卡不同。

例：8 卡、`D=2,P=2,T=2`：

| ranks | 数据坐标 | TP 组 | 专家归属（N=4） |
| --- | --- | --- | --- |
| 0, 1 | d=0,e=0 | [0,1] | experts 0,1，每个专家再 TP=2 |
| 2, 3 | d=0,e=1 | [2,3] | experts 2,3，每个专家再 TP=2 |
| 4, 5 | d=1,e=0 | [4,5] | experts 0,1 的第二份数据副本 |
| 6, 7 | d=1,e=1 | [6,7] | experts 2,3 的第二份数据副本 |

EP 组为 `[0,2]`、`[1,3]`、`[4,6]`、`[5,7]`；DP 组为 `[0,4]`、`[1,5]`、`[2,6]`、`[3,7]`。

该布局是本项目的明确约定，不把它称为 Megatron 的完全等价布局。Megatron 提供更复杂的专家/张量并行映射及通信重叠选项，可作后续优化参考。[Megatron MoE 文档](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)

### 4.3 配置合法性

首轮在任何参数替换之前检查：

- `D,P,T >= 1` 且 `D*P*T == world_size`；所有 rank 的配置摘要一致。
- Dense 模型要求 `P=1`；MoE 要求 `N % P == 0`。
- `num_heads % T == 0`、`num_kv_heads % T == 0`、`d_ff % T == 0`。
- 保留 `head_dim = global_d_model / global_num_heads`，不切单个 head 内部维度。
- fused gate/up 切片、dtype、参数共享关系、各层配置符合已注册转换规则。
- `data_sharding=fsdp` 的组合阶段要求 `D>1`；单卡退化测试可直接使用单位大小 context。

默认配置 `Hq=12,Hkv=4,d_ff=2432,N=4` 支持 `T=1,2,4` 和 `P=1,2,4`。8 张卡不意味着 `T=8` 合法；KV 复制也不能解决 12 个 Q heads 无法均匀分给 8 卡的问题。

## 5. TP 计算与反向传播

### 5.1 两种基础线性层

源码采用 `Y = X @ W.T`，`W=[O,I]`。为避免术语与存储顺序混淆，所有分片元数据都记录真实 tensor axis。

| 层 | 本地权重 | 前向 | 反向 |
| --- | --- | --- | --- |
| ColumnParallelLinear | `W_t=[O/T,I]`，切 axis 0 | `Y_t=X@W_t.T`，输出保留分片 | `dX=SUM_t(dY_t@W_t)`；dW 仅本地 |
| RowParallelLinear | `W_t=[O,I/T]`，切 axis 1 | `Y=SUM_t(X_t@W_t.T)`，输出复制 | dY 直接传给各本地分支；dX_t/dW_t 本地计算 |

采用明确的自定义 autograd 映射：

```text
copy_to_tp_region:
    forward: identity
    backward: all_reduce(SUM, tp_group)

reduce_from_tp_region:
    forward: all_reduce(SUM, tp_group)
    backward: identity
```

这是针对“每张 TP 卡计算同一份 loss、输出复制”的约定。TP 中不能除以 T；也不能在 row 输出的 backward 再做一次 SUM，否则会重复放大梯度。训练所需 collective 不直接依赖普通原地 `dist.all_reduce` 自动生成梯度；由自定义 Function 定义布局对应的反向。

首轮独立 gate/up 或 Q/K/V 投影可以各自执行输入梯度 SUM，正确但通信次数较多。后续在共享输入处放置一个 copy 映射，将多个分支的局部 dX 相加后再 SUM，减少 collective；同一输入路径不能同时保留两套归约。

### 5.2 Dense SwiGLU

```text
X [B,S,H]，TP 复制
  ├─ gate_proj: [F/T,H] ─ SiLU ─┐
  └─ up_proj:   [F/T,H] ────────× → hidden [B,S,F/T]
                                     ↓
                         down_proj: [H,F/T]
                                     ↓
                            TP SUM → [B,S,H]
```

gate 与 up 必须选择同一组中间通道。SwiGLU 是逐元素计算，可对本地分片继续使用原 Triton gate。只有 down 输出恢复全 hidden 宽度后才能进入 residual/dropout。

### 5.3 GQA

转换后的 Attention 保存全局配置与本地配置：

```text
local_num_heads = global_num_heads / T
local_num_kv_heads = global_num_kv_heads / T
head_dim 不变
num_kv_groups = global_num_heads / global_num_kv_heads，不变
local_attention_width = local_num_heads * head_dim
```

Q/K/V 按连续且对齐 GQA 分组的 head 区间切输出维；output_proj 切输入维。局部 attention 结果 flatten 为 `[B,S,local_attention_width]`，经 row-parallel output projection 后才变回 `[B,S,global_d_model]`。

不能仅把 `self.d_model` 改小：它的全局 hidden 语义仍被其他逻辑使用。拟通过 `TensorParallelAttention` adapter 显式管理本地 shape，并复用已有 RoPE、repeat_kv 和 attention kernel 调用。

RoPE 的 `positions` 跟本地 head 数展开，cache 复制。FlashAttention 只计算本地完整 heads，保留现有 sequence padding 行为。当前 `use_flash_attention=False` 路径调用的是项目的 `flash_attention_pytorch`，验证中沿用实际实现，不将它误写为直接调用 SDPA。

### 5.4 复制参数不等于复制梯度

| 参数 | TP 后梯度性质 | TP 同步 |
| --- | --- | --- |
| Q/K/V、output_proj、SwiGLU 权重分片 | 不同参数元素 | 不做参数梯度平均 |
| Q/K Norm `[head_dim]` | 同一参数在不同 heads 上的部分贡献 | 每个 optimizer step 前 SUM 一次 |
| block RMSNorm、final_norm | 输入及反向贡献已经在 TP 边界恢复完整 | 无需 TP SUM |
| 复制 embedding / LM head | 每卡拥有同一份完整梯度 | 无需 TP SUM |
| MoE router | 各 TP 卡相同路由和同一完整 router 梯度 | 无需 TP SUM |

这些规则必须登记为参数布局元数据，不能用“非分片参数全部平均”推导。还要避免 Q/K Norm hook 与统一 finalize 同时 SUM。

### 5.5 词表与损失接口

首轮 embedding 和输出词表矩阵在 TP 内复制，`tie_embeddings=True` 仍共享同一 Parameter；`False` 时 LM head 同样复制。`forward` 返回完整 `[B,S,V]` logits，现有 Torch/Triton 交叉熵保留。

代价是词表权重与 logits 显存不随 TP 缩小。当前 FSDP 对非 tied 的自定义 Linear head 仍可沿 D 切存储，但 embedding 不切。未来词表并行必须同时处理 tied weight 读取和分布式 softmax/交叉熵，不能只替换 embedding 就宣布完成。

## 6. EP token 生命周期

### 6.1 专家拥有权与存储

连续分配：

```text
local_experts = N / P
owner(global_expert_id) = global_expert_id // local_experts
local_id(global_expert_id) = global_expert_id % local_experts
```

router 保持 `[N,H]`，看到全部专家；每卡只注册自己的专家权重。远端专家不能作为隐藏子模块或 Parameter 留在 wrapper 中，否则不会节省参数与 optimizer 显存。

Top-1 保留 grouped 权重格式，仅切专家维；Top-K>1 保留原专家的全局编号映射，使用本地 ModuleDict 或等价映射。转换先完成全局权重同步与分片，再释放不属于本 rank 的完整参数。

### 6.2 前向 dispatch / compute / combine

所有 EP ranks 都必须进入同一个通信序列，即使某卡收到 0 个 token。

1. 来源卡把 `[B,S,H]` flatten 为 `[M,H]`，执行原 router。
2. Top-1 生成 `(token_id, expert_id)`；Top-K 生成 M*K 条 assignment，另在来源卡保存可微的归一化 routing weights。
3. 按 owner 排序 assignment，保存原 token/slot 索引与逆置换，统计 `send_counts[P]`。
4. 先交换 counts，获得 `recv_counts[P]`；发送 rows 的分段单位是 token assignment 行，不是字节或 hidden 元素数。
5. 对 packed hidden 执行可微 variable-split all-to-all；expert id 等整数元数据执行相同分段的非可微传输。
6. 接收端以 local expert id 再排序，构造本地 offsets，对每个本地专家执行 SwiGLU。
7. 撤销接收端的 expert 排序，恢复最初接收的 source-major 顺序。
8. 将输出按反向 splits all-to-all 送回来源卡。
9. 来源卡撤销 owner 排序：Top-1 恢复一一对应顺序；Top-K 将每条输出乘以来源端保存的 routing weight，再按 token 执行 `index_add`。

首轮不丢 token、不设容量上限。负载偏斜时允许单卡承接大量 token，因此 EP 不保证激活显存按 P 均分；必须测最坏路由。

PyTorch `all_to_all_single` 支持按 dim 0 指定不等长 input/output splits，本方案的生产通信基于此接口。[PyTorch all_to_all_single](https://docs.pytorch.org/docs/2.11/distributed.html#torch.distributed.all_to_all_single)

### 6.3 可微 all-to-all

```text
forward(x, send_splits, recv_splits, group):
    y = all_to_all_single(x, input_splits=send_splits,
                         output_splits=recv_splits, group=group)
backward(dy):
    dx = all_to_all_single(dy, input_splits=recv_splits,
                          output_splits=send_splits, group=group)
```

保存 group、splits、必要的 shape；counts、索引不求梯度。all-to-all 本身不平均、不除 P；来源激活梯度必须保持本地 loss 的尺度。Top-K 同一 token 的输入梯度由可微 gather/index-select 的反向累加。

一个 MoE 前向有 hidden 派发与输出回传两次浮点 all-to-all，反向各有对应的逆通信，总计四次浮点 token 通信，另有 counts/整数元数据通信。

### 6.4 零 token 和未使用专家

需要区分“空 tensor”与“无需参与通信”：即便空 tensor，collective 次序仍要一致。

- 单个空专家不发 kernel tile；全空接收端不能启动非法零 grid kernel。
- 全空路径仍保持对 dispatch 输入和专家参数的零梯度依赖，让 autograd 走完跨卡通信；不能直接返回一个与图无关的新空 tensor。
- 专家参数的 grad presence 在拥有同一专家的 DP 副本间先协调；有任何副本使用时，未使用副本补零再归约，所有成员调用相同 collective。
- 全局未使用专家的 optimizer 行为要与基线一致：Top-1 grouped 参数整体参与运算，空 expert slice 梯度为零；Top-K ModuleList 中整份未用专家通常 grad 为 None，AdamW 会跳过。不能未经说明把全部 None 改成零而改变 weight decay/momentum 行为。
- 上述 presence 是按实际注册 Parameter 记录；TP shards 的 optimizer step 状态也要一致。

Gloo 测试先探测 variable-split/all-zero 支持情况；不支持时提供测试专用的 padded all-gather + slicing 传输，其 backward 显式还原各来源梯度求和。不能把普通 all-gather 当作天然可微，也不能将这种测试替代实现当作 NCCL 已验证。

## 7. EP 的损失与梯度尺度

### 7.1 首轮保持原训练目标

定义 `L[d,e]` 为一个数据来源卡的本地 LM mean loss 加本地 router aux loss，TP replicas 重复计算同一个标量。目标为：

```text
L = (1 / (D*P)) * sum_d sum_e L[d,e]
```

首轮沿用现有 trainer 的“本地有效 token 平均，再对数据来源平均”语义；当不同来源有效 labels 数量不同，它不等价于所有有效 token 的全局平均。等价测试应显式按上述目标逐来源计算，不能盲目把 batch 拼起来只调用一次 mean CE。

后续可新增全局 token 加权模式：先获取有效 token 总数，再对每个本地 LM loss 加权；aux loss 使用独立、显式的权重规则。该模式会改变当前目标，不作为隐式修复混入首轮。

### 7.2 非专家与专家为什么缩放不同

反向从每个来源的 `L[d,e]` 开始，因此：

- 非专家参数只看到本地来源的梯度，需要跨 `(d,e)` 平均。
- 某个专家在 owner 卡收到所有 e 的路由计算，其梯度已经是 `sum_e`，随后跨 D 平均，再除 P。
- 不同 e 拥有不同专家，绝不能跨 EP 对专家参数 all-reduce。
- 来源 hidden 的 all-to-all 梯度不除 P，否则上游 Dense 参数再平均会多除一次。

无 FSDP 时最终规则：

| 参数类别 | optimizer 前操作 |
| --- | --- |
| Dense TP shard | 对相同 t 的 `(d,e)` replicas 平均 |
| Dense TP-replicated 参数 | 对 `(d,e)` 平均，不重复平均 t |
| Q/K Norm | 先 TP SUM，再对 `(d,e)` 平均 |
| router | 对 `(d,e)` 平均 |
| 本地专家 TP shard | 同一 e,t 跨 D 平均，然后乘 `1/P` |

P=1 退化为通常的数据并行；D=1 的 EP 也仍然需要专家梯度除 P。这项缩放在 accumulation 结束后对专家参数梯度做一次，不嵌入每次读取 weight 的 backward，也不修改专家输入梯度。

### 7.3 auxiliary loss

默认 `aux_loss_scope=local`：直接保留每个来源的 `mean_probs` 与 `load` 定义。这使 EP 分发只改变执行位置，不改变当前训练目标。

全 EP batch 的均衡损失是另一个可选目标，不能把各卡局部 aux 标量平均当作它：

```text
p_global = sum_e(sum_token router_probs[e]) / total_tokens
load_global = sum_e(assignment_counts[e]) / (K * total_tokens)
aux_global = N * coef * sum(p_global * load_global)
```

若未来开放该模式，需同时定义跨卡可微概率聚合或等价局部 surrogate：用 detached 的 global load、每来源的概率和，以及与最终数据平均一致的缩放。只对 detach 后的最终标量 all-reduce 会丢失 router 训练梯度。本轮只实现 local 模式，global 模式在配置中拒绝。

## 8. MoE 的 TP×EP 与 Triton 兼容

### 8.1 每个专家再做 TP

EP 决定专家归属，TP 切该卡所拥有专家的中间维。每个 `(d,e)` 的 TP 卡收到相同 token 和 expert 分段，分别计算专家的本地中间通道，再对 down 输出做 TP SUM。

首轮每个 t 都沿自己的 EP 组做 all-to-all。这会产生 T 份相同 hidden 通信，但简化了本地 kernel 接入和反向约定；模型样本与专家梯度不能因此再乘 T。后续可优化为 TP leader 通信或序列分片，但那需要新的布局转换和反向设计。

路由在 TP replicas 间必须一致，包括 Top-K 顺序及接收排序。调试模式比较 route/count 摘要；可由 TP leader 广播离散选择，各卡用相同 indices gather 自己的可微 router 概率。专家计算前确认所有 t 的 token 顺序相同，否则 TP SUM 会混合不同 token。

### 8.2 fused gate/up 的精确切片

令每个 t 的通道区间为 `[a:b]`，`b-a=F/T`。原权重：

```text
gate_up: [N, 2F, H]，axis 1 排列为 [全部 gate, 全部 up]
down:    [N, H, F]

local_gate_up = cat([
    gate_up[expert_start:expert_end, a:b, :],
    gate_up[expert_start:expert_end, F+a:F+b, :]
], dim=1).contiguous()
local_down = down[expert_start:expert_end, :, a:b].contiguous()
```

本地 grouped 模块的 `num_experts=N/P`、`d_ff=F/T`、投影 in/out_features 都需要对应更新。导出时反过来分别拼 gate/up，再恢复融合布局，不能按 rank 直接拼整个 local_gate_up。

### 8.3 内核处理原则

| 已有实现 | 并行影响 | 首轮处理 |
| --- | --- | --- |
| RMSNorm / RoPE | 输入 local heads 或完整 hidden | 保留；Norm 参数梯度通信由并行层处理 |
| SwiGLU gate | 中间通道缩小，gate/up 可能为非连续行视图 | 保留并检查实际 stride 要求；必要时显式 contiguous |
| FlashAttention | head 数缩小，head_dim 不变 | 优先保留；测试本地 shape 与前后向 |
| Variable-M grouped GEMM | 只看本地 experts 和 offsets | Torch 基线后接回原 kernel，使用本地 expert ids |
| Top-1 dispatch/combine | 当前依赖本卡的一一 token 排列 | EP 的跨卡排序先用 Torch；原 kernel 仅用于满足置换契约的本地步骤 |
| Top-K combine | 同 token 多条 assignment | 用可微加权 index_add；不复用只写一次的 Top-1 combine |
| fused CE | logits 保持完整 | 保留 |
| AdamW / squared_norm_partials | Parameter 变成本地 shard | 保留本地计算；范数的跨组去重另行处理 |

当前 grouped GEMM 的 forward 使用 Triton，backward 已经按专家循环调用 Torch matmul，并通过 `expert_offsets.detach().cpu()` 读 offsets。它可作为 EP 本地计算起点，但包含主机同步，不能预先宣称通信已与计算重叠。

`expert_backend=torch|triton|auto` 仅控制并行 MoE 专家适配路径；不删除或整体禁用其他 Triton 模块。显式 triton 模式不支持某条件时给出明确错误；auto 模式可按已知能力回退并记录原因，不能捕获任意异常后静默回退。回退选择要在相关 ranks 间一致。

## 9. 与现有 FSDP 组合

### 9.1 统一控制转换顺序

```text
构建完整模型 / 载入完整模型权重
  → 同步原始参数和 buffers，保留参数别名
  → 生成并校验完整并行计划
  → EP 选取本地专家 + TP 选取本地通道
  → 可选沿 D 对本地逻辑参数进一步 FSDP 分片
  → 应用受控 activation checkpointing
  → 构建 optimizer
```

TP 的计算分片必须先于 FSDP 的存储分片。FSDP gather 只恢复“当前 e,t 的完整本地计算权重”，不恢复全局所有专家或所有 TP 通道。外部不鼓励手动多层嵌套 wrapper，统一由 `wrap_parallel` 编排。

初始化首轮允许完整模型在 CPU 上创建、同步/加载后分片再迁移，避免每卡 GPU 暂时容纳整个原模型；大模型 meta/sharded 初始化留作后续。明确记录初始化峰值，不仅测稳定训练显存。

### 9.2 对 distributed.py 的必要改造

1. `_world_size`、`_rank`、broadcast/gather/reduce-scatter helpers 接受 group；broadcast 的全局 src 与 group 内 rank 显式转换。
2. `_ShardInfo` 保存逻辑参数名、全局形状、EP/TP 布局与 D 分片映射，不依赖新模块名猜测布局。
3. `_is_sharded_module` 改为已登记参数/模块策略，覆盖并行 Linear 与 grouped adapter，防止继承关系造成误匹配。
4. 原 `finish_gradient_synchronization()` 的统一 WORLD 平均改为参数分类规则。
5. `_AllGatherWeight.backward` 仅负责 D 维 reduce-scatter average；Q/K Norm TP SUM、Dense EP average、专家 `1/P` 归并到统一 finalize。
6. checkpoint 导出/载入分层重建：D 恢复本地计算权重，TP 恢复通道，EP 恢复专家。

为了让第一版组合易于证明，FSDP 对所有可分片参数统一只沿 D 切存储。Dense 因此仍在 e 维复制，较极致方案多占空间，但避免不同参数使用不同 FSDP mesh。

使用 FSDP 时：Dense 已经跨 D reduce-scatter average，再对相同 d,t 的 e shard 做 average；专家已跨 D average，仅再除 P；复制的 Norm/embedding/router 则按其元数据完成 D、EP 同步。相同 e 组参与 Dense 同步时 shard 形状和逻辑 slice 必须一致。

### 9.3 动态专家与 collective 次序

现有非 checkpoint FSDP 按模块注册顺序预取，不适合直接覆盖动态路由的 Top-K 专家调用。不同 D 副本可能使用不同专家，若只在“有 token”时触发 gather，会导致 collective 不匹配。

第一版组合禁用专家路径的跨模块预取，按全局专家编号的确定性顺序执行必要 gather/compute/release；空分支仍参加所需通信。Top-K 的 grad=None 语义通过使用标记与最终梯度状态处理，不能靠省略 collective 实现。grouped Top-1 为固定两个投影，仍须处理整卡空输入及重计算。

TP/EP×FSDP 在这项测试通过前明确拒绝配置，不能仅新增一个 `process_group` 参数就宣称组合成功。

## 10. 训练接口及公共协议

建议新增接口，名称在实现阶段可小幅调整：

```python
context = ParallelContext.from_distributed(
    tp_size=2, ep_size=2, dp_size=2,
)
model = wrap_parallel(
    base_model,
    context=context,
    config=ParallelConfig(
        data_sharding="replicated",  # 或 "fsdp"
        expert_backend="torch",     # 基线先行
        aux_loss_scope="local",
    ),
)
# 必须在 wrap 之后构建 optimizer。
trainer = Pretrainer(model, dataloader, device, train_config, ...)
trainer.train()
```

`ParallelModel` 提供 `forward`、`config`、`auxiliary_loss()`、`parallel_context`、`finalize_gradients()`、`clip_grad_norm_()`、`full_state_dict()`、`load_full_state_dict()`。`parameter_count()` 区分全局逻辑参数量、本地驻留参数量，避免把分片数量打印为模型总大小。

MoE adapter 优先保持 `MoELayer` 的类型兼容；wrapper 明确聚合每层当前 forward 的 aux_loss，不能因替换类型导致原 `isinstance` 过滤器漏算。不能用通用 `__getattr__` 无限代理掩盖缺失接口。

训练基础设施使用同一协议访问普通模型、旧 FSDP 及新并行模型，逐步替代多处硬编码 `isinstance(FullyShardedDataParallel)`。

### 10.1 一个 optimizer step 的固定顺序

1. `zero_grad`。
2. 对每个 accumulation microbatch：同步 TP 输入，前向、取得本次 aux loss、按 accumulation 次数缩放 loss、backward。
3. 最后不足完整 accumulation window 时保留现有校正。
4. `finalize_gradients()`：确定性顺序完成未在 backward 完成的参数同步、专家归一化和 grad presence 协调；每 step 只执行一次。TP/EP 激活梯度通信不能因 accumulation 延后。
5. AMP unscale；扫描归约后梯度的非有限值并 WORLD MAX 达成共识。
6. 无溢出时计算去重全局范数并裁剪、所有 rank 更新；有溢出时所有 rank 都跳过更新。
7. 同步更新 loss scale/growth tracker，清除该 step finalize 标志。

finalize 放在 unscale 前可以让 unscale 检查同步产生的非有限值，但所有 rank 必须先保持相同 loss scale。还需处理只在其他分片出现的溢出；不能直接让各 rank 独立调用 `GradScaler.step()`。拟新增小型并行 AMP 控制器，使用明确的全局有限值判断及 scale 状态管理，避免依赖未文档化私有字段。先验证 FP32/BF16，再开放 FP16。

### 10.2 梯度范数

当前函数将所有本地参数平方和 WORLD SUM，复制参数会被重复计数；混合并行必须根据逻辑参数元素去重。

通用定义：每个逻辑梯度元素只选一个 owner 对 norm 平方和贡献，分片的不同元素全部计入。例如：

- 无 FSDP 的 Dense TP shard 选 `d=0,e=0`，遍历所有 t。
- 无 FSDP 的专家 TP shard 选 `d=0`，遍历所有 e,t。
- 在 TP 复制的 Norm/router/embedding 只选 `t=0`；Dense 再选 `e=0,d=0`。
- 有 D-FSDP 的参数遍历 D 的不同 shards，不再只取 `d=0`；Dense 仍在 e 维去重。
- tied Parameter 按别名只计一次；padding 元素不计入。

每卡累加被分配给自己的 FP32 partials，WORLD SUM 得到完整逻辑 norm，所有梯度使用同一 clip scale。日志同时测试该 norm 与未并行参考相等。

### 10.3 RNG、评估与日志

权重先同步再切；运行时 TP replicas 的 residual dropout mask 一致，不同数据来源使用独立 RNG 流。当前 `seed+global_rank` 必须替换为按数据坐标派生的 seed。当前 Attention 虽存储 dropout 值，但 forward 未将其用于 attention 权重 dropout；本方案不顺带改变此行为。

checkpoint recomputation 必须恢复对应 RNG，保持 routing 与 collective 顺序。第一版 EP 不做包含跨卡通信的整个 MoE checkpoint，只对本地无通信计算段使用 checkpoint；扩大重计算范围前要测试 aux_loss 副作用，防止 backward 重算覆盖 forward 留下的 loss 引用。

评估、tokens/s 只计 `t=0` 的独立数据来源，再合并 d,e。验证 sampler 的补齐重复样本必须有 mask 或使用等长带标记的 padding，避免重复统计；同时保持所有相关 ranks 相同前向次数，不能简单使用不等长 sampler 导致 collective 挂起。训练首轮保留 DistributedSampler 的补齐语义并写入配置。

吞吐报告有效 tokens 总数 / 最慢 rank 的同步耗时，区分有效 label tokens 与输入 tokens。EP 路由量统计按全局 expert id 合并，TP replicas 去重；记录最大/平均 token 数、空专家比例、通信耗时与峰值显存。

## 11. Checkpoint 设计

### 11.1 两种用途

- 可移植完整模型导出：使用原模型 key 和 shape，可载入未 wrap 的模型；允许改 TP/EP 拓扑重新切分模型权重。
- 训练恢复：保存 rank-local optimizer、模型分片映射、AMP、CPU/CUDA/Python RNG、训练 epoch/batch 位置、sampler seed 等；首轮仅保证相同完整拓扑恢复。

仅 `world_size` 相等不足以恢复 optimizer，例如 8 卡 TP=4×DP=2 与 TP=2×EP=2×DP=2 的 shard 身份不同。加载要检查 D/P/T、rank 映射、参数布局、模型配置、dtype/optimizer 格式版本；不一致时在任何 load collective 前统一报错。

### 11.2 拼接规则

| 参数 | 完整模型重建 |
| --- | --- |
| 普通 TP column | 沿权重 axis 0 拼接 |
| 普通 TP row | 沿权重 axis 1 拼接 |
| grouped gate_up | 分别恢复 gate 与 up 的通道，再拼融合维，最后恢复专家维 |
| grouped down | 恢复输入通道，再恢复专家维 |
| Top-K experts | 按 global expert id 恢复原 ModuleList key |
| Norm/router/embedding | 验证副本一致后取一份；保留 tied 别名语义 |
| D-FSDP shards | 先去 padding 恢复当前 e,t 的局部权重 |

需要 collective 的导出由全部参与 rank 调用，不能仅在 `if rank == 0` 内调用；最终只由指定 rank 写完整模型。逐参数转 CPU/写入，避免在每张 GPU 同时堆放全量模型。

各 rank 文件完成后先汇报成功，再原子发布 manifest/完成标记。恢复前统一检查文件完整性，避免一张卡读文件失败而其余卡进入通信等待。旧 checkpoint 保留读取路径；“只加载模型”与“恢复训练”提供不同入口。

## 12. 拟修改文件与工作拆分

| 文件（新增项为计划路径） | 职责 |
| --- | --- |
| 新增 `training/parallel/context.py` | D/P/T 拓扑、groups、rank 映射、配置预检 |
| 新增 `training/parallel/collectives.py` | TP copy/reduce、可微 all-to-all、CPU 测试后端 |
| 新增 `training/parallel/layouts.py` | 参数身份、切片、同步组、norm owner、checkpoint 映射 |
| 新增 `training/parallel/tensor_parallel.py` | Linear/SwiGLU/GQA/grouped experts TP 转换 |
| 新增 `training/parallel/expert_parallel.py` | 本地专家拥有权、路由分发与合并 |
| 新增 `training/parallel/wrapper.py` | 统一转换、公共协议、梯度 finalize |
| 新增 `training/parallel/amp.py` | 全局溢出判断、同步 loss scale 和恢复 |
| 修改 `training/parallel/fsdp.py` | group-aware FSDP、布局集成、动态专家安全调度 |
| 修改 `training/pretrainer.py` | 通用协议、step 顺序、并行 clipping/logging |
| 修改 `workflows/pretrain.py` | 并行配置、sampler、初始化与 wrap |
| 修改 `training/checkpoint.py`、`evaluation.py` | 并行保存恢复、指标去重 |
| 按需小幅修改 `modeling/` | 抽取可复用计算方法或协议；保留原始单卡与 Triton 路径 |
| 新增 `tests/test_tensor_parallel.py`、`test_expert_parallel.py`、`test_parallel_training.py` | 分层验证与训练恢复 |
| 新增 `configs/pretrain_*_tp*.yaml`、`pretrain_*_ep*.yaml` | 4/8 卡示例；不覆盖原始配置 |

所有实现改动均在文档审阅后开始。若模块转换可以完全放在 `training/parallel/`，优先不改原模型类；但不以保持源码逐字不变为代价复制大量容易漂移的 Attention 逻辑。

## 13. 配置与服务器启动示例

以下为未来配置片段，需合并现有数据、tokenizer、输出目录与训练参数；当前入口尚不支持 `parallel`。

```yaml
parallel:
  tp_size: 2
  ep_size: 2
  dp_size: 2
  data_sharding: replicated   # FSDP 组合验收后可设 fsdp
  expert_backend: torch      # 基线通过后选择 auto / triton
  aux_loss_scope: local
  activation_checkpointing: false
  debug_collectives: false
```

原有 `use_fsdp` / `fsdp_config` 保持兼容：仅旧配置时映射到 `T=P=1,D=W`；同时提供新旧配置而含义冲突则报错，不能静默覆盖。

| 卡数 | 模型 | D | P | T | 验证重点 |
| --- | --- | --- | --- | --- | --- |
| 4 | Dense | 1 | 1 | 4 | 单个 batch 的 TP 正确性 |
| 4 | Dense | 2 | 1 | 2 | TP + 复制 DP / FSDP |
| 4 | MoE | 1 | 4 | 1 | 每卡一个专家的纯 EP |
| 4 | MoE | 1 | 2 | 2 | TP×EP，包含专家中间维切分 |
| 8 | Dense | 2 | 1 | 4 | TP + 复制 DP / FSDP |
| 8 | MoE | 2 | 2 | 2 | 三维组合、梯度尺度、恢复 |
| 8 | MoE | 1 | 4 | 2 | 专家分配到四组、组内 TP |

拟提供可直接运行的完整 YAML 后，再使用：

```bash
# 未来示例文件，当前尚未创建。
torchrun --standalone --nproc_per_node=4 -m ajllm.workflows.pretrain \
  --config configs/pretrain_moe_tp2_ep2.yaml

torchrun --standalone --nproc_per_node=8 -m ajllm.workflows.pretrain \
  --config configs/pretrain_moe_dp2_tp2_ep2.yaml
```

首轮单机一进程一 GPU。服务器启动前记录 `nvidia-smi topo -m`、torch/CUDA/NCCL/Triton 版本以及实际设备型号。TP 放在互联较快的相邻设备；EP all-to-all 同样依赖网络带宽，性能结论以服务器实测为准。

## 14. 本地与服务器验证计划

### 14.1 当前会话实际能力

本轮只做环境读取和代码审查，未进行并行实现或训练测试。项目 `.venv/bin/python` 报告：

```text
torch: 2.11.0+cu130
CUDA build: 13.0
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
Gloo compiled support: True
nvidia-smi: GPU access blocked by the operating system
```

用户的物理机器有一张 3080 Ti，但当前执行环境看不到可用 GPU；这不代表物理设备不存在。系统 `python3` 未安装 torch，后续验证应使用项目虚拟环境。Gloo 编译可用不等于多进程端口权限与所有 collective 已验证，实施时还需 smoke test。

即使 GPU 访问恢复，一张卡也不能验证真实多 GPU NCCL 拓扑、带宽和显存收益。不会将多个进程绑同一张 GPU 当作多 GPU 验收。

### 14.2 CPU 数学与多进程基线

先使用微型模型：`H=32,Hq=4,Hkv=2,F=64,N=4`，小词表、短序列、dropout=0、关闭 CUDA kernels，CPU attention 使用可行的 Torch 路径。必要时关闭 compile，避免编译开销遮盖通信问题。

| 层级 | 必测项目 | 通过条件 |
| --- | --- | --- |
| 无进程组 | T=P=D=1 wrap 退化、模型 keys/共享权重 | forward、grad、一步 AdamW 与原模型一致 |
| 切片单测 | GQA head 映射、fused gate/up 分段、专家编号、重建 | 数值标记权重 round-trip 完全一致 |
| 2/4 CPU processes | TP Linear、SwiGLU、GQA、Q/K Norm | 拼回 dW、dX、logits 与参考相等 |
| 2/4 CPU processes | EP Top-1、Top-K | 所有来源 output/dX、专家 grad、router aux grad 与参考相等 |
| 4 CPU processes | T=2,P=2,D=1 | 先 TP、再 EP 的完整 MoE block/模型等价 |
| 8 CPU processes（资源允许） | T=2,P=2,D=2 | 梯度缩放、参数 ownership、optimizer 更新等价 |
| FSDP 组合 | 上述组合加 D 分片 | 无 gather 次序错位、local shard 与参考对应 |
| 恢复 | 连训数步 vs 保存后同拓扑继续 | 模型、optimizer、step、RNG/AMP 状态一致 |

collective 设合理 timeout，子进程失败能终止整组；给每个 collective 加可选序号/层名/组名诊断，避免测试永久挂起。若测试后端缺少 reduce-scatter，可用 all-reduce + slice 的数学替代验证，但生产 NCCL 路径仍须单独验收。

### 14.3 必须覆盖的反例

1. 所有 token 路由到同一专家；其他 ranks 接收 0 行。
2. 单个来源发往某个 owner 的 split 为 0；全空本地专家集合输入。
3. Top-K 同 token 多路及重复 token 梯度累加；K=1 的主损失 router grad 应符合原语义。
4. `qk_norm=True`，确认 TP SUM 后等价；禁用它的测试不能替代该测试。
5. 非均匀有效 label 数量、全 `-100` labels、尾部 accumulation window。
6. 有些 DP 副本专家未使用，有些已使用；Top-K 全局未使用专家的 AdamW skip 行为。
7. 连续多个 microbatches、多个 steps，验证 finalize 不重复缩放已有累积梯度。
8. dropout>0 时 TP 副本一致；checkpoint 重算路由一致。
9. 一个 shard 注入 Inf/NaN，所有 ranks 同步跳步并更新 scale。
10. tied / untied embeddings 导出、旧模型权重载入、相同 world size 不同拓扑恢复应拒绝。
11. 默认模型 TP=8、N 无法整除 P、重复 wrap、未知模块、缺失 checkpoint 分片应清晰失败。
12. EP 不属于本 rank 的专家参数确实不在 optimizer 中；本地参数量符合计划。

参考执行：复制相同完整权重，按每个 `(d,e)` 的 batch 独立前向，按第 7 节定义构造参考目标；并行输出与对应来源比较，梯度先恢复为全局逻辑布局再比较。路由固定/离散选择对齐后再检查连续梯度，避免把 argmax 邻界变化误判为通信问题。

FP32 初始比较阈值可取 `atol=1e-5,rtol=1e-4`；低精度按具体 kernel 误差与原单卡误差另设阈值，同时记录最大绝对/相对误差。不能为了通过测试无依据地放宽阈值。

### 14.4 单卡 Triton 与多卡 NCCL

GPU 可访问时，一张 3080 Ti 可以验证：local TP shard shape 的原 kernel、local expert_count=1/2 的 grouped GEMM、空专家保护、Torch/Triton 前后向，以及单位并行规模的完整训练 smoke。数值比较必须使用同一模型、权重、batch 与路由；Top-K 临界 tie 导致路径不同需单独记录。

服务器 4/8 卡必须验证：真实 variable-split/all-zero NCCL、完整 TP/EP/FSDP 数值测试、FP16 溢出一致性、周期评估、保存恢复、短程训练（如 100 steps）无挂起或 NaN。先 Torch 后 Triton，先不 checkpoint 再启用受支持的本地计算 checkpoint。

最终报告将“CPU 数学通过”“单卡 kernel 通过”“4/8 卡 NCCL 通过”逐项列明；未运行项明确标为未验证。

## 15. 性能目标与验收顺序

粗略参数存储关系（忽略复制参数和 padding）：

```text
Dense TP 参数：约 1/T
专家 TP×EP 参数：约 1/(T*P)
叠加 D-FSDP：上述可分片参数再约 1/D
```

这些不是峰值总显存比例。完整词表、residual activations、EP 收发 buffers、最繁忙专家输入、FSDP gather 临时权重与 checkpoint 导出都会改变峰值。

性能测量固定相同有效 batch、序列长度、精度与训练目标，预热后测同步 wall time、有效 tokens/s、每卡峰值 allocated/reserved 显存、TP/EP/FSDP 通信时间、专家负载。小模型或 PCIe 环境可能因通信变慢，第一目标是正确与可扩展，不预设倍数加速。

| 阶段 | 交付 | 开放条件 |
| --- | --- | --- |
| A | Context、参数布局、collectives、训练协议、CPU harness | 通信反向和组定义通过 |
| B | Dense TP + 复制数据同步 | GQA/QK Norm/梯度裁剪/一步 optimizer 等价 |
| C | EP Torch Top-1/Top-K | 变长/空路由、aux、专家缩放通过 |
| D | MoE TP×EP | fused 切片、路由一致性、梯度等价通过 |
| E | 本地 Triton、AMP、评估与 checkpoint 完整链路 | kernel 对照、溢出同步、恢复通过 |
| F | TP/EP×FSDP 与受控 checkpoint | 动态专家 collective 次序、D shard 同步通过 |
| G | 4/8 卡验收与性能报告 | 实际服务器 NCCL 及短训通过 |

后续实现完成的最低交付是 A–F 的代码、对应测试与完整配置，明确报告本地能够运行的验证结果；G 需要实际多卡服务器。文档审阅时主要确认三个设计取舍：首轮词表复制、EP 保留本地 auxiliary loss、FSDP 只沿 D 维切存储。它们均优先保持当前训练语义和可审查性，后续优化可以在验证基线上继续进行。
