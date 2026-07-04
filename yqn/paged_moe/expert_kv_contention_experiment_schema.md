# MoE 模型推理中 HBM KV Cache 与 Expert Weights 容量争用实验方案

---

## 1. 研究目标

证明 MoE 模型在单 GPU 推理过程中，HBM 里 KV cache 和 expert weights 之间存在内存容量争用，且这种争用通过 preemption 机制导致 attention 和 expert 计算时间相互影响、非线性退化。

---

vllm的缺陷：
1.HBM中expert weights和kv cache的占用比例是静态的，没发根据系统的实际负载进行调整
2.model weights offload以层为单位进行offload和prefetch，容易拿上来很多没有被命中的expert weights还浪费了pcle带宽
  以层为单位的load store过于粗糙：
  prefetch_step 小:
  + buffer pool 小，KV cache 大
  - 传输可能来不及，wait/stall 高
  prefetch_step 大:
  + transfer 更容易 overlap
  - buffer pool 占 HBM，KV cache 变小，可能增加 preemption

## 2. 争用机制

### 2.1 内存关系

在固定大小的 GPU HBM 中：

```
可用 HBM = 总 HBM × gpu_memory_utilization
可用 HBM = backbone weights + expert weights(常驻) + prefetch buffer pool + KV cache

Expert 占得多 → KV cache 少 → 承载并发请求少 → 更早触发 preemption
Expert 被 offload → 每步 expert 计算慢 → 请求存活更久 → KV 占用时间更长 → 加剧 preemption
```

### 2.2 耦合路径

两者的耦合不是同一步内的带宽争用，而是 **跨步的占用时间放大**：

- KV cache 不足 → preemption → 请求被重新 prefill → attention 和 expert 都要重算 → 两者计算时间都增加
- Expert 在 CPU 上 → 每步 decode 慢 → 请求存活更久 → 占 KV block 时间更长 → 其他请求更容易被 preempt

### 2.3 vLLM v1 Preemption 行为

vLLM v1 引擎的 preemption 策略是 **recompute only**（无 swap）：

```python
# vllm/v1/core/sched/scheduler.py:929-949
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)      # KV blocks 直接释放（丢弃）
    request.num_computed_tokens = 0           # 重置为 0，下次完整重新 prefill
    request.num_preemptions += 1
    self.waiting.prepend_request(request)     # 放回等待队列
```

- **触发条件**：`allocate_slots` 返回 `None`（KV blocks 不够），scheduler 逐个 preempt 优先级最低的 running 请求，释放其 KV blocks，直到能分配为止。
- **代价**：完整重新 prefill（从 token 0 开始），需要重新过一遍所有 attention 层和所有 expert 层。

---

## 3. 模型选择

推荐 **Qwen/Qwen1.5-MoE-A2.7B**：

| 属性 | 值 |
|------|-----|
| 总参数量 | 14.3B |
| Active 参数量 | 2.7B |
| Expert 数/层 | 60 |
| Top-k | 2 |
| Expert weights 占比 | ~70% (~20GB FP16) |
| Backbone weights | ~8GB FP16 |
| 总模型大小 (FP16) | ~28GB |

选择理由：

- 60 experts/layer，expert 数量多，offload 粒度可调
- 单个 expert 体积小（~18MB），适合观测不同 offload 比例的差异
- backbone 不大，单卡能装下，留有空间做 KV cache 与 expert 的权衡

---

## 4. 实验设计

### 4.1 独立变量

本实验采用 **二维扫描**，两个独立变量：

| 自变量 | 含义 | 控制方式 |
|--------|------|---------|
| **R** (offload 比例) | expert weights 在 HBM 中的常驻比例，决定了 KV cache 与 expert 的静态 HBM 分配 | `--offload-group-size` / `--offload-num-in-group` |
| **N** (并发请求数) | 请求到达速率，决定了运行时 KV cache 的实际占用压力 | `benchmark_serving.py --request-rate` |

- R 控制的是"静态配置层面的分配比例"
- N 控制的是"运行时负载压力"
- 二者共同决定系统的性能表现

### 4.2 二维扫描矩阵

使用 `--gpu-memory-utilization 0.5` 将可用 HBM 限制到 ~40GB，使争用在合理并发下可见。

**R 维度（offload 比例，5 个配置）：**

| 配置 | offload 比例 | Expert 状态 | HBM 估算 | KV cache 容量 |
|------|-------------|------------|---------|--------------|
| R0 | 0% | 全部常驻 HBM | backbone 8 + experts 20 + KV 12 GB | ~12 GB |
| R25 | 25% 层 offload | 3/4 常驻 | backbone 8 + experts 15 + buffer 1 + KV 16 GB | ~16 GB |
| R50 | 50% 层 offload | 1/2 常驻 | backbone 8 + experts 10 + buffer 2 + KV 20 GB | ~20 GB |
| R75 | 75% 层 offload | 1/4 常驻 | backbone 8 + experts 5 + buffer 2 + KV 25 GB | ~25 GB |
| R100 | 100% 层 offload | 全部在 CPU | backbone 8 + buffer 2 + KV 30 GB | ~30 GB |

**N 维度（并发请求数，8 个级别）：**

N = 1, 2, 4, 8, 16, 32, 64, 128

**总共 5 × 8 = 40 组实验。**

### 4.3 启动命令

```bash
# R0: 0% offload
vllm serve Qwen/Qwen1.5-MoE-A2.7B \
  --gpu-memory-utilization 0.5 \
  --dtype float16

# R25: 25% offload (每 4 层 offload 1 层)
vllm serve Qwen/Qwen1.5-MoE-A2.7B \
  --gpu-memory-utilization 0.5 \
  --dtype float16 \
  --offload-group-size 4 --offload-num-in-group 1 \
  --offload-prefetch-step 2 \
  --offload-params w13_weight w2_weight

# R50: 50% offload (每 4 层 offload 2 层)
vllm serve Qwen/Qwen1.5-MoE-A2.7B \
  --gpu-memory-utilization 0.5 \
  --dtype float16 \
  --offload-group-size 4 --offload-num-in-group 2 \
  --offload-prefetch-step 2 \
  --offload-params w13_weight w2_weight

# R75: 75% offload (每 4 层 offload 3 层)
vllm serve Qwen/Qwen1.5-MoE-A2.7B \
  --gpu-memory-utilization 0.5 \
  --dtype float16 \
  --offload-group-size 4 --offload-num-in-group 3 \
  --offload-prefetch-step 2 \
  --offload-params w13_weight w2_weight

# R100: 100% offload (每层都 offload)
vllm serve Qwen/Qwen1.5-MoE-A2.7B \
  --gpu-memory-utilization 0.5 \
  --dtype float16 \
  --offload-group-size 1 --offload-num-in-group 1 \
  --offload-prefetch-step 2 \
  --offload-params w13_weight w2_weight
```

### 4.4 Benchmark 负载

对每个 R 配置，扫描所有 N 值：

```bash
for N in 1 2 4 8 16 32 64 128; do
  python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model Qwen/Qwen1.5-MoE-A2.7B \
    --dataset-name sharegpt \
    --request-rate $N \
    --num-prompts 200 \
    --save-result \
    --result-dir results/R${R_LABEL}_N${N}
done
```

### 4.5 分析策略

实验完成后，按以下顺序分析：

**第一步：全景热力图**

以 R 为 x 轴、N 为 y 轴，绘制 throughput / E2E_latency / preemption_rate 的二维热力图，展示全貌。

```
          R0    R25    R50    R75    R100
     ┌─────┬─────┬─────┬─────┬─────┐
N=1  │  ●  │  ●  │  ●  │  ●  │  ●  │
N=4  │  ●  │  ●  │  ●  │  ●  │  ●  │
N=16 │  ●  │  ●  │  ★  │  ●  │  ●  │  ← ★ = throughput 峰值
N=64 │  ●  │  ●  │  ●  │  ●  │  ●  │
N=128│  ●  │  ●  │  ●  │  ●  │  ●  │
     └─────┴─────┴─────┴─────┴─────┘
```

热力图能直观展示：对于每个 N，最优的 R 不同；对于每个 R，能承载的最大 N 不同。

**第二步：找到争用最严重的 (N, R) 区域**

选取标准：

- 找到 throughput 下降最陡的区域（性能退化的"悬崖"）
- 找到 preemption 刚开始频繁触发的 N 值（从 0 到非 0 的拐点）
- 对比相邻 R 配置在同一 N 下的差异最大的点

**第三步：深入分析争用最严重的点**

针对选出的 (N\*, R\*) 组合，展示细粒度指标：

- T_attn_per_request 和 T_expert_per_request 的分布
- preemption 次数分布
- 单请求时间线（何时被 preempt、何时恢复、重算了多少）

**第四步：对比论证**

在同一个 N\* 下，对比所有 R 配置的 T_attn 和 T_expert：

- 证明 R 增大时 T_expert 上升（offload 直接代价）但 T_attn 因 preemption 减少而下降
- 证明 R 减小时 T_expert 下降但 T_attn 因 preemption 增加而上升
- 两者不可兼得 → 争用存在

---

## 5. 观测指标

### 5.1 核心指标（请求粒度）

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| `T_attn_per_request` | 单个请求从到达到完成，累计经过 attention 层的总计算时间 | hook attention forward，按 request_id 累加 |
| `T_expert_per_request` | 单个请求从到达到完成，累计经过 MoE 层的总计算时间 | hook MoE forward，按 request_id 累加 |
| `num_preemptions` | 单个请求被 preempt 的次数 | `request.num_preemptions`（scheduler 已记录） |
| `recompute_tokens` | 因 preemption 重新 prefill 的 token 数 | `num_preemptions × prompt_length` |

### 5.2 Cache Hit Rate 指标（争用的核心观测量）

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| `kv_cache_hit_rate` | KV prefix cache 命中率。反映 KV cache 容量是否足够复用已有前缀，避免重新 prefill | scheduler 中已有 `PrefixCacheStats.num_hits / num_queries`，可直接读取 |
| `expert_cache_hit_rate` | Expert 权重 HBM 命中率。token 路由到的 expert 是否在 HBM 中（常驻层 = hit，offloaded 层 = miss） | 在 FusedMoE.forward() 中统计：每步 batch 中有多少 token-expert 对来自常驻层 vs offloaded 层 |

**定义：**

```
expert_cache_hit_rate (per step) =
    Σ(常驻层中的 token-expert 调用数) / Σ(所有层的 token-expert 调用数)

kv_cache_hit_rate (per request) =
    prefix cache 命中的 token 数 / 请求总 prompt token 数
```

**争用关联：**

```
R 低（expert 多在 HBM）:
  expert_cache_hit_rate 高 → expert 计算快
  但 KV cache 小 → kv_cache_hit_rate 低 → 更多 prefill 重算 → TPOT ↑

R 高（expert 多在 CPU）:
  expert_cache_hit_rate 低 → expert 计算慢 → TPOT ↑
  但 KV cache 大 → kv_cache_hit_rate 高 → 少 preemption


两个 hit rate 此消彼长，TPOT 受两者共同影响
```

### 5.3 系统级指标

| 指标 | 含义 |
|------|------|
| `TTFT` | Time to First Token（首 token 延迟） |
| `TPOT` | Time Per Output Token（每 token 生成延迟） |
| `E2E_latency` | 请求端到端延迟 |
| `throughput` | 系统吞吐量 (tokens/s) |
| `kv_utilization` | KV cache block 使用率 |

### 5.4 插桩位置

```
1. Attention 计时:
   vllm/attention/ 中的 attention forward 方法
   在 forward 前后记录 torch.cuda.Event，按 request_id 累加

2. Expert 计时 + expert hit/miss 统计:
   vllm/model_executor/layers/fused_moe/layer.py 中 FusedMoE.forward()
   - 前后记录 torch.cuda.Event，按 request_id 累加
   - 通过 layer 是否 offloaded 判定本次调用是 hit 还是 miss
   - 累加每步的 hit_count 和 total_count

3. Preemption 统计:
   vllm/v1/core/sched/scheduler.py:_preempt_request()
   request.num_preemptions 已有，可直接读取

4. KV cache hit rate:
   vllm/v1/core/sched/scheduler.py 中 PrefixCacheStats
   num_hits / num_queries

5. KV cache 使用率:
   vllm/v1/core/kv_cache_manager.py
   free_blocks / total_blocks
```

---

## 6. 预期结果与论证逻辑

### 6.1 图 1: Throughput 热力图 (R × N)

```
热力图: Throughput (tokens/s)，颜色越深吞吐越高

          R0     R25    R50    R75    R100
     ┌──────┬──────┬──────┬──────┬──────┐
N=1  │ ██▓  │ ██░  │ █░░  │ █░░  │ ░░░  │  ← 低 N: R0 最优（expert 快，KV 够用）
N=4  │ ███  │ ███  │ ██░  │ █░░  │ ░░░  │
N=16 │ ██▓  │ ███  │ ████ │ ██▓  │ █░░  │  ← 中 N: R50 最优（平衡点）
N=64 │ ░░░  │ █░░  │ ██▓  │ ████ │ ██▓  │  ← 高 N: R75 最优（KV 大，能撑住）
N=128│ ░░░  │ ░░░  │ █░░  │ ██▓  │ ██░  │
     └──────┴──────┴──────┴──────┴──────┘
              最优 R 随 N 向右移动 →
```

**论证要点**：热力图的"亮带"（最优区域）是一条从左上到右下的对角线，说明最优 HBM 分配比例随负载变化而移动。没有任何一列（固定 R）在所有行（N）上都最优 → 静态分配必然次优。

### 6.2 图 2: 固定 N\*，扫描 R → T_attn 与 T_expert 的此消彼长

选取争用最明显的 N\*（throughput 下降最陡的 N），固定 N\* 扫描 R：

```
时间 (ms)
  |
  |  T_expert                        T_attn
  |   per_req                        per_req
  |      /                        \
  |     /                          \
  |    /    ← offload 越多           \  ← preemption 越少
  |   /       expert 越慢              \   attn 重算越少
  |  /                                  \
  | /                                    \___
  +────────────────────────────────────────→ R
  R0                                      R100
```

**论证要点**：在同一负载 N\* 下，R 增大时 T_expert 单调上升（offload 直接代价），T_attn 单调下降（KV 大了 preemption 少了）。两条线存在交叉点——交叉点左侧 expert 是瓶颈（快但 KV 不够），右侧 KV 是瓶颈（够用但 expert 慢）。**两者不可兼得** 即争用的直接证据。

### 6.3 图 3: 固定 R，扫描 N → preemption 拐点不同

```
Avg preemptions per request

    R0          R50           R100
    |  /        |    /        |          /
    | /         |   /         |         /
    |/          |  /          |        /
    +──→ N      +──→ N       +──→ N
    N_0         N_50          N_100

    N_0 < N_50 < N_100 (preemption 触发点随 R 右移)
```

**论证要点**：

- R 越大 → KV 越大 → preemption 触发越晚（N\_0 < N\_50 < N\_100）
- 但 R 越大 → 每次 preemption 后的 recompute 代价越高（expert offloaded，重新 prefill 更慢）
- 在 N\_0 < N < N\_100 的区间内，不同 R 配置的性能差异最显著——这就是争用的观测窗口

### 6.4 图 4: 双 Cache Hit Rate 与 TPOT 的关联（核心论证图）

**视角 A：固定 N\*，扫描 R，三线叠加**

```
      ↑ hit_rate / TPOT (归一化)
  1.0 |
      |  kv_hit \_____                ___/ expert_hit
      |          \    \___    ___/   /
      |           \       \/       /
      |            \     / \     /           TPOT
      |             \   /   \   /          ___/
      |              \/      \/          /
      |          ← 交叉点 →           __/
  0.0 +──────────────────────────────────→ R
      R0                                R100
```

**论证要点**：

- `kv_cache_hit_rate` 随 R 增大而上升（KV cache 更大，prefix 复用更多）
- `expert_cache_hit_rate` 随 R 增大而下降（更多 expert 在 CPU）
- 两条 hit rate 线交叉，**TPOT 在交叉点附近取最小值**
- 偏离交叉点任何一侧，某一方的 miss rate 升高，TPOT 被拉高
- 这直接证明 TPOT 由两个 cache hit rate 共同决定，二者此消彼长

**视角 B：固定 R，扫描 N**

```
      ↑ hit_rate
  1.0 |
      | expert_hit ──────────────────  (固定 R，不随 N 变化)
      |
      | kv_hit ────────\
      |                 \____          ← N 增大后 KV 开始 miss
      |                      \____
  0.0 +────────────────────────────→ N
                              N*

      ↑ TPOT
      |                        /
      |                      /         ← kv_hit 下降拖动 TPOT 上升
      |                    /
      | ─────────────────/
      +────────────────────────────→ N
                              N*

  kv_cache_hit_rate 下降的拐点 N* ≈ TPOT 上升的拐点
  → 证明 kv cache miss 直接驱动 TPOT 退化
```

**论证要点**：在固定 R 下（expert hit rate 是常数），KV cache hit rate 随 N 增大而下降，TPOT 随之上升，两者拐点重合。这建立了 `kv_miss → TPOT ↑` 的因果链。结合图 6.2 中 `expert_miss → TPOT ↑` 的证据，可以得出结论：**TPOT 同时受两个 cache miss rate 驱动**。

### 6.5 分析与展示策略

1. **先展示热力图**（图 1）→ 让读者直观看到"没有最优静态配置"
2. **选出争用最严重的 N\***（throughput 退化斜率最大的 N）→ 深入分析
3. **在 N\* 处展示 T_attn vs T_expert 的此消彼长**（图 2）→ 证明两者不可兼得
4. **展示双 cache hit rate 与 TPOT 的关联**（图 4）→ 量化争用对延迟的驱动关系

---

## 7. 核心结论模板

> 实验表明，在固定 HBM 条件下，MoE 模型的 expert weights 和 KV cache 存在显著的容量争用：
>
> 1. **KV 约束反噬 expert**：在 R0 配置中，expert 权重全部常驻 HBM，单步 expert 计算时间恒定，但由于 KV cache 容量小，preemption 在 N > N\_0 后触发，请求被迫重新 prefill，T\_expert\_per\_request 出现非预期的跳升。这个跳升完全由 KV 容量不足引起，直接证明 KV 约束反噬了 expert 的有效计算效率。
>
> 2. **Expert 约束反噬 KV**：在 R100 配置中，KV cache 容量最大，理论上可支撑最高并发，但 expert offloading 使每步 decode 变慢，请求占用 KV block 时间更长，preemption 的实际触发点低于纯 KV 容量的理论上限。
>
> 3. **无最优静态划分**：Throughput 热力图的最优带沿对角线分布——低负载时 R0 最优，高负载时 R75/R100 最优——表明最优 HBM 分配比例是负载相关的，任何固定 R 都无法在全负载范围内最优。
>
> 4. **量化争用代价**：在争用最严重的 (N\*, R\*) 点，T\_expert 因 preemption recompute 增加了 X%，T\_attn 因 expert 延迟导致的 KV 长占用增加了 Y%，系统 throughput 相比理论上限损失了 Z%。

---

## 8. 后续方向

基于实验结论，后续工作方向：

1. **动态联合内存管理器**：实现一个统一的 HBM pool，让 KV blocks 和 expert weights 根据实时负载动态竞争分配
2. **Routing-aware expert cache**：基于 MoE router 的 topk_ids 统计 hot/cold expert，优先缓存热点 expert
3. **Active/Inactive 感知的 offload 策略**：识别 inactive KV blocks 和 cold experts，优先 offload 到 CPU
4. **联合调度算法**：在 scheduler 层面感知两类 cache 的压力，做出全局最优的 preemption/eviction 决策
