# MiniMind 大模型训练核心笔记

## 模型架构 (Model Architecture)
CausalLM (总架构)
```text
└── Model (骨架)
    ├── Embedding (词嵌入层)
    ├── Dropout (随机丢弃层，可选)
    ├── Blocks (Transformer 核心层，重复堆叠 N 层)
    │   └── Block (单层内部结构)
    │       ├── Pre-Att RMSNorm ───> Attention (MHA,GQA...) ──> 残差相加 (+)
    │       └── Pre-MLP RMSNorm ───> MLP (SwiGLU,MoE...) ──> 残差相加 (+)
    └── Final RMSNorm (尾部归一化)
└── LM Head (输出映射头)
```
## 预训练Pretrain

### 1 Dataset
*   **原始格式**：`{"text": "用户输入或语料...<im_start>...<im_end>"}` 
*   **流水线处理**：
    1.  **Token化**：使用 Tokenizer 将文本转为数字 `input_ids`，并进行固定长度截断（`max_seq_len-2`）。
    2.  **边界填充**：序列首尾添加 `BOS` (Begin of String) 和 `EOS` (End of String)，长度不足的部分用 `PAD` 填充。
    3.  **Label对齐与掩码**：克隆 `input_ids` 作为 `labels`，并将 `labels` 中所有 `PAD` 对应标签强行赋为 `-100`，并在CrossEntropy中忽视
*   **数据载入项 (Dataset Item)**：最终返回包含 `input_ids` 和 `labels` 的字典或元组。

### 2. 训练Train
*   **解耦大/小 Batch (梯度累积)**：
    *   **Micro-Batch Size**（代码里的 `batch_size`）：受限于**硬件显存**，是单张显卡单次吞入的真实样本量。
    *   **Global-Batch Size**（全局大 Batch）：由 $\text{Micro-Batch} \times \text{Accumulation Steps} \times \text{显卡数}$ 决定，满足**算法收敛**需要的平稳梯度。
    *   **控制流**：每个 Micro-Batch 送入模型计算梯度但不更新；直到累积满指定的 Steps，多卡汇总平均梯度，触发 `optimizer.step()` 统一更新。
*   **断点续训与随机种子锁死**：
    *   每个 Epoch 开始时，单卡利用 `setup_seed(42 + epoch)`，多卡利用 `sampler.set_epoch(epoch)`。
    *   **核心目的**：将“随机洗牌”变成“可确定的随机”。确保崩溃重启后，能精准复现当前 Epoch 的数据乱序序列，配合 `SkipBatchSampler` 直接闭眼裁切掉前 $N$ 个已训练的 Step，实现无缝断点续训。

### 3. 权重保存 (Save)
*   **分布式保护**：加装 `is_main_process()` 栅栏，**仅允许主进程（GPU 0）执行写入**，防止多卡同时写同一文件导致硬盘数据损坏。
*   **模型剥壳 (Unwrapping)**：
    *   **DDP分布式壳**：若使用 `DistributedDataParallel`，需提取 `model.module`，去除权重名称前缀中的 `module.`。
    *   **编译加速壳**：若使用 `torch.compile`，需通过 `getattr(model, '_orig_mod', model)` 剥离编译外壳。
*   **双轨制保存策略**：
    1.  **部署上线权重 (`.pth`)**：纯网络参数。遍历 `state_dict` 字典，通过 `.half().cpu()` 将 FP32 压缩为 **FP16/BF16（体积缩减 50%）** 并移至内存保存。
    2.  **断点恢复全家桶 (Checkpoint)**：不仅包含模型纯权重，还完整打包了优化器状态 (`optimizer_state_dict`)、混合精度缩放器 (`scaler`)、当前运行进度 (`epoch`、`step`) 等全套控制流数据。

---

## 🛠️ 三、 核心训练技术详解 (Tos)

### 1. 动态学习率：无预热变体余弦衰减
*   **数学公式**：
    $$\eta_t = \text{lr} \times \left(0.1 + 0.45 \times \left(1 + \cos\left(\pi \times \frac{\text{current\_step}}{\text{total\_steps}}\right)\right)\right)$$
*   **运行轨迹**：利用余弦函数将学习率缩放平移。训练开始时为 $100\%$ 设定初始值；随着训练推进呈“先慢后快再变慢”的丝滑曲线跌落；**训练结束时强行维持在 $10\%$ 的底线不归零**。
*   **工业目的**：防止微调末尾阶段由于学习率彻底变 0 导致模型丧失参数修正能力，保留 $10\%$ 微弱电流有助于模型消化长尾数据并打磨细节。

### 2. 混合精度前向传播 (`autocast`)
*   **底层逻辑**：使用 `with torch.amp.autocast('cuda')` 包裹前向传播。
*   **作用**：在不牺牲模型泛化性能的前提下，自动将前向矩阵乘法由高内存的 FP32 降档为 FP16/BF16 计算。**显存吞吐量直接翻倍**，同时激活显卡硬件级 Tensor Cores 算力引擎，大幅提升训练吞吐。

### 3. 梯度防线：`scaler` 缩放与梯度裁剪 (`clip_grad_norm_`)
*   **`GradScaler` (混合精度保护)**：由于 FP16 表达范围窄，反向传播极小梯度易发生“梯度下溢”变成 0。`scaler` 在反向传播前将 Loss 整体人为放大 $2^{16}$ 倍保存精度，并在优化器更新前 `unscale_` 还原缩放。
*   **`clip_grad_norm_` (梯度裁剪)**：在优化器迈步前，检查全局梯度向量的范数（模长）。若超过阈值（如 `1.0`），则强制等比例截断至阈值范围内，**方向不变，长度锁死**，彻底杜绝因脏数据导致梯度爆炸、Loss 飙出 `NaN` 的惨剧。

### 4. 实验监控仪表盘 (`wandb`)
*   **核心价值**：大模型微调的“全时监控天眼”。
*   **实战用法**：初始化后在代码循环中通过 `wandb.log()` 异步上传每个 Step 的 `loss`、`logits_loss`、`aux_loss` 以及当前 `learning_rate`。
*   **工业红利**：脱离枯燥的终端黑窗口，在云端生成可视化的实时折线图。支持跨实验曲线对比、显存/系统功耗监控、超参数归档，是算法工程师调参优化的决策核心。