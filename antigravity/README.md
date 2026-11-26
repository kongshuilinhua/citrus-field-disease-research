# Antigravity 独立研究归档：主方法 C3 冻结定型、受控双尺度消融与历史方法学审计（与 GPT、DS 严格隔离）

> **隔离归档说明**：本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的主方法 `C3` 冻结配置、在线/离线逐框一致性核验、受控尺度消融（`A1–A5`）、公平测速及前置方法学审计，与仓库内 **GPT / 协调者**（`coordination/`、`docs/项目现状_*.md`、`docs/research/`）及 **DeepSeek（DS）**（`docs/experiments/`）完全分目录独立归档，互不混写、互不覆盖。

---

## 一、论文唯一主方法定型：`C3`（`0.82× + 1.25×` 双尺度目标表征与校准后验融合，`v5` 冻结版）

- **主报告（含 Stage 1 在线/离线核验、Stage 2 受控消融 `A1–A5` 与论文方法章节数学公式）**：
  - **[v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md](v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md)**
- **冻结权重哈希与源域锁定超参清单**：
  - **[v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json](v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json)**（固化全部 12 份底层权重 `SHA-256`、`C3` 融合权重 `[0.30, 0.20, 0.50]` 及真实果园唯一评测四模型 `YOLOv8n / Full DINOv2 / C1 / C3`）
- **论文级受控消融与公平测速配图**：
  - [fig1_c3_controlled_scale_ablation_and_bootstrap.png](v5_c3_main_method_and_ablation/figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png)（严格固定 `w_det = 0.30` 受控尺度消融 `A1–A5` & `B = 1,000` 逐图配对 Bootstrap 95% 置信区间森林图）
  - [fig2_c3_fair_end2end_latency_and_pareto.png](v5_c3_main_method_and_ablation/figures/fig2_c3_fair_end2end_latency_and_pareto.png)（含多目标场景 `≥3框` 与 `P95` 尾部延迟的端到端公平测速及帕累托权衡）

### 核心证据链摘要
1. **Stage 1 在线与离线推理 100% 一致**：修复单图 `auto=True` 矩形缩放与 `non_max_suppression` 原地改写 `xywh` 隐患后，3 种子 × 3 划分共抽检 800 个前景框**类别不一致数 = `0`**（最大后验绝对误差 `≤ 5e-6`），端到端实时 `AP50–95` 与离线结果完全吻合（`val 90.43% / testB 79.19% / testC 84.96%`）。
2. **Stage 2 受控尺度消融（固定 `w_det = 0.30`、零额外 MLP 参数、`B = 1,000` 配对 Bootstrap）**：
   - **优于任一单尺度**：在 `testB / testC` 上，`A3/C3 (0.82×+1.25×)` 显著优于标准单尺度 `A1 (1.25×)`（`+1.14pt`, $p<0.001$ / `+0.65pt`, $p=0.044$；对比历史 `C1` 为 `+0.92pt`, $p=0.002$ / `+1.12pt`, $p<0.001$）及紧致单尺度 `A2 (0.82×)`（`+1.37pt`, $p<0.001$ / `+2.47pt`, $p<0.001$）；
   - **排除普通双视角 TTA 解释**：在复杂自然田间背景 `testB` 上，等算力相近双裁剪 `A4a (1.15×+1.35×)`（`77.70%`）与水平翻转 `A4b (1.25×+HFlip)`（`77.80%`）因无法剥离框内邻叶干扰，甚至未超过单框 `A1`（`78.08%`），`C3` 较 `A4a` 显著提升 **`+1.49pt`（`95% CI [+0.76%, +2.29%]`, $p<0.001$；逐种子 `val` 选权亦提升 `+1.23pt`, $p=0.002$）**、较 `A4b` 提升 **`+1.37pt`（$p<0.001$）**；同时诚实报告在无杂叶纯白板 `testC` 上 `C3` 仍优于 `A4a`（`+0.96pt`, $p=0.010$），与 `A4b` 差距缩小（`+0.30pt`, $p=0.414$）；
   - **证明继续扩大背景无收益**：三尺度 `A5 (0.82×+1.25×+2.50×)` 相对 `C3` 变化为 **`-0.26pt` ($p=0.284$) / `-0.42pt` ($p=0.072$)**，多目标场景延迟却从 `30.04 ms` 增至 `41.27 ms`（慢 `37.4%`）。

---

## 二、历史研究与方法学审计归档导航

- **`v4_orchard_candidate_arsenal/`**：前置只读可行性审计（证伪源域复杂逐框学习门控）、两层八候选探索库及离线保形风险控制（CRC）/ PCR 分层送检模拟（其 CLI 已同步更新至 Stage 1 冻结清单）。
- **`v3_bgopta_and_dtemop/`**：同几何背景干预实验（分离分辨率缩水与背景干扰）、三尺度拼接 vs 差分逐图 Bootstrap、检测头术语纠偏与负结果完整披露。
- **`v2_tcd_mge/` & 根级目录**：早期多粒度门控探索留档；本地全部权重 `SHA-256` 清单见 [MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)。
