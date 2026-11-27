# Antigravity 独立研究归档（v1.1 统计收尾定稿）：双尺度与单刀遮罩机制判决、逐种子报告与 6 模型盲测就绪

> **隔离归档说明**：本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的模型配置、在线/离线一致性核验、受控消融（`A1–A5`）、同几何背景干预机制检验（`G1–G4`）、同会话 10 模式公平测速及勘误定稿，与仓库内 **GPT** 及 **DeepSeek（DS）** 目录严格分置，互不混写。

---

## 一、第一阶段统计收尾定稿（`v5_c3_main_method_and_ablation/` — v1.1）

- **总报告 v1.1（逐种子 `[S0, S1, S2]` + 均值、零调参等权 `C3_equal` 主表与正式勘误段）**：
  - **[v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md](v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md)**
- **背景干预与“两刀 vs 单刀遮罩”判决专题报告 v1.1**：
  - **[v5_c3_main_method_and_ablation/docs/02_G1至G4灰环与模糊外环背景干预机制实验及口径修正专题报告.md](v5_c3_main_method_and_ablation/docs/02_G1至G4灰环与模糊外环背景干预机制实验及口径修正专题报告.md)**
- **预注册 6 模型盲测清单（12 份底层权重 `SHA-256` 零改动）**：
  - **[v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json](v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json)**（`YOLOv8n / Full DINOv2 / C1 / G2_blur / A4b / C3`，附 `C3_equal`）
- **叠加 3 随机种子散点的无重叠论文级配图**：
  - [fig1_c3_controlled_scale_ablation_and_bootstrap.png](v5_c3_main_method_and_ablation/figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png)
  - [fig2_c3_fair_end2end_latency_and_pareto.png](v5_c3_main_method_and_ablation/figures/fig2_c3_fair_end2end_latency_and_pareto.png)
  - [fig3_g1_to_g4_gray_and_blur_ring_mechanism.png](v5_c3_main_method_and_ablation/figures/fig3_g1_to_g4_gray_and_blur_ring_mechanism.png)

### 核心定稿结论摘要
1. **“两刀 vs 单刀遮罩”直接配对检验结论**：在复杂田间背景 `testB` 上，双尺度相对单刀框外高斯模糊遮罩 `G2_blur`（`[80.52, 77.37, 79.37]`，均值 `79.09%`）的差值为 `C3_frozen − G2_blur = +0.122%`（`p = 0.780`）与 `C3_equal − G2_blur = -0.237%`（`p = 0.554`），均不显著；按预设规则将方法节界定为“单刀模糊遮罩为最简机制实现（`1.0×` ViT），无遮罩双尺度为等效多视角增强实现（`2.0×` ViT）”，冻结协议保持不变。
2. **主表零自由度化与诚实勘误**：主表采用零调参等权 `C3_equal [0.30, 0.35, 0.35]`（`testB [80.49, 77.34, 78.67]` 均值 `78.83% > A1 78.08%, p=0.042`；`testC [85.76, 84.22, 83.76]` 均值 `84.58%`），冻结版 `C3_frozen`（`79.19% / 84.96%`）置于表注；明确标出 `DiD` 负向收缩为 `2/3` 种子现象（`DiD_gray` Seed 0 为 `+0.51pt` 无收缩），并正式修正等权口径 `DiD` 为不显著（`p = 0.122 / 0.098`）。

---

## 二、历史研究与方法学审计归档导航
- **`v4_orchard_candidate_arsenal/`**：前置只读可行性审计、候选探索及离线保形风险控制（CRC）/ PCR 分层送检模拟（CLI 与清单已同步至 6 模型协议）。
- **`v3_bgopta_and_dtemop/`**：前置同几何背景干预、三尺度拼接 vs 差分 Bootstrap 与检测头术语纠偏。
- **`v2_tcd_mge/` & 根级目录**：早期探索留档；权重哈希见 [MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)。
