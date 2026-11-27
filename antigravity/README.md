# Antigravity 独立研究归档（四周计划一至四阶段可立即执行项全量收尾版）

> **隔离归档说明**：本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的模型配置、在线/离线一致性核验、受控消融（`A1–A5`）、背景干预机制检验（`G1–G4`）、外部公开集全矩阵补齐、190 张胡柚双盲标注与解盲管线以及完整中文论文初稿，与仓库内 **GPT** 及 **DeepSeek（DS）** 目录严格分置，互不混写。

---

## 一、第二至四阶段收尾归档（`v6_stage2to4_orchard_kit_and_paper_draft/` — 2026-11-27 新增）

- **第四阶段完整六章中文学术论文初稿（逐种子 `[S0, S1, S2]` + 均值、正负机制互证、仅留表 7 唯一解盲回填槽位）**：
  - **[v6_stage2to4_orchard_kit_and_paper_draft/docs/03_第四阶段_完整中文论文初稿_跨背景柑橘症状检测后验校准与背景抑制研究.md](v6_stage2to4_orchard_kit_and_paper_draft/docs/03_第四阶段_完整中文论文初稿_跨背景柑橘症状检测后验校准与背景抑制研究.md)**
- **第二阶段 190 张双人独立盲标、$\kappa \ge 0.70$ 门禁与第三人仲裁执行手册**：
  - **[v6_stage2to4_orchard_kit_and_paper_draft/docs/01_第二阶段_190张双人独立盲标规范_Kappa质控与仲裁操作手册.md](v6_stage2to4_orchard_kit_and_paper_draft/docs/01_第二阶段_190张双人独立盲标规范_Kappa质控与仲裁操作手册.md)**
- **可直接转发合作公司的胡柚、香柚现有数据 6 项核实函（纯净无内部批注）**：
  - **[v6_stage2to4_orchard_kit_and_paper_draft/docs/02_向合作公司核实胡柚香柚现有数据6项关键信息函.md](v6_stage2to4_orchard_kit_and_paper_draft/docs/02_向合作公司核实胡柚香柚现有数据6项关键信息函.md)**
- **190 张 / 148 采样隔离簇权威主索引与自动化质控/解盲脚本**：
  - [orchard190_148clusters_master_index.csv](v6_stage2to4_orchard_kit_and_paper_draft/configs/orchard190_148clusters_master_index.csv)（已解决 2 对跨批次同名文件碰撞，标记 `#057` 废片与 9 张马克笔图；本地 100% 盲态几何框工作包已生成于仓库外 `D:\新版柑橘\胡柚190张_双人独立标注与仲裁工作包_20261127\`）
  - [stage2_kappa_arbitration_and_coco_builder.py](v6_stage2to4_orchard_kit_and_paper_draft/scripts/stage2_kappa_arbitration_and_coco_builder.py)（双人盲标 $\kappa$ 门禁、匿名化盲态仲裁表导出与 COCO 真值构建）
  - [stage3_oneshot_unblind_and_decision_gate.py](v6_stage2to4_orchard_kit_and_paper_draft/scripts/stage3_oneshot_unblind_and_decision_gate.py)（人工仲裁真值就绪后的唯一一次 6 模型解盲评测与 Gate A/B/C 自动判决）
- **外部三大公开数据集全矩阵补齐 + 190 张 `12MP` 手机实拍无标签测速/多拍自洽 JSON 与综合配图**：
  - [stage2to4_external_and_orchard190_preunblind_complete.json](v6_stage2to4_orchard_kit_and_paper_draft/results/stage2to4_external_and_orchard190_preunblind_complete.json)
  - [fig4_external_with_negatives_and_huyou190_deployment.png](v6_stage2to4_orchard_kit_and_paper_draft/figures/fig4_external_with_negatives_and_huyou190_deployment.png)

---

## 二、第一阶段统计收尾定稿（`v5_c3_main_method_and_ablation/` — v1.1）

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

---

## 三、历史研究与方法学审计归档导航

- **`v4_orchard_candidate_arsenal/`**：前置只读可行性审计、候选探索及离线保形风险控制（CRC）/ PCR 分层送检模拟（CLI 与清单已同步至 6 模型及跨批次相对路径消歧协议）。
- **`v3_bgopta_and_dtemop/`**：前置同几何背景干预、三尺度拼接 vs 差分 Bootstrap 与检测头术语纠偏。
- **`v2_tcd_mge/` & 根级目录**：早期探索留档；权重哈希见 [MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)。
