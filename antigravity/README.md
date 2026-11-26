# Antigravity 独立研究归档：唯一主方法 C3 冻结定型、受控双尺度消融、G1–G4 背景机制检验与公平测速（与 GPT、DS 严格隔离）

> **隔离归档说明**：本目录（`antigravity/`）专门存放 **Antigravity** 独立完成的唯一主方法 `C3` 冻结配置、在线/离线逐框一致性核验、受控尺度消融（`A1–A5`）、受控灰环与模糊外环机制检验（`G1–G4`）、同会话公平测速及前置方法学审计，与仓库内 **GPT / 协调者**（`coordination/`、`docs/项目现状_*.md`、`docs/research/`）及 **DeepSeek（DS）**（`docs/experiments/`）完全分目录独立归档，互不混写、互不覆盖。

---

## 一、论文唯一主方法定型：`C3`（`0.82× + 1.25×` 双尺度目标表征与校准后验融合，`v5`）

- **总报告（含 Stage 1 一致性核验、Stage 2 `A1–A5` 消融、Stage 2b `G1–G4` 机制实验、同会话 `A4b` 测速与论文方法公式）**：
  - **[v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md](v5_c3_main_method_and_ablation/docs/01_C3主方法锁定_一致性核验与受控双尺度消融报告.md)**
- **Stage 2b 灰环/模糊外环机制实验与真实果园评测修复专题报告**：
  - **[v5_c3_main_method_and_ablation/docs/02_G1至G4灰环与模糊外环背景干预机制实验及口径修正专题报告.md](v5_c3_main_method_and_ablation/docs/02_G1至G4灰环与模糊外环背景干预机制实验及口径修正专题报告.md)**
- **冻结权重哈希与揭盲前预声明外部果园 5 模型清单**：
  - **[v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json](v5_c3_main_method_and_ablation/configs/c3_frozen_manifest.json)**（固化 12 份底层权重 `SHA-256`、唯一主方法 `C3 [0.30, 0.20, 0.50]` 及外部果园预声明 5 模型 `YOLOv8n / Full DINOv2 / C1 / A4b / C3`）
- **无重叠论文级配图**：
  - [fig1_c3_controlled_scale_ablation_and_bootstrap.png](v5_c3_main_method_and_ablation/figures/fig1_c3_controlled_scale_ablation_and_bootstrap.png)
  - [fig2_c3_fair_end2end_latency_and_pareto.png](v5_c3_main_method_and_ablation/figures/fig2_c3_fair_end2end_latency_and_pareto.png)
  - [fig3_g1_to_g4_gray_and_blur_ring_mechanism.png](v5_c3_main_method_and_ablation/figures/fig3_g1_to_g4_gray_and_blur_ring_mechanism.png)

### 三组核心支撑证据链摘要
1. **方法效果（`A1–A5` 受控消融 + 单独立自由度稳健性）**：`C3` 在 `testB / testC` 取得 `79.19±1.10% / 84.96±1.38%`，在 `testB` 上显著超越 `A1 (1.25×)`（`+1.14pt`, $p<0.001$）、`A2 (0.82×)`（`+1.37pt`, $p<0.001$）、相近双裁剪 `A4a (1.15×+1.35×)`（`+1.49pt`, $p<0.001$）及水平翻转 `A4b (1.25×+HFlip)`（`+1.37pt`, $p<0.001$）；零调参等权 `C3_equal [0.30, 0.35, 0.35]`（`78.83% / 84.58%`）与冻结 `C3` 差异不显著（$p \ge 0.10$）且同样显著超越单尺度基线；在当前冻结线性头框架下追加 `2.50×`（`A5`）未见稳定正收益（`-0.26pt / -0.42pt`）。
2. **机制分析（`G1–G4` 同几何灰环与模糊外环双重差分检验）**：在复杂田间多叶 `testB` 上（`1.25×` 框外面积平均占 `43.8%`），仅模糊框外高频杂叶即令单尺度显著提升 **`+1.01pt`（$p=0.024$）**，同时令引入 `0.82×` 的双尺度增益由 `+1.13pt` 显著收缩 `79%–87%`（`DiD_gray = -0.98pt`, $p<0.001$；`DiD_blur = -0.89pt`, $p=0.014$），而在无邻叶干扰的 `testC / val` 上无单尺度干预跃升，实证了背景干扰抑制机制及其场景依赖性。
3. **公平测速与外部果园评测就绪**：同会话 `RTX 4060 Ti FP32` 实测 `C3` 平均延迟 `18.98 ms (val) / 17.18 ms (testC)`，`A4b` 为 `18.67 ms / 16.78 ms`（均剔除跨协议 `TCD-MGE` 混比）；果园 CLI 已完成一对一匹配、不确定样本隔离与按簇配对 Bootstrap 加固。

---

## 二、历史研究与方法学审计归档导航

- **`v4_orchard_candidate_arsenal/`**：前置只读可行性审计、早期候选探索及离线保形风险控制（CRC）/ PCR 分层送检模拟（CLI 与冻结清单已同步至最新 5 模型协议）。
- **`v3_bgopta_and_dtemop/`**：前置同几何背景干预、三尺度拼接 vs 差分逐图 Bootstrap 与检测头术语纠偏（含 `TCD-MGE` 独立协议测速历史记录）。
- **`v2_tcd_mge/` & 根级目录**：早期多粒度门控探索留档；本地全部权重哈希清单见 [MODEL_CHECKPOINTS_MANIFEST.csv](MODEL_CHECKPOINTS_MANIFEST.csv)。
